"""3D Studio — multi-model image/text-to-3D web app.

Pipelines (all hot-loaded at startup, one per GPU):
  cuda:0  AniGen                 — rigged textured GLB      (~8 GB)
  cuda:1  HunyuanDiT (full)      — text -> image, bilingual (~9 GB)
  cuda:2  Hunyuan3D shape        — image -> static mesh     (~5.5 GB)
  cuda:3  Hunyuan3D paint        — textures the mesh        (~7 GB)

HunyuanDiT is the sole T2I backend. We bypass hy3dgen's thin wrapper (which
truncates at 60 chars and hardcodes a Chinese style suffix) and call the
underlying diffusers PAG pipeline directly with our English pose-focused
prompt + negative at guidance 7.5 / pag 2.0 / 35 steps.

Port 9000, bound to 0.0.0.0 for LAN access.
"""
import asyncio
import gc
import io
import json as _json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

import torch
# --- transformers compat patches --------------------------------------------
# (1) transformers >=5.5 refuses to load .bin (torch.load) checkpoints unless
#     torch>=2.6 (CVE-2025-32434). We pin torch 2.5.1 for xformers ABI compat
#     and every .bin we load is from pre-fetched official HF weights on local
#     disk, not user input. Patch the check.
# (2) transformers 5.5 internally accesses `self.all_tied_weights_keys` on
#     custom remote-code models (e.g. briaai/RMBG-1.4) that predate this API.
#     Add a class-level default so subclasses without it don't crash.
try:
    import transformers.utils.import_utils as _tui
    _tui.check_torch_load_is_safe = lambda: None
    import transformers.modeling_utils as _tmu
    _tmu.check_torch_load_is_safe = lambda: None
    if not hasattr(_tmu.PreTrainedModel, "all_tied_weights_keys"):
        _tmu.PreTrainedModel.all_tied_weights_keys = {}
except Exception as _e:
    print(f"[startup] transformers patch skipped: {_e}")
# ----------------------------------------------------------------------------
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVER_DIR))

from anigen.pipelines import AnigenImageTo3DPipeline  # noqa: E402
from anigen.utils.ckpt_utils import ensure_ckpts  # noqa: E402
from anigen.utils.random_utils import set_random_seed  # noqa: E402
from prompt_harness import (  # noqa: E402
    DEFAULT_STYLE,
    enrich_prompt,
    list_styles,
)

RESULTS_DIR = REPO_ROOT / "results" / "web"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = SERVER_DIR / "static"

ANIGEN_DEVICE = "cuda:0"
HUNYUANDIT_DEVICE = "cuda:1"
HUNYUANDIT_DEVICE_IDX = 1
HY3D_SHAPE_DEVICE = "cuda:2"
HY3D_SHAPE_DEVICE_IDX = 2
HY3D_PAINT_DEVICE = "cuda:3"
HY3D_PAINT_DEVICE_IDX = 3
ZERO123_DEVICE = "cuda:3"  # same card as paint — sequential use, no collision
ZERO123_DEVICE_IDX = 3

HUNYUANDIT_MODEL = "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers"
HUNYUANDIT_CONTROLNET_POSE = "Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Pose"
POSES_DIR = Path(__file__).resolve().parent / "static" / "poses"

app = FastAPI(title="3D Studio")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

jobs: dict[str, dict] = {}
gen_lock = asyncio.Lock()

anigen_pipe = None
hunyuandit_pipe = None  # HunyuanDiT (full, undistilled) on cuda:1
hunyuandit_cn_pipe = None  # HunyuanDiTControlNetPipeline for pose conditioning
hy3d_shape_pipe = None
hy3d_mv_pipe = None  # tencent/Hunyuan3D-2mv multi-view variant, cuda:2
hy3d_paint_pipe = None
zero123_pipe = None  # sudo-ai/zero123plus-v1.2 — novel view synthesis on cuda:3
bg_remover = None  # hy3dgen.rembg.BackgroundRemover — applied before Hunyuan3D
hy3d_error = None
hy3d_mv_error = None
hunyuandit_error = None
hunyuandit_cn_error = None
zero123_error = None


@app.on_event("startup")
async def startup():
    global anigen_pipe, hunyuandit_pipe, hunyuandit_cn_pipe, hy3d_shape_pipe, hy3d_mv_pipe, hy3d_paint_pipe, bg_remover, zero123_pipe
    global hy3d_error, hy3d_mv_error, hunyuandit_error, hunyuandit_cn_error, zero123_error

    print("[startup] ensure_ckpts")
    ensure_ckpts()

    print("[startup] loading RMBG-1.4 background remover on cuda:2")
    t0 = time.time()
    try:
        from transformers import pipeline as _hf_pipeline
        rmbg_pipe = _hf_pipeline(
            "image-segmentation",
            model="briaai/RMBG-1.4",
            trust_remote_code=True,
        )
        # Pipeline's device arg is inconsistent across versions; move the model
        # explicitly and update the stored device for inference.
        rmbg_pipe.model.to(HY3D_SHAPE_DEVICE)
        rmbg_pipe.device = torch.device(HY3D_SHAPE_DEVICE)

        class _RMBGWrapper:
            """Callable wrapper so callers stay device-agnostic."""
            def __init__(self, pipe):
                self.pipe = pipe
            def __call__(self, image):
                with torch.cuda.device(HY3D_SHAPE_DEVICE_IDX):
                    return self.pipe(image)

        bg_remover = _RMBGWrapper(rmbg_pipe)
        print(f"[startup] RMBG-1.4 ready in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"[startup] RMBG-1.4 FAILED, falling back to hy3dgen.rembg: {type(e).__name__}: {e}")
        traceback.print_exc()
        try:
            from hy3dgen.rembg import BackgroundRemover
            bg_remover = BackgroundRemover()
            print(f"[startup] hy3dgen BackgroundRemover fallback ready")
        except Exception as e2:
            print(f"[startup] fallback also FAILED: {type(e2).__name__}: {e2}")

    print(f"[startup] loading AniGen on {ANIGEN_DEVICE} (use_ema=True)")
    t0 = time.time()
    anigen_pipe = AnigenImageTo3DPipeline.from_pretrained(
        ss_flow_path="ckpts/anigen/ss_flow_duet",
        slat_flow_path="ckpts/anigen/slat_flow_auto",
        device=ANIGEN_DEVICE,
        use_ema=True,
    )
    anigen_pipe.to(torch.device(ANIGEN_DEVICE))
    print(f"[startup] AniGen ready in {time.time()-t0:.1f}s")

    print(f"[startup] loading HunyuanDiT ControlNet-Pose on {HUNYUANDIT_DEVICE}")
    t0 = time.time()
    try:
        from diffusers import HunyuanDiT2DControlNetModel, HunyuanDiTControlNetPipeline
        with torch.cuda.device(HUNYUANDIT_DEVICE_IDX):
            cn = HunyuanDiT2DControlNetModel.from_pretrained(
                HUNYUANDIT_CONTROLNET_POSE, torch_dtype=torch.float16,
            )
            hunyuandit_cn_pipe = HunyuanDiTControlNetPipeline.from_pretrained(
                HUNYUANDIT_MODEL,
                controlnet=cn,
                torch_dtype=torch.float16,
            )
            hunyuandit_cn_pipe.to(HUNYUANDIT_DEVICE)
        # hunyuandit_pipe also points at the ControlNet pipe — single source of
        # T2I. When no pose is selected we pass controlnet_conditioning_scale=0
        # to make the CN layer a no-op.
        hunyuandit_pipe = hunyuandit_cn_pipe
        print(f"[startup] HunyuanDiT+ControlNet ready in {time.time()-t0:.1f}s")
    except Exception as e:
        hunyuandit_cn_error = f"{type(e).__name__}: {e}"
        hunyuandit_error = hunyuandit_cn_error
        print(f"[startup] HunyuanDiT+ControlNet FAILED: {hunyuandit_cn_error}")
        traceback.print_exc()

    try:
        print(f"[startup] loading Hunyuan3D-2 shape on {HY3D_SHAPE_DEVICE}")
        t0 = time.time()
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        with torch.cuda.device(HY3D_SHAPE_DEVICE_IDX):
            hy3d_shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                "tencent/Hunyuan3D-2",
                device=HY3D_SHAPE_DEVICE,
            )
        print(f"[startup] Hunyuan3D-2 shape ready in {time.time()-t0:.1f}s")
    except Exception as e:
        hy3d_error = f"shape_load: {type(e).__name__}: {e}"
        print(f"[startup] Hunyuan3D-2 shape FAILED: {hy3d_error}")
        traceback.print_exc()

    try:
        print(f"[startup] loading Hunyuan3D-2mv (multi-view) on {HY3D_SHAPE_DEVICE}")
        t0 = time.time()
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        with torch.cuda.device(HY3D_SHAPE_DEVICE_IDX):
            hy3d_mv_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                "tencent/Hunyuan3D-2mv",
                subfolder="hunyuan3d-dit-v2-mv",
                use_safetensors=True,
                device=HY3D_SHAPE_DEVICE,
            )
        print(f"[startup] Hunyuan3D-2mv ready in {time.time()-t0:.1f}s")
    except Exception as e:
        hy3d_mv_error = f"{type(e).__name__}: {e}"
        print(f"[startup] Hunyuan3D-2mv FAILED: {hy3d_mv_error}")
        traceback.print_exc()

    try:
        print(f"[startup] loading Zero123++ (novel-view) on {ZERO123_DEVICE}")
        t0 = time.time()
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
        with torch.cuda.device(ZERO123_DEVICE_IDX):
            zero123_pipe = DiffusionPipeline.from_pretrained(
                "sudo-ai/zero123plus-v1.2",
                custom_pipeline="sudo-ai/zero123plus-pipeline",
                torch_dtype=torch.float16,
            )
            zero123_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                zero123_pipe.scheduler.config, timestep_spacing="trailing",
            )
            zero123_pipe.to(ZERO123_DEVICE)
        print(f"[startup] Zero123++ ready in {time.time()-t0:.1f}s")
    except Exception as e:
        zero123_error = f"{type(e).__name__}: {e}"
        print(f"[startup] Zero123++ FAILED: {zero123_error}")
        traceback.print_exc()

    if hy3d_shape_pipe is not None:
        try:
            print(f"[startup] loading Hunyuan3D-2 paint on {HY3D_PAINT_DEVICE}")
            t0 = time.time()
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            # Paint pipeline ignores a 'device' kwarg and defaults its sub-models to
            # cuda:0. Force cuda:3 as the default device during load so all sub-
            # components (multiview diffusion, delight, rasterizer) allocate there.
            with torch.cuda.device(HY3D_PAINT_DEVICE_IDX):
                hy3d_paint_pipe = Hunyuan3DPaintPipeline.from_pretrained(
                    "tencent/Hunyuan3D-2",
                )
                _move_paint_to_device(hy3d_paint_pipe, HY3D_PAINT_DEVICE)
            print(f"[startup] Hunyuan3D-2 paint ready in {time.time()-t0:.1f}s")
        except Exception as e:
            hy3d_error = (hy3d_error + " | " if hy3d_error else "") + f"paint_load: {type(e).__name__}: {e}"
            print(f"[startup] Hunyuan3D-2 paint FAILED: {hy3d_error}")
            traceback.print_exc()

    print("[startup] READY")


class GenerateRequest(BaseModel):
    prompt: str
    model: str = "anigen"  # "anigen" | "hunyuan3d" | "hunyuan3dmv"
    style: str = DEFAULT_STYLE
    pose: str = ""  # id of a pose from the gallery, empty = no pose constraint
    settings: dict = {}


# Defaults applied when a settings field is missing. Kept in one place so the
# /api/defaults endpoint can hand them to the UI on first load.
DEFAULT_SETTINGS = {
    "t2i": {
        "custom_suffix": "",
        "custom_negative": "",
        "guidance_scale": 7.5,
        "pag_scale": 2.0,
        "num_inference_steps": 35,
    },
    "anigen": {
        "cfg_scale_ss": 8.0,
        "cfg_scale_slat": 3.5,
        "joints_density": 2,
        "smooth_skin_weights": True,
        "smooth_skin_weights_iters": 100,
        "smooth_skin_weights_alpha": 1.0,
        "filter_skin_weights": True,
    },
    "hunyuan3d": {
        "enable_texture": True,
        "enable_bg_removal": True,
        "max_polygons": 120_000,
    },
}


def _get(settings: dict, section: str, key: str):
    """Read settings[section][key] with fallback to DEFAULT_SETTINGS."""
    val = (settings or {}).get(section, {}).get(key)
    if val is None or val == "":
        return DEFAULT_SETTINGS[section][key]
    return val


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {
        "models": {
            "anigen": {
                "ready": anigen_pipe is not None,
                "label": "AniGen — rigged + animatable",
                "rigged": True,
            },
            "hunyuan3d": {
                "ready": hy3d_shape_pipe is not None,
                "textured": hy3d_paint_pipe is not None,
                "label": "Hunyuan3D-2 — textured mesh (static)",
                "rigged": False,
                "error": hy3d_error,
            },
            "hunyuan3dmv": {
                "ready": hy3d_mv_pipe is not None,
                "textured": hy3d_paint_pipe is not None,
                "label": "Hunyuan3D-2mv — multi-view textured mesh",
                "rigged": False,
                "multiview": True,
                "zero123_ready": zero123_pipe is not None,
                "error": hy3d_mv_error,
            },
            "t2i": {
                "ready": hunyuandit_pipe is not None,
                "label": "HunyuanDiT v1.1 (full)",
                "error": hunyuandit_error,
            },
        },
        "styles": list_styles(),
        "active_jobs": sum(1 for j in jobs.values() if j["status"] not in ("done", "error")),
        "total_jobs": len(jobs),
    }


@app.post("/api/enrich")
async def api_enrich(req: GenerateRequest):
    positive, negative, applied_style, aspect = enrich_prompt(req.prompt, req.style)
    # Apply user custom overrides
    custom_suffix = (req.settings.get("t2i", {}).get("custom_suffix") or "").strip()
    if custom_suffix:
        positive = f"{positive}, {custom_suffix}"
    custom_neg = (req.settings.get("t2i", {}).get("custom_negative") or "").strip()
    if custom_neg:
        negative = custom_neg
    return {
        "enriched": positive,
        "negative": negative,
        "style": applied_style,
        "aspect": aspect,
        "t2i": "hunyuandit",
    }


@app.get("/api/defaults")
async def api_defaults():
    return DEFAULT_SETTINGS


@app.get("/api/poses")
async def api_poses():
    """List all available pose-gallery entries for the UI picker."""
    try:
        from pose_gallery import POSES
        items = []
        for key, (label, _fn) in POSES.items():
            path = POSES_DIR / f"{key}.png"
            if path.exists():
                items.append({"key": key, "label": label, "url": f"/static/poses/{key}.png"})
        return {"poses": items, "ready": hunyuandit_cn_pipe is not None}
    except Exception as e:
        return {"poses": [], "ready": False, "error": f"{type(e).__name__}: {e}"}


@app.post("/api/cancel/{job_id}")
async def api_cancel(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "no such job")
    if jobs[job_id]["status"] in ("done", "error", "cancelled"):
        return {"ok": False, "note": "job already finished"}
    jobs[job_id]["cancel_requested"] = True
    return {"ok": True, "note": "cancel will apply at next stage boundary"}


def _is_cancelled(job_id: str) -> bool:
    return bool(jobs.get(job_id, {}).get("cancel_requested"))


@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(400, "empty prompt")
    if hunyuandit_pipe is None:
        raise HTTPException(503, f"t2i not ready: {hunyuandit_error}")
    model = req.model.lower()
    if model == "anigen":
        if anigen_pipe is None:
            raise HTTPException(503, "anigen not ready")
    elif model == "hunyuan3d":
        if hy3d_shape_pipe is None:
            raise HTTPException(503, f"hunyuan3d not ready: {hy3d_error}")
    elif model == "hunyuan3dmv":
        if hy3d_mv_pipe is None:
            raise HTTPException(503, f"hunyuan3dmv not ready: {hy3d_mv_error}")
        if zero123_pipe is None:
            raise HTTPException(503, f"zero123++ not ready (needed to auto-generate views): {zero123_error}")
    else:
        raise HTTPException(400, f"unknown model: {req.model}")

    positive, negative, style, aspect = enrich_prompt(prompt, req.style)
    # Apply custom overrides on top of style preset
    t2i_cfg = (req.settings or {}).get("t2i", {})
    custom_suffix = (t2i_cfg.get("custom_suffix") or "").strip()
    if custom_suffix:
        positive = f"{positive}, {custom_suffix}"
    custom_neg = (t2i_cfg.get("custom_negative") or "").strip()
    if custom_neg:
        negative = custom_neg

    pose_id = (req.pose or "").strip()
    if pose_id and not (POSES_DIR / f"{pose_id}.png").exists():
        pose_id = ""

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "status": "queued",
        "prompt": prompt,
        "enriched_prompt": positive,
        "negative_prompt": negative,
        "style": style,
        "aspect": aspect,
        "model": model,
        "t2i": "hunyuandit",
        "pose": pose_id,
        "settings": req.settings,
        "created": time.time(),
    }
    asyncio.create_task(run_job(job_id, positive, negative, aspect, model, req.settings, pose_id))
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def api_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "no such job")
    return jobs[job_id]


@app.post("/api/generate_image")
async def api_generate_image(
    image: UploadFile = File(...),
    model: str = Form("anigen"),
    settings: str = Form("{}"),
    note: str = Form(""),
):
    """Skip T2I — take an uploaded image directly to the selected 3D model."""
    if image is None:
        raise HTTPException(400, "no image uploaded")
    model = model.lower()
    if model == "anigen":
        if anigen_pipe is None:
            raise HTTPException(503, "anigen not ready")
    elif model == "hunyuan3d":
        if hy3d_shape_pipe is None:
            raise HTTPException(503, f"hunyuan3d not ready: {hy3d_error}")
    elif model == "hunyuan3dmv":
        if hy3d_mv_pipe is None:
            raise HTTPException(503, f"hunyuan3dmv not ready: {hy3d_mv_error}")
        if zero123_pipe is None:
            raise HTTPException(503, f"zero123++ not ready: {zero123_error}")
    else:
        raise HTTPException(400, f"unknown model: {model}")

    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"could not decode image: {e}")

    try:
        settings_dict = _json.loads(settings) if settings else {}
        if not isinstance(settings_dict, dict):
            settings_dict = {}
    except Exception:
        settings_dict = {}

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "status": "queued",
        "prompt": note.strip() or "(uploaded image)",
        "enriched_prompt": None,
        "negative_prompt": None,
        "style": None,
        "aspect": None,
        "model": model,
        "t2i": None,
        "source": "upload",
        "input_filename": image.filename,
        "settings": settings_dict,
        "created": time.time(),
    }
    asyncio.create_task(run_job_from_image(job_id, pil, model, settings_dict))
    return {"job_id": job_id}


@app.post("/api/generate_views")
async def api_generate_views(
    image: UploadFile = File(...),
    seed: int = Form(0),
):
    """Stateless Zero123++ preview — run on a front image and return URLs to
    the 3 synthesized views. Client then resubmits them to
    /api/generate_multiview as regular uploads."""
    if zero123_pipe is None:
        raise HTTPException(503, f"zero123++ not ready: {zero123_error}")
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    try:
        front = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"could not decode image: {e}")

    preview_id = uuid.uuid4().hex[:12]
    out_dir = RESULTS_DIR / f"preview_{preview_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    views = await asyncio.to_thread(_zero123_extend, front, int(seed), out_dir)
    front.save(out_dir / "front.png")
    urls = {"front": f"/results/preview_{preview_id}/front.png"}
    for name, img in views.items():
        img.save(out_dir / f"{name}.png")
        urls[name] = f"/results/preview_{preview_id}/{name}.png"
    return {"preview_id": preview_id, "urls": urls}


@app.post("/api/generate_multiview")
async def api_generate_multiview(
    image_front: UploadFile = File(...),
    image_back: UploadFile | None = File(None),
    image_left: UploadFile | None = File(None),
    image_right: UploadFile | None = File(None),
    settings: str = Form("{}"),
    note: str = Form(""),
):
    """Multi-view image -> Hunyuan3D-2mv shape (+ optional paint)."""
    if hy3d_mv_pipe is None:
        raise HTTPException(503, f"hunyuan3dmv not ready: {hy3d_mv_error}")

    async def _read(up: UploadFile | None):
        if up is None:
            return None
        raw = await up.read()
        if not raw:
            return None
        try:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None

    views = {
        "front": await _read(image_front),
        "back": await _read(image_back),
        "left": await _read(image_left),
        "right": await _read(image_right),
    }
    views = {k: v for k, v in views.items() if v is not None}
    if "front" not in views:
        raise HTTPException(400, "front view is required")

    try:
        settings_dict = _json.loads(settings) if settings else {}
        if not isinstance(settings_dict, dict):
            settings_dict = {}
    except Exception:
        settings_dict = {}

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "status": "queued",
        "prompt": note.strip() or f"(multi-view: {', '.join(views.keys())})",
        "model": "hunyuan3dmv",
        "t2i": None,
        "source": "multiview",
        "views": list(views.keys()),
        "settings": settings_dict,
        "created": time.time(),
    }
    asyncio.create_task(run_job_multiview(job_id, views, settings_dict))
    return {"job_id": job_id}


async def run_job_multiview(job_id: str, views: dict, settings: dict):
    try:
        async with gen_lock:
            if _is_cancelled(job_id):
                jobs[job_id]["status"] = "cancelled"
                return
            seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
            jobs[job_id]["seed"] = seed
            job_dir = RESULTS_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            for name, img in views.items():
                img.save(job_dir / f"input_{name}.png")
            # Use the front view as the main preview for the UI
            jobs[job_id]["preview_url"] = f"/results/{job_id}/input_front.png"

            if _is_cancelled(job_id):
                jobs[job_id]["status"] = "cancelled"
                return

            # If uploaded views are incomplete (only front or front+one), fill with Zero123++
            if set(views.keys()) < {"front", "back", "left", "right"} and zero123_pipe is not None:
                jobs[job_id]["status"] = "views"
                gen_views = await asyncio.to_thread(_zero123_extend, views["front"], seed, job_dir)
                for k, v in gen_views.items():
                    if k not in views:
                        views[k] = v
                        v.save(job_dir / f"view_{k}.png")
                jobs[job_id]["views_urls"] = {k: f"/results/{job_id}/view_{k}.png" for k in views if (job_dir / f"view_{k}.png").exists() or k == "front"}

            jobs[job_id]["status"] = "mesh"
            glb_path = job_dir / "mesh.glb"
            await asyncio.to_thread(_hy3d_mv, views, str(glb_path), seed, job_id, settings)
            jobs[job_id]["glb_url"] = f"/results/{job_id}/mesh.glb"
            jobs[job_id]["status"] = "done"
            jobs[job_id]["finished"] = time.time()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[job {job_id}] ERROR: {tb}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = f"{type(e).__name__}: {e}"


def _prep_for_zero123(img: Image.Image, size: int = 512, save_path: str = None) -> Image.Image:
    """Condition image for Zero123++: bg-remove, find subject bbox (alpha OR
    luminance fallback), pad square, resize. Non-conforming inputs produce
    distorted novel views, so we're strict about centering."""
    import numpy as np

    # 1. Background removal
    if bg_remover is not None:
        try:
            rgba = bg_remover(img.convert("RGBA") if img.mode != "RGBA" else img)
        except Exception as e:
            print(f"[zero123 prep] bg_remover failed, using raw: {e}")
            rgba = img.convert("RGBA")
    else:
        rgba = img.convert("RGBA")
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")

    # 2. Find foreground. Use alpha if it actually encodes foreground
    #    (non-trivial channel), else fall back to luminance (non-white = subject).
    alpha_np = np.array(rgba.split()[-1])
    alpha_range = int(alpha_np.max()) - int(alpha_np.min())
    if alpha_range > 50:
        fg_mask = alpha_np > 20
        reason = "alpha"
    else:
        lum = np.array(rgba.convert("L"))
        fg_mask = lum < 245  # anything not near-white is subject
        reason = "luminance"

    ys, xs = np.where(fg_mask)
    if len(xs) and len(ys):
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    else:
        x0, y0, x1, y1 = 0, 0, rgba.width, rgba.height
    print(f"[zero123 prep] bbox={x0,y0,x1,y1} from={reason} img={rgba.size}")
    rgba = rgba.crop((x0, y0, x1, y1))

    # 3. Pad to square with ~10% margin.
    w, h = rgba.size
    margin = int(max(w, h) * 0.10)
    side = max(w, h) + 2 * margin
    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 255))
    canvas.paste(rgba, ((side - w) // 2, (side - h) // 2), rgba)

    # 4. Resize to Zero123's cond size.
    result = canvas.resize((size, size), Image.LANCZOS).convert("RGB")
    if save_path:
        try:
            result.save(save_path)
        except Exception as e:
            print(f"[zero123 prep] save failed: {e}")
    return result


def _zero123_extend(front_img, seed: int = 0, save_dir=None):
    """Take a front PIL image, synthesize back/left/right via Zero123++.

    Zero123++ v1.2 outputs a 3x2 grid of 6 views at azimuths 30/90/150/210/270/330
    and alternating elevations 20/-10. We pick the 90° (right), 210° (≈back),
    and 270° (left) tiles for Hy3D-2mv.
    """
    if zero123_pipe is None:
        return {}
    cond_save = str(save_dir / "zero123_cond.png") if save_dir else None
    cond = _prep_for_zero123(front_img, size=512, save_path=cond_save)
    print(f"[zero123] synthesizing novel views (cond={cond.size})")
    t0 = time.time()
    with torch.cuda.device(ZERO123_DEVICE_IDX):
        gen = torch.Generator(device=ZERO123_DEVICE).manual_seed(int(seed))
        grid = zero123_pipe(cond, num_inference_steps=36, generator=gen).images[0]
    if save_dir:
        try:
            grid.save(save_dir / "zero123_grid.png")
        except Exception as e:
            print(f"[zero123] grid save failed: {e}")
    # Zero123++ output is a 2-column × 3-row grid (tall), NOT 3x2.
    # Row-major azimuths: (0,0)=30°, (1,0)=90°, (0,1)=150°, (1,1)=210°, (0,2)=270°, (1,2)=330°
    w, h = grid.size
    tw, th = w // 2, h // 3
    tiles = []
    for row in range(3):
        for col in range(2):
            tiles.append(grid.crop((col * tw, row * th, (col + 1) * tw, (row + 1) * th)))
    # tiles[1] = az 90° (right), tiles[3] = az 210° (≈back), tiles[4] = az 270° (left)
    out = {"right": tiles[1], "back": tiles[3], "left": tiles[4]}
    print(f"[zero123] done in {time.time()-t0:.1f}s grid={grid.size} tile={tw}x{th}")
    gc.collect()
    torch.cuda.empty_cache()
    return out


def _hy3d_mv(views: dict, output_glb: str, seed: int, job_id: str, settings: dict):
    settings = settings or {}
    enable_bg = bool(_get(settings, "hunyuan3d", "enable_bg_removal"))
    enable_tex = bool(_get(settings, "hunyuan3d", "enable_texture"))
    max_faces = int(_get(settings, "hunyuan3d", "max_polygons"))

    # Remove background from each view independently before shape
    if bg_remover is not None and enable_bg:
        cleaned = {}
        for k, img in views.items():
            try:
                cleaned[k] = bg_remover(img.convert("RGBA") if img.mode != "RGBA" else img)
            except Exception as e:
                print(f"[hy3dmv] bg_remover failed on {k}: {e}; using raw")
                cleaned[k] = img
        views = cleaned
        print(f"[hy3dmv] bg removed on {len(views)} views")

    print(f"[hy3dmv] running mv shape with views: {list(views.keys())}")
    with torch.cuda.device(HY3D_SHAPE_DEVICE_IDX):
        gen = torch.Generator(device=HY3D_SHAPE_DEVICE).manual_seed(int(seed))
        try:
            meshes = hy3d_mv_pipe(
                image=views,
                num_inference_steps=30,
                octree_resolution=380,
                num_chunks=20000,
                generator=gen,
                output_type="trimesh",
            )
        except TypeError:
            meshes = hy3d_mv_pipe(image=views)
    mesh = meshes[0] if isinstance(meshes, (list, tuple)) else meshes

    mesh = _hy3d_simplify(mesh, max_faces)

    if job_id and _is_cancelled(job_id):
        print(f"[job {job_id}] cancel requested, skipping paint")
        mesh.export(output_glb)
        gc.collect()
        torch.cuda.empty_cache()
        return

    if hy3d_paint_pipe is not None and enable_tex:
        try:
            # Paint uses a single reference image — use the front view
            ref_img = views["front"] if "front" in views else next(iter(views.values()))
            with torch.cuda.device(HY3D_PAINT_DEVICE_IDX):
                mesh = hy3d_paint_pipe(mesh, image=ref_img)
        except Exception as e:
            print(f"[hy3dmv] paint failed: {e}")
            traceback.print_exc()
    elif not enable_tex:
        print(f"[hy3dmv] texture disabled")

    mesh.export(output_glb)
    gc.collect()
    torch.cuda.empty_cache()


async def run_job_from_image(job_id: str, pil_img, model: str, settings: dict):
    try:
        async with gen_lock:
            if _is_cancelled(job_id):
                jobs[job_id]["status"] = "cancelled"
                return

            seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
            jobs[job_id]["seed"] = seed
            job_dir = RESULTS_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            pil_img.save(job_dir / "input.png")
            jobs[job_id]["preview_url"] = f"/results/{job_id}/input.png"

            if _is_cancelled(job_id):
                jobs[job_id]["status"] = "cancelled"
                return

            glb_path = job_dir / "mesh.glb"
            if model == "anigen":
                jobs[job_id]["status"] = "mesh"
                await asyncio.to_thread(_anigen, pil_img, str(glb_path), seed, settings)
                skel_path = job_dir / "skeleton.glb"
                if skel_path.exists():
                    jobs[job_id]["skeleton_url"] = f"/results/{job_id}/skeleton.glb"
            elif model == "hunyuan3d":
                jobs[job_id]["status"] = "mesh"
                await asyncio.to_thread(_hy3d, pil_img, str(glb_path), seed, job_id, settings)
            elif model == "hunyuan3dmv":
                jobs[job_id]["status"] = "views"
                views = await asyncio.to_thread(_zero123_extend, pil_img, seed, job_dir)
                views["front"] = pil_img
                for name, vimg in views.items():
                    vimg.save(job_dir / f"view_{name}.png")
                jobs[job_id]["views_urls"] = {k: f"/results/{job_id}/view_{k}.png" for k in views}
                jobs[job_id]["status"] = "mesh"
                await asyncio.to_thread(_hy3d_mv, views, str(glb_path), seed, job_id, settings)

            jobs[job_id]["glb_url"] = f"/results/{job_id}/mesh.glb"
            jobs[job_id]["status"] = "done"
            jobs[job_id]["finished"] = time.time()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[job {job_id}] ERROR: {tb}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = f"{type(e).__name__}: {e}"


async def run_job(job_id: str, positive_prompt: str, negative_prompt: str, aspect: str, model: str, settings: dict, pose_id: str = ""):
    try:
        async with gen_lock:
            if _is_cancelled(job_id):
                jobs[job_id]["status"] = "cancelled"
                return

            jobs[job_id]["status"] = "image"
            seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
            jobs[job_id]["seed"] = seed
            job_dir = RESULTS_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            img = await asyncio.to_thread(_t2i_hunyuandit, positive_prompt, seed, negative_prompt, aspect, settings, pose_id)
            img.save(job_dir / "input.png")
            jobs[job_id]["preview_url"] = f"/results/{job_id}/input.png"

            if _is_cancelled(job_id):
                jobs[job_id]["status"] = "cancelled"
                return

            glb_path = job_dir / "mesh.glb"
            if model == "anigen":
                jobs[job_id]["status"] = "mesh"
                await asyncio.to_thread(_anigen, img, str(glb_path), seed, settings)
                skel_path = job_dir / "skeleton.glb"
                if skel_path.exists():
                    jobs[job_id]["skeleton_url"] = f"/results/{job_id}/skeleton.glb"
            elif model == "hunyuan3d":
                jobs[job_id]["status"] = "mesh"
                await asyncio.to_thread(_hy3d, img, str(glb_path), seed, job_id, settings)
            elif model == "hunyuan3dmv":
                jobs[job_id]["status"] = "views"
                views = await asyncio.to_thread(_zero123_extend, img, seed, job_dir)
                views["front"] = img
                for name, vimg in views.items():
                    vimg.save(job_dir / f"view_{name}.png")
                jobs[job_id]["views_urls"] = {k: f"/results/{job_id}/view_{k}.png" for k in views}
                if _is_cancelled(job_id):
                    jobs[job_id]["status"] = "cancelled"
                    return
                jobs[job_id]["status"] = "mesh"
                await asyncio.to_thread(_hy3d_mv, views, str(glb_path), seed, job_id, settings)

            jobs[job_id]["glb_url"] = f"/results/{job_id}/mesh.glb"
            jobs[job_id]["status"] = "done"
            jobs[job_id]["finished"] = time.time()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[job {job_id}] ERROR: {tb}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = f"{type(e).__name__}: {e}"


ASPECT_SIZES = {
    "square": (1024, 1024),
    "portrait": (768, 1024),
    "wide": (1024, 768),
}


def _t2i_hunyuandit(prompt: str, seed: int, negative_prompt: str = "", aspect: str = "square", settings: dict = None, pose_id: str = "") -> Image.Image:
    settings = settings or {}
    gen = torch.Generator(device=HUNYUANDIT_DEVICE).manual_seed(int(seed))
    w, h = ASPECT_SIZES.get(aspect, ASPECT_SIZES["square"])
    neg = negative_prompt or "blurry, low quality, deformed, text, watermark"
    steps = int(_get(settings, "t2i", "num_inference_steps"))
    guidance = float(_get(settings, "t2i", "guidance_scale"))
    pag = float(_get(settings, "t2i", "pag_scale"))

    # ControlNet pipe is always the backend. When a pose is selected, use the
    # pose image with conditioning_scale=1.0. Without a pose, feed a black
    # placeholder and zero out the conditioning so the CN layers are inert.
    if hunyuandit_cn_pipe is None:
        raise RuntimeError(f"hunyuandit_cn_pipe unavailable: {hunyuandit_cn_error}")
    if pose_id:
        pose_img = Image.open(POSES_DIR / f"{pose_id}.png").convert("RGB").resize((w, h), Image.LANCZOS)
        cn_scale = 1.0
    else:
        pose_img = Image.new("RGB", (w, h), (0, 0, 0))
        cn_scale = 0.0
    print(f"[t2i] aspect={aspect} {w}x{h} steps={steps} cfg={guidance} pose={pose_id or 'none'} cn_scale={cn_scale}")
    print(f"[t2i] positive ({len(prompt)} chars): {prompt[:200]}")
    print(f"[t2i] negative ({len(neg)} chars): {neg[:200]}")

    with torch.cuda.device(HUNYUANDIT_DEVICE_IDX):
        result = hunyuandit_cn_pipe(
            prompt=prompt,
            negative_prompt=neg,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=w,
            height=h,
            control_image=pose_img,
            controlnet_conditioning_scale=cn_scale,
            generator=gen,
        )
    gc.collect()
    torch.cuda.empty_cache()
    img = result.images[0] if hasattr(result, "images") else result[0]
    if isinstance(img, list):
        img = img[0]
    return img


def _prep_for_anigen(img: Image.Image, size: int = 1024) -> Image.Image:
    """Center and frame the subject for AniGen.

    AniGen's internal preprocess handles background removal but not framing.
    If the character is small within the image or offset, the resulting mesh
    gets squashed/distorted. We bg-remove (RMBG-1.4), crop to the foreground
    bbox, pad to square with margin, and resize to AniGen's preferred size.
    """
    import numpy as np
    if bg_remover is not None:
        try:
            rgba = bg_remover(img.convert("RGBA") if img.mode != "RGBA" else img)
        except Exception as e:
            print(f"[anigen prep] bg_remover failed, using raw: {e}")
            rgba = img.convert("RGBA")
    else:
        rgba = img.convert("RGBA")
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")

    alpha_np = np.array(rgba.split()[-1])
    alpha_range = int(alpha_np.max()) - int(alpha_np.min())
    if alpha_range > 50:
        fg = alpha_np > 20
    else:
        fg = np.array(rgba.convert("L")) < 245

    ys, xs = np.where(fg)
    if len(xs) and len(ys):
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    else:
        x0, y0, x1, y1 = 0, 0, rgba.width, rgba.height
    print(f"[anigen prep] bbox={x0,y0,x1,y1} img={rgba.size}")
    rgba = rgba.crop((x0, y0, x1, y1))

    w, h = rgba.size
    margin = int(max(w, h) * 0.08)
    side = max(w, h) + 2 * margin
    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 255))
    canvas.paste(rgba, ((side - w) // 2, (side - h) // 2), rgba)
    return canvas.resize((size, size), Image.LANCZOS).convert("RGB")


def _anigen(image: Image.Image, output_glb: str, seed: int, settings: dict = None):
    settings = settings or {}
    # Frame the subject cleanly — massive quality improvement for uploads and
    # T2I outputs that have the character off-center or small in the frame.
    prepped = _prep_for_anigen(image, size=1024)
    set_random_seed(seed, deterministic=False)
    anigen_pipe.run(
        prepped,
        seed=seed,
        cfg_scale_ss=float(_get(settings, "anigen", "cfg_scale_ss")),
        cfg_scale_slat=float(_get(settings, "anigen", "cfg_scale_slat")),
        joints_density=int(_get(settings, "anigen", "joints_density")),
        no_smooth_skin_weights=not bool(_get(settings, "anigen", "smooth_skin_weights")),
        no_filter_skin_weights=not bool(_get(settings, "anigen", "filter_skin_weights")),
        smooth_skin_weights_iters=int(_get(settings, "anigen", "smooth_skin_weights_iters")),
        smooth_skin_weights_alpha=float(_get(settings, "anigen", "smooth_skin_weights_alpha")),
        output_glb=output_glb,
    )
    gc.collect()
    torch.cuda.empty_cache()


def _move_paint_to_device(paint_pipe, device: str):
    """Walk the Hunyuan3DPaintPipeline and move every torch nn.Module to `device`.

    The pipeline wraps a dict of sub-models (multiview_model, delight_model, etc.)
    which don't all honor a uniform .to() — move anything that quacks like a Module.
    """
    import torch.nn as nn
    moved = 0
    visited = set()

    def walk(obj, depth=0):
        nonlocal moved
        if id(obj) in visited or depth > 4:
            return
        visited.add(id(obj))
        if isinstance(obj, nn.Module):
            try:
                obj.to(device)
                moved += 1
            except Exception:
                pass
            return
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v, depth + 1)
        elif hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                walk(v, depth + 1)
    walk(paint_pipe)
    print(f"[hy3d] paint: moved {moved} sub-modules to {device}")


HY3D_MAX_FACES = 120_000  # cap for browser rendering; raw output is ~1M


def _hy3d_simplify(mesh, target_faces: int):
    """Reduce a Hunyuan3D mesh to ~target_faces via trimesh's Blender/fast-quadric."""
    try:
        import trimesh
        if not isinstance(mesh, trimesh.Trimesh):
            return mesh
        if len(mesh.faces) <= target_faces:
            return mesh
        try:
            simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
            print(f"[hy3d] decimated: {len(mesh.faces)} -> {len(simplified.faces)} faces")
            return simplified
        except Exception as e:
            print(f"[hy3d] quadric decimation unavailable ({e}); trying pymeshlab fallback")
            try:
                import pymeshlab
                ms = pymeshlab.MeshSet()
                ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
                ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces, preserveboundary=True)
                m = ms.current_mesh()
                simplified = trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix(), process=False)
                if hasattr(mesh, 'visual'):
                    simplified.visual = mesh.visual  # texture is lost on decimation; best-effort
                print(f"[hy3d] pymeshlab decimated: {len(mesh.faces)} -> {len(simplified.faces)} faces (texture may be lost)")
                return simplified
            except Exception as e2:
                print(f"[hy3d] pymeshlab fallback failed ({e2}); returning original")
                return mesh
    except Exception as e:
        print(f"[hy3d] simplify skipped: {e}")
        return mesh


def _hy3d(image: Image.Image, output_glb: str, seed: int, job_id: str = "", settings: dict = None):
    settings = settings or {}
    enable_bg = bool(_get(settings, "hunyuan3d", "enable_bg_removal"))
    enable_tex = bool(_get(settings, "hunyuan3d", "enable_texture"))
    max_faces = int(_get(settings, "hunyuan3d", "max_polygons"))

    if bg_remover is not None and enable_bg:
        try:
            image = bg_remover(image.convert("RGBA") if image.mode != "RGBA" else image)
            print(f"[hy3d] bg removed, mode={image.mode} size={image.size}")
        except Exception as e:
            print(f"[hy3d] bg_remover failed, using raw image: {e}")
    elif not enable_bg:
        print(f"[hy3d] bg removal disabled by settings")

    # Shape generation on cuda:2
    with torch.cuda.device(HY3D_SHAPE_DEVICE_IDX):
        gen = torch.Generator(device=HY3D_SHAPE_DEVICE).manual_seed(seed)
        try:
            meshes = hy3d_shape_pipe(image=image, generator=gen)
        except TypeError:
            meshes = hy3d_shape_pipe(image=image)
    mesh = meshes[0] if isinstance(meshes, (list, tuple)) else meshes

    # Decimate before paint — keeps browser happy and paint baking faster.
    mesh = _hy3d_simplify(mesh, max_faces)

    # Early cancel exit — paint is the slow stage, so honor cancel here.
    if job_id and _is_cancelled(job_id):
        print(f"[job {job_id}] cancel requested, skipping paint")
        mesh.export(output_glb)
        gc.collect()
        torch.cuda.empty_cache()
        return

    # Texture pass on cuda:3 (pinned separately so no collision with shape)
    if hy3d_paint_pipe is not None and enable_tex:
        try:
            with torch.cuda.device(HY3D_PAINT_DEVICE_IDX):
                mesh = hy3d_paint_pipe(mesh, image=image)
        except Exception as e:
            print(f"[hy3d] paint failed, using untextured shape: {e}")
            traceback.print_exc()
    elif not enable_tex:
        print(f"[hy3d] texture disabled by settings, exporting untextured mesh")

    mesh.export(output_glb)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
