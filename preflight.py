"""Preflight diagnostic — verify the env can run the server.

Run before `python app.py` if you've just done an install or are debugging
a startup failure. Reports each check with PASS / WARN / FAIL and exits
non-zero if anything's broken.

  python preflight.py
"""
from __future__ import annotations
import importlib
import os
import shutil
import sys


GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
RED = "\033[1;31m"
RESET = "\033[0m"


class Result:
    def __init__(self):
        self.fails = 0
        self.warns = 0

    def ok(self, msg: str):
        print(f"  {GREEN}PASS{RESET}  {msg}")

    def warn(self, msg: str, hint: str = ""):
        self.warns += 1
        print(f"  {YELLOW}WARN{RESET}  {msg}")
        if hint:
            print(f"           {hint}")

    def fail(self, msg: str, hint: str = ""):
        self.fails += 1
        print(f"  {RED}FAIL{RESET}  {msg}")
        if hint:
            print(f"           {hint}")


def check_python(r: Result):
    print(f"\n{YELLOW}python{RESET}")
    v = sys.version_info
    if v.major == 3 and 10 <= v.minor <= 12:
        r.ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        r.warn(
            f"Python {v.major}.{v.minor}.{v.micro}",
            "tested on 3.10 and 3.11; 3.13+ may have wheel availability issues."
        )


def check_torch(r: Result):
    print(f"\n{YELLOW}torch / cuda{RESET}")
    try:
        import torch
        r.ok(f"torch {torch.__version__}")
        if not torch.version.cuda:
            r.fail("torch was installed without CUDA support",
                   "reinstall: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            return
        r.ok(f"CUDA {torch.version.cuda}")
        n = torch.cuda.device_count()
        if n == 0:
            r.fail("no CUDA devices visible",
                   "check `nvidia-smi`; the driver may not be loaded.")
            return
        r.ok(f"{n} CUDA device(s) detected")
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"           cuda:{i} = {name} (SM {cap[0]}.{cap[1]}, {mem:.1f} GB)")
        if n < 4:
            r.warn(f"server expects 4 GPUs; found {n}",
                   "edit cuda:N constants in app.py if you have a different layout.")
    except Exception as e:
        r.fail(f"could not import torch: {e}",
               "run setup.sh or pip install torch==2.5.1.")


def check_pkg(r: Result, mod: str, label: str = "", required: bool = True, hint: str = ""):
    label = label or mod
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "")
        r.ok(f"{label} {v}".strip())
    except Exception as e:
        if required:
            r.fail(f"could not import {label}: {e}", hint)
        else:
            r.warn(f"could not import {label}: {e}", hint)


def check_packages(r: Result):
    print(f"\n{YELLOW}server deps{RESET}")
    check_pkg(r, "fastapi")
    check_pkg(r, "uvicorn")
    check_pkg(r, "diffusers", hint="pip install diffusers==0.37.1")
    check_pkg(r, "transformers", hint="pip install transformers==5.5.4")
    check_pkg(r, "accelerate")
    check_pkg(r, "sentencepiece", hint="pip install sentencepiece — required for HunyuanDiT T5 tokenizer.")
    check_pkg(r, "tiktoken", hint="pip install tiktoken — required by transformers 5.5.")
    check_pkg(r, "rtree", hint="pip install rtree — required by trimesh proximity for AniGen skin-weight transfer.")
    check_pkg(r, "trimesh")
    check_pkg(r, "pymeshlab", required=False, hint="optional: improves Hunyuan3D mesh decimation fallback.")

    print(f"\n{YELLOW}3D pipelines{RESET}")
    check_pkg(r, "anigen", label="anigen (VAST-AI-Research/AniGen)",
              hint="ensure ~/AniGen is on PYTHONPATH or pip install -e it.")
    check_pkg(r, "hy3dgen", label="hy3dgen (Tencent-Hunyuan/Hunyuan3D-2)",
              hint="cd ~/Hunyuan3D-2 && pip install -e .")
    check_pkg(r, "xformers")
    check_pkg(r, "spconv.pytorch", label="spconv")
    check_pkg(r, "pytorch3d")
    check_pkg(r, "nvdiffrast")


def check_nvdiffrast_runtime(r: Result):
    print(f"\n{YELLOW}nvdiffrast runtime{RESET}")
    try:
        import torch
        import nvdiffrast.torch as dr
        if torch.cuda.device_count() == 0:
            r.warn("no CUDA device — skipping nvdiffrast runtime check")
            return
        ctx = dr.RasterizeCudaContext(device="cuda:0")
        r.ok("RasterizeCudaContext initialized on cuda:0")
    except Exception as e:
        r.fail(f"nvdiffrast runtime failed: {e}",
               "rebuild from source for your GPU's compute capability:\n"
               "    pip install https://github.com/NVlabs/nvdiffrast/archive/refs/tags/v0.3.3.tar.gz --no-build-isolation")


def check_disk(r: Result):
    print(f"\n{YELLOW}disk{RESET}")
    home = os.path.expanduser("~")
    cache = os.path.join(home, ".cache", "huggingface")
    used = shutil.disk_usage(home)
    free_gb = used.free / 1e9
    if free_gb >= 100:
        r.ok(f"{free_gb:.0f} GB free on /home (>= 100 GB recommended for weight cache)")
    elif free_gb >= 50:
        r.warn(f"only {free_gb:.0f} GB free; HF weights total ~75 GB.")
    else:
        r.fail(f"only {free_gb:.0f} GB free; ~75 GB needed for HF weight cache.")
    if os.path.isdir(cache):
        size = sum(
            sum(os.path.getsize(os.path.join(d, f)) for f in fs)
            for d, _, fs in os.walk(cache)
        ) / 1e9
        print(f"           HF cache currently {size:.1f} GB at {cache}")


def check_weight_repos(r: Result):
    print(f"\n{YELLOW}HF weight cache{RESET}")
    home = os.path.expanduser("~")
    repos = [
        "Tencent-Hunyuan--HunyuanDiT-v1.1-Diffusers",
        "Tencent-Hunyuan--HunyuanDiT-v1.1-ControlNet-Diffusers-Pose",
        "tencent--Hunyuan3D-2",
        "tencent--Hunyuan3D-2mv",
        "sudo-ai--zero123plus-v1.2",
        "briaai--RMBG-1.4",
    ]
    cache_root = os.path.join(home, ".cache", "huggingface", "hub")
    for repo in repos:
        path = os.path.join(cache_root, f"models--{repo}")
        if os.path.isdir(path):
            r.ok(f"cached: {repo.replace('--', '/')}")
        else:
            r.warn(f"not cached: {repo.replace('--', '/')}",
                   "will download on first server start (slow). Run setup.sh without --skip-weights, "
                   "or python -c \"from huggingface_hub import snapshot_download; snapshot_download('"
                   f"{repo.replace('--', '/')}')\"")


def check_pose_gallery(r: Result):
    print(f"\n{YELLOW}pose gallery{RESET}")
    here = os.path.dirname(os.path.abspath(__file__))
    poses_dir = os.path.join(here, "static", "poses")
    expected = ["tpose.png", "apose.png", "idle.png", "walking.png"]
    if not os.path.isdir(poses_dir):
        r.warn(f"static/poses/ does not exist", "run: python pose_gallery.py static/poses")
        return
    missing = [p for p in expected if not os.path.isfile(os.path.join(poses_dir, p))]
    if missing:
        r.warn(f"missing pose images: {missing}", "rerun: python pose_gallery.py static/poses")
    else:
        r.ok(f"{len([f for f in os.listdir(poses_dir) if f.endswith('.png')])} pose PNGs in static/poses/")


def main():
    print(f"{YELLOW}3D Studio — preflight diagnostic{RESET}")
    r = Result()
    check_python(r)
    check_torch(r)
    check_packages(r)
    check_nvdiffrast_runtime(r)
    check_disk(r)
    check_weight_repos(r)
    check_pose_gallery(r)
    print()
    if r.fails:
        print(f"{RED}{r.fails} FAIL{RESET}, {r.warns} warn — fix the FAILs before starting the server.")
        sys.exit(1)
    elif r.warns:
        print(f"{YELLOW}{r.warns} warn{RESET}, no FAIL — server should start but check the warnings.")
        sys.exit(0)
    else:
        print(f"{GREEN}all clear.{RESET} run: python app.py")
        sys.exit(0)


if __name__ == "__main__":
    main()
