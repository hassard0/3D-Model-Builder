# Troubleshooting

Every failure mode we've hit during development plus the fix. Symptom-first;
search this page for the error message you saw.

## Install-time

### `CondaToSNonInteractiveError: Terms of Service have not been accepted`

Conda 26 requires you to accept the TOS for the default channels. Either:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

…or just use `conda-forge` exclusively:

```bash
conda create -n anigen -c conda-forge --override-channels python=3.10 -y
```

### `RuntimeError: Cuda error: 209[cudaFuncGetAttributes(&attr, (void*)fineRasterKernel)]`

The prebuilt nvdiffrast wheel that AniGen's setup pulls is compiled for
Ampere only. On Volta (V100, SM 7.0) the kernel image isn't present.

**Fix**: rebuild from source. The source path uses JIT compilation which
auto-targets the current GPU's SM:

```bash
pip uninstall nvdiffrast -y
pip install https://github.com/NVlabs/nvdiffrast/archive/refs/tags/v0.3.3.tar.gz \
            --no-build-isolation
```

### `ModuleNotFoundError: No module named 'rtree'`

Trimesh's `proximity.closest_point` (used by AniGen during skin-weight
transfer) needs `rtree`.

**Fix**: `pip install rtree`

### `ValueError: tiktoken is required to read a tiktoken file`

Triggered when loading some HunyuanDiT components. The tokenizer auto-detector
in transformers 5.5 routes through tiktoken for some configurations.

**Fix**: `pip install tiktoken`

### `ValueError: Error parsing line b'\x0e' in spiece.model`

After installing tiktoken, transformers 5.5 incorrectly tries to parse
SentencePiece `.model` files (T5 tokenizer) as tiktoken BPE files. The fix
is to install `sentencepiece` so transformers routes T5 through the proper
SpmConverter:

**Fix**: `pip install sentencepiece`

### `OutOfMemoryError: CUDA out of memory` at startup, cuda:1

You're trying to load both plain HunyuanDiT and HunyuanDiT-ControlNet on the
same 16 GB card. They don't fit together.

**Fix**: `app.py` only loads the ControlNet variant and aliases
`hunyuandit_pipe = hunyuandit_cn_pipe`. Without a pose selected, we pass
`controlnet_conditioning_scale=0` to make the CN layer a no-op. If you've
modified the code to load both, revert.

### `ImportError: flash_attn` or `flash-attn-2 not supported on SM 7.0`

Volta doesn't support Flash Attention 2. Flash Attention is required by
Puppeteer (Seed3D) and a few other libraries.

**Fix**: don't install `flash-attn`. xFormers + SDPA cover attention
performance fine on V100. If you must run a model that requires FA2, run it
on an Ampere or newer card.

## Runtime

### `Due to a serious vulnerability issue in torch.load... upgrade torch to at least v2.6`

Transformers 5.5+ refuses to load `.bin` (pickle) checkpoints with
`torch < 2.6` because of CVE-2025-32434. We pin torch 2.5.1 because xformers
ABI breaks on torch 2.6.

**Fix**: already monkey-patched in `app.py`:

```python
import transformers.utils.import_utils as _tui
_tui.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils as _tmu
_tmu.check_torch_load_is_safe = lambda: None
```

Only safe because every `.bin` we load is from pre-fetched official HF
weights, not user input.

### `'BriaRMBG' object has no attribute 'all_tied_weights_keys'`

`briaai/RMBG-1.4` is a custom remote-code model whose class predates the
`all_tied_weights_keys` attribute that transformers 5.5 expects.

**Fix**: already monkey-patched in `app.py`:

```python
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = {}
```

### Hunyuan3D-2 paint loads on cuda:0 instead of cuda:3

The `Hunyuan3DPaintPipeline` constructor ignores any `device=` kwarg and
defaults all sub-modules to the current default CUDA device (cuda:0).

**Fix**: in `app.py`, paint is loaded inside
`with torch.cuda.device(HY3D_PAINT_DEVICE_IDX):` and a helper
`_move_paint_to_device()` walks every sub-`nn.Module` calling `.to(device)`.
At runtime, every paint call is also wrapped in the same device context.

### Hunyuan3D-2 mesh has a floor / backdrop plane stuck to it

The shape model treats the entire image (including backdrop) as the subject
when no alpha channel is provided.

**Fix**: RMBG-1.4 is applied to the input image before Hy3D shape (toggle
**enable_bg_removal** in settings to override). If the backdrop still leaks,
the bg-removal model failed for that subject — either the subject has near-
white parts that get matted out, or near-black parts that get included as bg.
Switch the rembg session model in `_RMBGWrapper` (e.g., `isnet-general-use`).

### Browser hangs or 3D viewer is sluggish on Hunyuan3D-2 output

Raw Hy3D mesh is ~1M triangles. Browser GLTF parsing on a multi-megabyte file
is slow.

**Fix**: server-side decimation runs by default with a 120k face cap. If
output is still huge, lower `max_polygons` in Hy3D-2 settings.

### Zero123++ output views look distorted / off-center

Zero123++ requires its conditioning image to be:

* Background-removed (white or transparent)
* Cropped tight to the subject
* Padded square
* Resized to ~512×512

We do all of that in `_prep_for_zero123()`. If you bypass that path, expect
poor output. If the output is still wonky, the subject is probably too
unusual (extreme pose, occlusion, multiple objects) — Zero123 was trained
on Objaverse-style 3D-rendered assets.

### Zero123++ tile extraction grabs wrong slices of the grid

Zero123++ v1.2 outputs a **2-column × 3-row** grid (tall), not 3×2.
Row-major, the azimuths are: (0,0)=30°, (1,0)=90°, (0,1)=150°, (1,1)=210°,
(0,2)=270°, (1,2)=330°.

**Fix**: in `_zero123_extend()`, splitter uses `tw=w//2, th=h//3`, picks
`tiles[1]` (right, az=90°), `tiles[3]` (back, az=210°), `tiles[4]` (left,
az=270°).

### HunyuanDiT prompt is silently truncated

`hy3dgen.text2image.HunyuanDiTPipeline.__call__` truncates the input prompt
at 60 characters and appends a hardcoded Chinese suffix. Our long
English-only enriched prompts get clipped before HunyuanDiT ever sees the
pose cues.

**Fix**: bypass the wrapper. Call the underlying diffusers pipeline directly
via `hunyuandit_cn_pipe(...)`. Already done in `_t2i_hunyuandit()`.

### Generated character is in a casual / cross-legged pose despite "T-pose" in prompt

HunyuanDiT's English prompt fidelity is moderate; "T-pose" tokens don't have
strong representation in its training data. With ControlNet-Pose loaded,
the actual fix is to **pick the T-pose tile in the Pose gallery** before
hitting Generate — that forces structure via ControlNet, which is much
stronger than prompt tokens alone.

### AniGen's rig has overlapping legs (single column down the body center)

AniGen needs to see two distinct legs in the 2D image to rig them. If your
character wears a long dress, robe, gown, or coat that hides leg geometry,
AniGen produces a single-column "leg stack" centered on the body.

**Workarounds**:

1. Re-prompt with explicit short clothing: `wearing shorts, bare legs visible`.
2. Use the **Rig editor**: enable Show Classification, click a red leg sphere,
   drag it sideways. Repeat for the other leg. Save edited GLB.
3. Switch to **Hunyuan3D-2** (no rigging — produces a static mesh; rig
   externally in Blender / Mixamo).

### Procedural Wave animation moves a leg instead of an arm

Means the skeleton classifier got fooled — what we labeled as "arm" is
actually a leg in the 3D mesh.

**Diagnose**: enable **Show classification** in Display. Red dots should be
on the legs and blue on the arms. If they're swapped or both stacked, the
rig itself is the issue (see the previous entry).

**Fix**: use the rig editor to fix bone positions, OR regenerate with a
better source image (T-pose ControlNet, no occluding clothing).

### Mesh feet are floating above or sinking through the grid

The viewer grounds objects by their detected ankle bones (lowest leaf of
each leg chain). If no legs are detected, it falls back to the mesh
bounding box.

**Diagnose**: open browser console (F12). You should see one of:

* `[ground] shifted by 0.123 so ankles sit at y=0` — ankle grounding worked
* (no log) — fallback to bbox-based grounding, can be off if hair / cape
  extends below feet

If feet still float and ankles weren't detected, the leg chains are too
short (`< 1` bone after the hip) — toggle **Show classification** to verify.

### "Synthesizing novel views" stage is slow on the first job after restart

Zero123++ uses `custom_pipeline="sudo-ai/zero123plus-pipeline"` which JIT-
compiles a custom pipeline class on first use. Subsequent calls are fast.

### Server takes too long to start

Cold start is ~2 minutes — RMBG (3s) + AniGen (44s) + HunyuanDiT+ControlNet
(8s) + Hy3D shape (29s) + Hy3D-2mv (28s) + Zero123++ (3s) + Hy3D paint (14s).

If you want faster iteration during development, you can comment out the
unused models in `app.py`'s `startup()`. The first generation request after
their removal will fail with a 503 from the corresponding endpoint.

### `Address already in use` on port 9000

Another instance of the server is still running.

**Fix**:

```bash
pkill -9 -f 'server/app.py'
sleep 2
nohup /tmp/run_server.sh > /tmp/anigen_server.log 2>&1 &
```

Or pick a different port in `app.py:if __name__ == "__main__"` (uvicorn args).

### `model/gltf-binary` MIME type not recognized in Firefox

Firefox is more strict about content-type for `<model-viewer>` and similar
embeds. The server already sets the correct MIME type. If you're proxying
through nginx, make sure the proxy preserves it (don't override with
`text/octet-stream`).

## Performance / quality

### Procedural animations look stiff / wrong direction

The bone-axis math uses `cross(child_direction, world_up)` which degenerates
for vertical bones (legs in T-pose). The fallback is world-X axis, which
works for characters facing -Z but not +Z.

**Fix**: hard-refresh the viewer to pick up the latest `viewer.js` (the
classifier and axis math have been reworked several times). If still wrong,
the rig itself has unusual geometry — share the `[skel]` log from the
console to debug.

### Animation playback freezes after switching characters

Three.js `AnimationMixer` holds references to the previous skeleton. The
viewer rebuilds it on every load.

**Fix**: hard-refresh; should not occur in current code. If it does, check
console for errors during GLB load.

### Browser memory usage growing

Each new generation adds another GLB to the scene without disposing the
previous. The viewer's `clearModel()` traverses and disposes geometry +
materials on each new load.

**Diagnose**: open browser memory profiler (Chrome devtools → Memory tab),
take heap snapshots before and after generations.

## Debugging tips

* **`/tmp/anigen_server.log`** — full server log including all `[startup]`,
  `[t2i]`, `[hy3d]`, `[zero123]` traces.
* **Browser console (F12)** — `[skel]` prints the bone classification on
  every model load; `[ground]` reports the grounding offset; `[rig-edit]`
  prints when you click a classification sphere.
* **`results/web/<job_id>/`** on the server — every job's `input.png`,
  `view_*.png` (multi-view), `mesh.glb`, `skeleton.glb` (AniGen),
  `zero123_cond.png` (the actual cond image fed to Zero123),
  `zero123_grid.png` (raw 2×3 output grid). Inspecting these reveals
  exactly what each stage saw and produced.
* **`/api/health`** — sanity-check that all models loaded.
* **`/api/status/<job_id>`** — current state and timing.
