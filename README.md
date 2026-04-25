# 3D Model Builder

Web studio that turns a text prompt or a reference image into a textured (and
optionally rigged) 3D mesh, served from a single FastAPI process with all
models hot-loaded across multiple GPUs.

**Pipeline at a glance**

```
text prompt ──► HunyuanDiT (+ControlNet-Pose) ──► 2D image ─┐
                                                            ├──► AniGen          ──► rigged GLB
uploaded image ─────────────────────────────────────────────┤
                                                            ├──► Hunyuan3D-2     ──► textured static GLB
multi-view inputs ─► (Zero123++ fills missing) ─────────────┘    Hunyuan3D-2mv   ──► textured static GLB (multi-view shape)
```

A three.js viewer renders the result with orbit controls, lighting presets,
camera presets, procedural animations, a skeleton classification overlay,
and an interactive bone-drag editor.

## Models

| Stage              | Model                                                | GPU     | Approx VRAM |
| ------------------ | ---------------------------------------------------- | ------- | ----------- |
| Text → image       | `Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers` + `…ControlNet-Diffusers-Pose` | cuda:1  | 9.5 GB      |
| Background removal | `briaai/RMBG-1.4`                                    | cuda:2  | 0.2 GB      |
| Image → rigged 3D  | `VAST-AI/AniGen` (ss_flow_duet + slat_flow_auto, EMA)| cuda:0  | 8 GB        |
| Image → static 3D  | `tencent/Hunyuan3D-2` (shape + paint)                | cuda:2/3| 12 GB       |
| Multi-view → 3D    | `tencent/Hunyuan3D-2mv`                              | cuda:2  | 5.5 GB      |
| Novel-view synth   | `sudo-ai/zero123plus-v1.2`                           | cuda:3  | 5 GB        |

Tested on 4× NVIDIA Tesla V100-SXM2 16 GB (Volta, SM 7.0). Volta-specific
quirks (no bf16, no flash-attn-2, prebuilt CUDA wheels often Ampere-only) are
documented in `DEPLOY.md`.

## Features

* **Text or image input.** Type a prompt, drop an image, or fill multi-view
  slots — the server picks the right path.
* **Pose ControlNet.** 15 pre-rendered OpenPose skeletons (T-pose, A-pose,
  walking, etc.) you can pick from to force the generated 2D image into a
  specific pose. Critical for getting AniGen rigs right.
* **Zero123++ novel-view synthesis** for Hunyuan3D-2mv when you only have a
  front view — generates back/left/right.
* **Live 3D viewer** with orbit controls, animation playback, lighting
  presets (studio / neutral / sunset / neon / night) plus per-light sliders,
  camera presets (front / 3-4 / side / back / top), skeleton + classification
  overlays, wireframe, grid, auto-rotate.
* **Procedural animations** built from skeleton topology: idle sway, breathe,
  spin, walk, run, wave, sit, crouch, jump, wiggle, dance.
* **Bone editor.** Click a colored sphere in the classification overlay,
  drag the gizmo to reposition that bone (mesh deforms via skinning), press
  R to switch to rotate mode, Esc to deselect. Download the edited GLB.
* **Cancel** an in-flight job at the next stage boundary; up to ~60 s saved
  on Hunyuan3D-2 paint.
* **Settings modal** with per-model parameters persisted to localStorage:
  guidance/PAG/steps for T2I, CFG/joint density/skin smoothing for AniGen,
  texture toggle and polygon cap for Hunyuan3D-2.
* **Job history** under `results/web/<job_id>/`.

## Repo layout

```
app.py              FastAPI server + pipeline wiring + monkey-patches
prompt_harness.py   Style presets and prompt enrichment
pose_gallery.py     Generates 15 OpenPose skeleton images at 1024×1024
static/
  index.html        Main UI
  viewer.js         three.js viewer, animations, rig editor
  poses/            (generated) pose skeleton PNGs
DEPLOY.md           Server install instructions (V100-specific)
```

## Quick start

This stack is tuned for a 4× V100 server. Single-GPU isn't supported — it
relies on having distinct devices for AniGen, HunyuanDiT, Hy3D shape, Hy3D
paint. See `DEPLOY.md` for the full install walkthrough.

```bash
# 1. Generate the pose gallery (one-off; pose PNGs aren't checked in)
python pose_gallery.py static/poses

# 2. Run the server
python app.py
# binds 0.0.0.0:9000
```

Open `http://localhost:9000` (or your tailnet IP). Type a prompt or upload an
image, pick a model, hit Generate.

## License notes

* AniGen, HunyuanDiT, Hunyuan3D-2 — Tencent / VAST-AI non-commercial licenses.
  Use accordingly.
* Zero123++ — Apache 2.0.
* RMBG-1.4 — Bria AI Community License (non-commercial).
* This repo's own code: MIT.
