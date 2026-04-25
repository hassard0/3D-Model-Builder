# 3D Model Builder

A self-hosted web studio for generating 3D characters and objects from text
prompts or reference images. Six ML models are hot-loaded across four GPUs,
orchestrated by a single FastAPI process, and rendered in a three.js viewer
with procedural animations and an interactive bone-drag rig editor.

```
┌─────────────────────────────┐    ┌─────────────────────────────┐
│  Text prompt                │───►│  HunyuanDiT  +  ControlNet  │
│  + style preset             │    │  Pose (optional T-pose)     │
│  + optional pose pick       │    │           cuda:1            │
└─────────────────────────────┘    └────────────┬────────────────┘
                                                ▼
   ┌──────────────┐         ┌────────────────────────────────────┐
   │ Upload image │────────►│        2D conditioning image       │
   └──────────────┘         └────────────┬───────────────────────┘
                                         ▼
                          (RMBG-1.4 background removal — cuda:2)
                                         ▼
            ┌────────────────────────────┴──────────────────────────────┐
            ▼                            ▼                              ▼
   ┌──────────────────┐      ┌────────────────────┐      ┌──────────────────────────┐
   │ AniGen (cuda:0)  │      │ Hunyuan3D-2        │      │ Hunyuan3D-2mv (cuda:2)   │
   │ image → rigged   │      │ shape (cuda:2)     │      │ multi-view → mesh        │
   │ textured GLB     │      │ paint (cuda:3)     │      │ Zero123++ (cuda:3) fills │
   └──────────────────┘      └────────────────────┘      │ missing views            │
                                                          └──────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │ three.js viewer with procedural animations, │
                    │ lighting/camera presets, classification     │
                    │ overlay, and interactive bone editor        │
                    └─────────────────────────────────────────────┘
```

## Documentation

* **[ARCHITECTURE.md](ARCHITECTURE.md)** — pipeline diagrams, GPU placement, model details
* **[USAGE.md](USAGE.md)** — common workflows and UI walkthrough
* **[API.md](API.md)** — REST endpoint reference
* **[SETTINGS.md](SETTINGS.md)** — every tunable knob in the gear modal
* **[DEPLOY.md](DEPLOY.md)** — install on a fresh server (V100-tested)
* **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** — every failure mode I've hit and the fix

## Features

### Inputs
* **Text → image → 3D** with HunyuanDiT v1.1 (full, undistilled, bilingual CLIP+mT5)
* **Image upload → 3D** to skip the T2I step
* **Multi-view → 3D** with up to 4 hand-uploaded views (front/back/left/right)
* **Auto-fill missing views** via Zero123++ novel-view synthesis when only the front is provided
* **Pose ControlNet** with 15 pre-rendered OpenPose skeletons (T-pose, A-pose, idle, walking, running, sitting, crouching, wave, jump, power, victory, fighting, kneeling, ballet, dab) for forcing pose structure during T2I
* **Style presets**: Character (rigged-friendly), Creature, Object, Stylized, Realistic — each with style-specific positive suffixes and negative prompts

### 3D backends
* **AniGen** — `VAST-AI/AniGen`. Image-to-rigged-3D producing a textured GLB with skeleton, joints, and skinning weights. Uses EMA weights for higher quality.
* **Hunyuan3D-2** — `tencent/Hunyuan3D-2`. Image-to-textured static mesh. ~1 M raw triangles, decimated client-side cap (default 120k).
* **Hunyuan3D-2mv** — `tencent/Hunyuan3D-2mv`. Multi-view textured mesh; better geometry from 2-4 views.

### Viewer
* **Procedural animations** generated from skeleton topology: idle sway, breathe, spin, walk, run, wave, sit, crouch, jump, wiggle, dance. The classifier identifies legs/arms/spine geometrically; rotation axes are derived per-bone (perpendicular to bone-direction × world-up).
* **Lighting controls** with 5 environment presets (studio / neutral / sunset / neon / night), per-light intensity sliders (ambient, key, fill, rim), exposure slider, background color picker.
* **Camera presets** — Front, 3/4, Side, Back, Top, Fit
* **Display toggles** — skeleton, classification overlay (color-coded R=legs G=spine B=arms Y=other), wireframe, grid, auto-rotate
* **Interactive bone-drag rig editor**: click any classification sphere to attach a translate/rotate gizmo to that bone. Drag to reposition; mesh deforms live via skinning. **R** = rotate mode, **T** = translate mode, **Esc** = deselect. Save the corrected rig as a new GLB.
* **Job cancellation** at stage boundaries (saves up to ~60 s on a Hunyuan3D-2 paint job)
* **Settings persisted** to `localStorage` per browser

### Server
* Single FastAPI process, all six models hot-loaded at startup (~2 minute cold boot)
* Job locking — one generation at a time across all models (avoids GPU thrash)
* Per-stage status tracking (`queued` → `image` → `views` → `mesh` → `done` / `error` / `cancelled`)
* Result files persist on disk under `results/web/<job_id>/`

## Hardware

Tested on Ubuntu 24.04 with **4× NVIDIA Tesla V100-SXM2 16 GB** (Volta, SM 7.0,
no native bf16, no flash-attn-2 support). VRAM at steady state:

| GPU    | Models                                                | Used   | Free  |
| ------ | ----------------------------------------------------- | ------ | ----- |
| cuda:0 | AniGen                                                | 8.2 GB | 8 GB  |
| cuda:1 | HunyuanDiT + ControlNet-Pose (single fused pipeline)  | 9.5 GB | 7 GB  |
| cuda:2 | Hunyuan3D-2 shape + Hunyuan3D-2mv + RMBG-1.4         | 10.5 GB | 5 GB |
| cuda:3 | Hunyuan3D-2 paint + Zero123++                         | 11.0 GB | 5 GB |

Should also run on Ampere or newer with simpler dependency pins (no need for
Volta-specific rebuilds). Single-GPU is *not* supported — the design relies on
distinct devices to keep models hot.

## Quick start

```bash
git clone https://github.com/hassard0/3D-Model-Builder.git
cd 3D-Model-Builder
./setup.sh             # one-command install (~30 min, including ~75 GB weight prefetch)
python preflight.py    # verify the env
python app.py          # bind 0.0.0.0:9000
```

Then open `http://localhost:9000` in a browser. See [DEPLOY.md](DEPLOY.md)
for what `setup.sh` does step-by-step, or to install components separately.

If `setup.sh` fails partway, re-run it — every step is idempotent and
checks for prior completion before doing work.

## License

* This repo's orchestration code: [MIT](LICENSE)
* AniGen, HunyuanDiT, Hunyuan3D-2, Hunyuan3D-2mv: Tencent / VAST-AI
  non-commercial licenses — abide by their terms
* Zero123++: Apache 2.0
* RMBG-1.4: BRIA AI Community License (non-commercial)

This is a research / personal-use stack. Verify the downstream model licenses
before any commercial deployment.

## Acknowledgements

Built on top of the open releases from
[VAST-AI-Research/AniGen](https://github.com/VAST-AI-Research/AniGen),
[Tencent-Hunyuan/Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2),
[Tencent-Hunyuan/HunyuanDiT](https://github.com/Tencent-Hunyuan/HunyuanDiT),
[SUDO-AI-3D/zero123plus](https://github.com/SUDO-AI-3D/zero123plus), and
[BRIA AI's RMBG](https://huggingface.co/briaai/RMBG-1.4).
