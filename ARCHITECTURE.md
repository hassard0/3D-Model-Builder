# Architecture

## Process model

A single FastAPI process owns the GPUs and all model pipelines. Jobs run
serially under an `asyncio.Lock` so only one generation hits any GPU at a
time. This avoids OOM from concurrent runs and removes the need for a queue
manager — the lock IS the queue.

```
┌──────────────── FastAPI process (vulcan:9000) ────────────────┐
│                                                               │
│   ┌──────────────┐   ┌────────────┐   ┌──────────────────┐    │
│   │ HTTP routes  │──►│ jobs[]     │   │ gen_lock         │    │
│   │ /api/...     │   │ status     │   │ (asyncio.Lock)   │    │
│   └──────────────┘   └────────────┘   └────────┬─────────┘    │
│                                                ▼              │
│              ┌──────────────────────────────────────────┐     │
│              │  run_job()  /  run_job_from_image()      │     │
│              │  /  run_job_multiview()                  │     │
│              └─────────────────┬────────────────────────┘     │
│                                ▼                              │
│   ┌──────────────────────────────────────────────────┐        │
│   │                                                  │        │
│   │   _t2i_hunyuandit  ─►  cuda:1                    │        │
│   │   _zero123_extend  ─►  cuda:3                    │        │
│   │   bg_remover       ─►  cuda:2                    │        │
│   │   _anigen          ─►  cuda:0                    │        │
│   │   _hy3d            ─►  cuda:2 → cuda:3           │        │
│   │   _hy3d_mv         ─►  cuda:2 → cuda:3           │        │
│   │                                                  │        │
│   └──────────────────────────────────────────────────┘        │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Pipeline data flow

### Path A: text-to-rigged

```
prompt
  │
  ▼ enrich_prompt(style="character")
{positive, negative, aspect=portrait}
  │
  ▼ HunyuanDiT pipeline (cuda:1, optional ControlNet-Pose)
2D image (768×1024 PNG)
  │
  ▼ _prep_for_anigen — RMBG-1.4 → bbox crop → pad square → resize 1024
preprocessed RGB image
  │
  ▼ AniGen.run() (cuda:0, EMA weights)
GLB with mesh + skeleton + skinning + texture
```

### Path B: text-to-static

```
prompt → HunyuanDiT (same as above)
  │
  ▼ _prep for Hunyuan3D — RMBG-1.4 alpha matte
RGBA image
  │
  ▼ Hunyuan3D-2 shape (cuda:2)
trimesh (~1M triangles raw)
  │
  ▼ _hy3d_simplify — quadric decimation to user cap (default 120k)
decimated trimesh
  │
  ▼ Hunyuan3D-2 paint (cuda:3)
textured trimesh
  │
  ▼ mesh.export()
GLB
```

### Path C: text-to-multiview

```
prompt → HunyuanDiT
  │
  ▼ Zero123++ (cuda:3) on prepared front image
6-view 2-col × 3-row grid (320×320 tiles)
  │
  ▼ split into right (az=90°) / back (az=210°) / left (az=270°)
{front, right, back, left} dict of PIL images
  │
  ▼ Hunyuan3D-2mv (cuda:2)
trimesh
  │
  ▼ paint (cuda:3) using front view as texture reference
textured GLB
```

### Path D: image upload

Same as A/B/C but skips the T2I step. The uploaded image becomes the front
view directly. For multi-view mode, the user can drop additional views into
slots, and Zero123++ fills any gaps.

## GPU placement strategy

Sequential vs parallel use of each card determines what coexists:

* **cuda:0 — AniGen alone.** AniGen's flow models occupy ~8 GB during runs
  and we don't want anything else paging in/out.
* **cuda:1 — HunyuanDiT + ControlNet-Pose merged into one pipeline.** They
  can't fit as two separate pipelines on a 16 GB card. We always use the
  ControlNet variant; when no pose is selected we pass
  `controlnet_conditioning_scale=0` to make it a no-op.
* **cuda:2 — Hunyuan3D-2 shape + Hunyuan3D-2mv + RMBG-1.4.** The two shape
  variants are mutually exclusive per job (you pick one model in the UI),
  so they don't collide at runtime. RMBG-1.4 is tiny (~180 MB).
* **cuda:3 — Hunyuan3D-2 paint + Zero123++.** Paint runs after shape; Zero123
  runs before shape. Sequential, no collision.

If you have more or fewer cards, see `app.py:67-79` for the device constants.
The minimum sensible config is 3 cards: combine cuda:2 and cuda:3 onto one
card if and only if you don't need Hunyuan3D-2mv with paint.

## Skeleton classification

`viewer.js:analyzeSkeleton(model)` reads the loaded GLB's bone tree and
classifies bones into legs / arms / spine purely from world-space positions.

Algorithm:

1. Find the root bone — the bone whose parent is *not* a bone.
2. For each direct child bone of the root, probe two chains:
   * **Down probe** — repeatedly follow the child whose direction is most
     aligned with world-down. Length and total Y delta determine if this is
     a leg.
   * **Up probe** — same but with world-up. The longest upward chain is the
     spine.
3. Arms = any bone-chain branching off a spine bone that isn't on the spine.
4. Legs sorted left-to-right by world X for deterministic ordering.

`buildClassificationHelpers()` overlays small colored spheres on each bone
(R=leg, G=spine, B=arm, Y=other, white=root) so you can verify the
classification visually. This is critical for debugging procedural
animations — if Wave appears to move a leg, the classifier got fooled and
you'll see blue dots on what should be legs.

## Procedural animation derivation

For each bone, `boneSwingAxis(bone)` computes a world-space axis
perpendicular to the bone-to-child direction *and* world-up:

```js
swingAxis = (childPos - bonePos) × (0, 1, 0)
```

This axis is the natural hinge for forward/backward swing in the character's
own frame, regardless of how the rig is oriented in world coordinates. We
then convert this world axis into the bone's parent-local frame via
`localAxisFromWorld()` so the rotation composes correctly with the bone's
rest quaternion.

Walk, Run, Sit, Crouch, Jump all use this. Wave additionally uses
`boneLiftAxis(bone)` (perpendicular to swing × bone direction) for sideways
arm raises.

## Settings persistence

The gear modal builds a JSON object stored under `localStorage["anigen_settings_v1"]`.
The schema is:

```json
{
  "t2i": {
    "custom_suffix": "",
    "custom_negative": "",
    "guidance_scale": 7.5,
    "pag_scale": 2.0,
    "num_inference_steps": 35
  },
  "anigen": {
    "cfg_scale_ss": 8.0,
    "cfg_scale_slat": 3.5,
    "joints_density": 2,
    "smooth_skin_weights": true,
    "smooth_skin_weights_iters": 100,
    "smooth_skin_weights_alpha": 1.0,
    "filter_skin_weights": true
  },
  "hunyuan3d": {
    "enable_texture": true,
    "enable_bg_removal": true,
    "max_polygons": 120000
  }
}
```

The server hands out canonical defaults at `GET /api/defaults` and merges
the request's `settings` over them in `_get(settings, section, key)`. Empty
strings and missing keys fall through to defaults so partial settings dicts
work.

See [SETTINGS.md](SETTINGS.md) for what each field controls.

## Image preprocessing

Two preprocessing pipelines protect downstream model quality:

* **`_prep_for_zero123(img)`** — bg-remove (RMBG fallback to luminance), crop
  to foreground bbox, pad square with 10% margin, resize 512×512. Without
  this, Zero123++ output is wildly off-center because the model expects
  centered subjects on white in square aspect.

* **`_prep_for_anigen(img)`** — same shape, larger size (1024). AniGen does
  its own bg removal but doesn't re-frame; pre-centering the subject is the
  single biggest lever on rig and mesh quality.

## Compatibility patches

`app.py` applies several runtime patches on import. See top-of-file comments
and [TROUBLESHOOTING.md](TROUBLESHOOTING.md):

1. `transformers.modeling_utils.check_torch_load_is_safe` — bypass torch≥2.6
   CVE gate so we can load `.bin` weights on torch 2.5.1.
2. `PreTrainedModel.all_tied_weights_keys = {}` default — RMBG-1.4 custom
   model class predates this transformers 5.5 API.
3. We bypass `hy3dgen.text2image.HunyuanDiTPipeline` entirely (truncates
   prompts at 60 chars, hardcodes Chinese suffix) and call diffusers'
   `HunyuanDiTControlNetPipeline` directly.
4. Hy3D paint sub-modules are walked and explicitly moved to `cuda:3` after
   load — the official paint constructor ignores `device=` and defaults to
   `cuda:0`.
