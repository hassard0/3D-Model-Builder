# Settings reference

Every knob in the gear modal, what it does, and recommended ranges. Settings
persist to `localStorage["anigen_settings_v1"]` and apply to every generation
until you change or reset them.

## Text → Image (HunyuanDiT)

These tune the 2D conditioning image, which is the upstream of every 3D
result. Spend most of your tuning here — bad image quality is upstream of
every other quality issue.

### `custom_suffix`

A free-form string appended to the positive prompt after the style suffix.
Use to nudge style or composition without rewriting the prompt.

**Examples**:
* `cinematic lighting, dramatic angle` — for hero shots
* `flat colors, vector art style` — for stylized output
* `unreal engine 5, ray-traced` — for realism push

**Default**: empty.

### `custom_negative`

If non-empty, **replaces** the entire style negative prompt. Otherwise the
style preset's negative is used. Set this only if you know exactly what you're
doing — the default character negatives are tuned to prevent crossed legs,
side views, dress-occluded legs, etc.

**Default**: empty (use style default).

### `guidance_scale`

Classifier-free-guidance strength for HunyuanDiT. Higher = stricter prompt
adherence but more contrast/saturation artifacts. Lower = more creative
deviation.

**Range**: 1 - 15
**Recommended**: 6 - 9
**Default**: 7.5

### `pag_scale`

Perturbed-Attention Guidance scale. Improves structural fidelity to the
prompt at high values. Set to 0 to disable PAG entirely (faster). Note: PAG
is only used in the non-ControlNet code path; if you have a pose selected,
this is ignored.

**Range**: 0 - 5
**Recommended**: 1.3 - 2.5
**Default**: 2.0

### `num_inference_steps`

Number of denoising steps for HunyuanDiT. More steps = smoother output, more
time. Diminishing returns above 50.

**Range**: 10 - 100
**Recommended**: 25 - 50
**Default**: 35

## AniGen

These tune the rigging stage. Defaults are tuned for stylized 3D characters;
realistic humans may benefit from tweaks.

### `cfg_scale_ss`

Sparse-Structure flow CFG scale — the first AniGen pass that produces
skeleton + coarse shape.

**Range**: 1 - 15
**Recommended**: 6 - 10
**Default**: 8.0 (was 7.5 before EMA tuning)

### `cfg_scale_slat`

Structured-Latent flow CFG scale — the second AniGen pass that produces
detailed geometry and articulation.

**Range**: 1 - 8
**Recommended**: 2.5 - 5
**Default**: 3.5

### `joints_density`

Skeleton joint density level (0 = coarsest, 4 = densest). Higher values give
more articulated rigs (better for fine animation) but slow generation
slightly and can produce noisy joint placements.

**Range**: 0 - 4
**Recommended**: 1 - 3
**Default**: 2

### `smooth_skin_weights` (boolean)

Toggle smoothing of skinning weights. Off → sharp, blocky deformations;
on → smoother bends.

**Default**: true

### `smooth_skin_weights_iters`

Number of smoothing iterations when `smooth_skin_weights=true`.

**Range**: 0 - 500
**Recommended**: 50 - 200
**Default**: 100

### `smooth_skin_weights_alpha`

Smoothing strength multiplier.

**Range**: 0 - 2
**Recommended**: 0.5 - 1.5
**Default**: 1.0

### `filter_skin_weights` (boolean)

Geodesic-distance filter for skin weights. Removes weights from bones that
are far away from the affected vertices in surface distance — prevents an
arm bone from accidentally influencing a leg vertex when they're close in
Euclidean space but far in surface distance.

**Default**: true

## Hunyuan3D-2

These tune the static-mesh generation pathway. Affects both single-image
(`hunyuan3d`) and multi-view (`hunyuan3dmv`) jobs.

### `enable_texture` (boolean)

Run the paint stage to texture the mesh. Off → fast (~30-60 s) gray mesh,
on → slow (~2-3 min) but much prettier output.

**Default**: true. Turn off when you want fast iteration on shape only,
or when you'll re-texture in Blender / Substance.

### `enable_bg_removal` (boolean)

Apply RMBG-1.4 to the input image before Hy3D shape. Without this, the
backdrop sometimes leaks into the mesh as a floor / wall plane.

**Default**: true. Turn off only if your input is already a transparent PNG
(though RMBG running again is harmless).

### `max_polygons`

Cap on the output mesh's face count. Hy3D-2 raw output is around 1 M
triangles; we run quadric decimation down to this cap before paint and
export. Higher = more detail, slower paint, larger GLB. Lower = faster, less
detail.

**Range**: 5,000 - 1,000,000
**Recommended**: 50,000 - 200,000
**Default**: 120,000

## Resetting

The Reset button in the modal clears `localStorage["anigen_settings_v1"]`
and reloads the canonical defaults from `GET /api/defaults`. The page does
not reload — pose / model / image selections are preserved.
