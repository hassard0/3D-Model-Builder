# API Reference

All endpoints are served by `app.py` on port 9000. Responses are JSON unless
otherwise noted. Errors return a standard FastAPI `{"detail": "..."}` body
with a 4xx/5xx status code.

## Health and metadata

### `GET /api/health`

Returns server readiness and per-model status.

```json
{
  "models": {
    "anigen":      {"ready": true, "label": "AniGen — rigged + animatable", "rigged": true},
    "hunyuan3d":   {"ready": true, "label": "Hunyuan3D-2 — textured mesh (static)", "rigged": false, "textured": true},
    "hunyuan3dmv": {"ready": true, "label": "Hunyuan3D-2mv — multi-view textured mesh", "rigged": false, "textured": true, "multiview": true, "zero123_ready": true},
    "t2i":         {"ready": true, "label": "HunyuanDiT v1.1 (full)"}
  },
  "styles": [
    {"key": "character", "label": "Character (rigged-friendly)"},
    {"key": "creature",  "label": "Creature / animal"},
    ...
  ],
  "active_jobs": 0,
  "total_jobs":  17
}
```

### `GET /api/defaults`

Canonical default values for the settings dict. Used by the UI to populate
the modal on first load.

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

### `GET /api/poses`

List the available pose-gallery entries.

```json
{
  "poses": [
    {"key": "tpose",   "label": "T-pose",   "url": "/static/poses/tpose.png"},
    {"key": "apose",   "label": "A-pose",   "url": "/static/poses/apose.png"},
    {"key": "idle",    "label": "Idle",     "url": "/static/poses/idle.png"},
    ...
  ],
  "ready": true
}
```

`ready: false` means the ControlNet pipeline failed to load and selecting a
pose will be ignored.

### `POST /api/enrich`

Preview the prompt that will be sent to HunyuanDiT, without running a job.
Used by the UI to show "what HunyuanDiT sees" live as you type.

Request:

```json
{
  "prompt": "a robot dog with antennae",
  "model":  "anigen",
  "style":  "character",
  "settings": {"t2i": {"custom_suffix": "cinematic"}}
}
```

Response:

```json
{
  "enriched": "a robot dog with antennae, 3D character reference turnaround, ..., cinematic",
  "negative": "cross-legged, legs crossed, ..., text, watermark",
  "style":    "character",
  "aspect":   "portrait",
  "t2i":      "hunyuandit"
}
```

## Generation endpoints

All three return `{"job_id": "<12-char hex>"}`. Poll
`GET /api/status/{job_id}` to follow progress.

### `POST /api/generate` — text → image → 3D

JSON body:

| Field      | Type   | Required | Default                 |
| ---------- | ------ | -------- | ----------------------- |
| `prompt`   | string | yes      | —                       |
| `model`    | string | no       | `"anigen"`              |
| `style`    | string | no       | `"character"`           |
| `pose`     | string | no       | `""` (none)             |
| `settings` | object | no       | `{}` (use server defaults) |

`model` ∈ `{"anigen", "hunyuan3d", "hunyuan3dmv"}`.
`style` ∈ `{"character", "creature", "object", "stylized", "realistic"}`.
`pose` is one of the keys returned from `/api/poses`, or `""` for free-form.

Special behavior for `model=hunyuan3dmv`: HunyuanDiT generates the front view,
Zero123++ synthesizes the other 3 views, then Hy3D-2mv fuses them.

### `POST /api/generate_image` — uploaded image → 3D (skip T2I)

Multipart form:

| Field      | Type   | Required |
| ---------- | ------ | -------- |
| `image`    | file   | yes      |
| `model`    | string | no       |
| `settings` | string | no       |
| `note`     | string | no       |

`settings` is a JSON-encoded string. `note` is just stored on the job for
UI display and doesn't affect generation.

For `model=hunyuan3dmv`, the uploaded image becomes the front view and
Zero123++ generates back/left/right.

### `POST /api/generate_multiview` — multi-view → mesh (no T2I, no Zero123)

Multipart form:

| Field         | Type | Required |
| ------------- | ---- | -------- |
| `image_front` | file | yes      |
| `image_back`  | file | no       |
| `image_left`  | file | no       |
| `image_right` | file | no       |
| `settings`    | string | no     |
| `note`        | string | no     |

If only `front` is given, Zero123++ fills the gaps. If 2-3 views are given,
only the missing slots are synthesized. If all 4 are given, no Zero123 step.

### `POST /api/generate_views` — Zero123++ preview only

Synthesize and return URLs to back/left/right views from a single front
image, without running shape or paint.

Multipart form:

| Field   | Type | Required |
| ------- | ---- | -------- |
| `image` | file | yes      |
| `seed`  | int  | no       |

Response:

```json
{
  "preview_id": "abc123def456",
  "urls": {
    "front": "/results/preview_abc123def456/front.png",
    "back":  "/results/preview_abc123def456/back.png",
    "left":  "/results/preview_abc123def456/left.png",
    "right": "/results/preview_abc123def456/right.png"
  }
}
```

The UI uses this for the **🎲 Generate other views** button. Client fetches
the resulting images back as Blobs and submits them as regular uploads to
`/api/generate_multiview` for the actual mesh build.

## Job lifecycle

### `GET /api/status/{job_id}`

Returns the current state of a job. Status progresses through:

| Status      | Meaning                                                                    |
| ----------- | -------------------------------------------------------------------------- |
| `queued`    | Created, waiting for `gen_lock` (another job is in flight)                 |
| `image`     | Running T2I (or saving uploaded/multi-view input)                          |
| `views`     | Running Zero123++ to synthesize novel views (multi-view path only)         |
| `mesh`      | Running the 3D model (AniGen / Hy3D shape+paint / Hy3D-2mv shape+paint)    |
| `done`      | Finished successfully; `glb_url` and possibly `skeleton_url` are populated |
| `error`     | Failed; `error` field has the exception message                            |
| `cancelled` | User pressed Cancel before the job finished a stage boundary               |

Example response (mid-flight Hunyuan3D-2mv job):

```json
{
  "status":           "views",
  "prompt":           "a small orange cat, 3D render",
  "enriched_prompt":  "a small orange cat, 3D render, 3D character reference turnaround, ...",
  "negative_prompt":  "cross-legged, ...",
  "style":            "character",
  "aspect":           "portrait",
  "model":            "hunyuan3dmv",
  "t2i":              "hunyuandit",
  "pose":             "tpose",
  "settings":         {...},
  "created":          1776912345.6,
  "seed":             1109017497,
  "preview_url":      "/results/abc123def456/input.png",
  "views_urls": {
    "front": "/results/abc123def456/view_front.png",
    "right": "/results/abc123def456/view_right.png",
    "back":  "/results/abc123def456/view_back.png",
    "left":  "/results/abc123def456/view_left.png"
  }
}
```

When `done`:

```json
{
  "status":        "done",
  "glb_url":       "/results/abc123def456/mesh.glb",
  "skeleton_url":  "/results/abc123def456/skeleton.glb",  // AniGen only
  "finished":      1776912543.2,
  ...
}
```

### `POST /api/cancel/{job_id}`

Request graceful cancellation. Cancellation is honored at stage boundaries —
the running stage finishes but subsequent stages are skipped. For Hunyuan3D
jobs, cancelling during shape gives you the untextured mesh and skips paint
(saves ~60 s).

```json
{"ok": true, "note": "cancel will apply at next stage boundary"}
```

## Static files

* `/static/*` — JS, CSS, pose PNGs
* `/results/<job_id>/*` — generated artifacts (input images, mesh.glb, skeleton.glb, view_*.png)
* `/results/preview_<id>/*` — Zero123 preview output
* `/` — the main UI (`static/index.html`)

GLB files are served with `Content-Type: model/gltf-binary` so browsers
recognize them.

## Concurrency

A single `asyncio.Lock` (`gen_lock`) serializes generation work. New jobs are
accepted (`status: queued`) but won't actually run until the previous job
releases the lock. There's no explicit queue depth limit — the server will
accept arbitrary `POST /api/generate` calls and queue them.

If you need higher throughput, the bottleneck is single-process: split into
multiple worker processes pinned to different sets of GPUs, or run multiple
servers on different ports behind a load balancer.
