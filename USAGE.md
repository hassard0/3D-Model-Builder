# Usage

UI walkthrough and the typical workflows.

## UI map

```
┌─────────────────────────────────────────────────────────────────┐
│ 3D Studio  [model]  [style]  [prompt input]  [Gen] [⚙ ⚙]        │   ← header
├─────────────────────────────────────────────────────────────────┤
│ status line / progress                                          │
├─────────────────────────────────────────────────────────────────┤
│ HunyuanDiT sees: <enriched prompt>                              │   ← debug
├──────────────────────────────────────────────┬──────────────────┤
│                                              │ Model            │
│                                              │ (description +   │
│                                              │  badges)         │
│                                              │                  │
│                                              │ Pose (ControlNet)│
│                                              │ [thumb grid]     │
│             3D viewer                        │                  │
│       (orbit / drag / zoom)                  │ 2D preview       │
│                                              │ [image]          │
│                                              │                  │
│                                              │ Multi-view inputs│
│                                              │ [4-slot grid]    │
│                                              │                  │
│                                              │ Animation        │
│                                              │ [clip selector]  │
│                                              │                  │
│                                              │ Display          │
│                                              │ [toggles]        │
│                                              │                  │
│                                              │ Camera           │
│                                              │ [presets]        │
│                                              │                  │
│                                              │ Lighting         │
│                                              │ [sliders]        │
│                                              │                  │
│                                              │ Rig editor       │
│                                              │ Export           │
└──────────────────────────────────────────────┴──────────────────┘
```

## Workflows

### 1. Text → rigged character (the main flow)

1. Pick **AniGen — rigged** in the model dropdown.
2. Pick the **Character** style.
3. (Optional but recommended) Click **T-pose** in the Pose gallery to force
   the 2D image into a clean rigging-friendly pose via ControlNet.
4. Type a prompt: `a fox warrior in leather armor`.
5. Hit **Generate**.

What happens:

* HunyuanDiT generates a 768×1024 image with your prompt + the character
  style suffix + (optionally) ControlNet T-pose conditioning.
* Server pre-processes the image (RMBG-1.4 → bbox crop → square pad → 1024×1024).
* AniGen produces a rigged GLB (~30 seconds).
* The viewer auto-loads it, frames it on the grid, and populates the
  Animation dropdown with procedural clips.
* Click **Walk** or **Wave** to play.

Total time: ~90 seconds.

### 2. Text → static textured mesh

Same as above but pick **Hunyuan3D-2 — textured (1 image)**. No skeleton,
no animations, but better surface detail and texture quality. Choose this
when the character will be a static render or you'll rig externally.

Total time: ~2-3 minutes (paint stage is the long pole).

### 3. Text → multi-view → mesh

Pick **Hunyuan3D-2mv — multi-view**. Two ways to use it:

**Auto mode** — type a prompt and hit Generate. The server runs HunyuanDiT,
then Zero123++ synthesizes back/left/right from the front, then Hy3D-2mv
fuses all 4. You'll see the Multi-view inputs panel populate with thumbnails
as views are generated.

**Manual mode** — drop your own images into the Front (required), Back, Left,
and Right slots. Front is required; the others are optional and Zero123++
fills any empty slots. Click **🎲 Generate other views** for a preview-only
run that lets you inspect the Zero123 output before committing to the full
mesh build.

### 4. Image upload → 3D

Click the **🖼** icon in the header, pick an image (or drag-drop onto the
viewer). The image appears in the 2D preview pane. Pick the model you want
(AniGen for rigging, Hunyuan3D-2 for static texture, Hunyuan3D-2mv for
auto-multiview). Hit Generate. T2I is skipped.

For best results, upload an image where the subject:

* Is centered
* Faces the camera roughly head-on
* Has a clean / mostly empty background
* Has visible legs (especially for AniGen — see [TROUBLESHOOTING.md](TROUBLESHOOTING.md))

### 5. Cancelling a slow job

Hit the red **Cancel** button next to Generate. The server finishes the
running stage and skips subsequent ones. Most useful for Hunyuan3D-2 paint —
cancel mid-job and you get an untextured mesh in ~50 % of the time.

### 6. Adjusting parameters per job

Click the **⚙** gear icon. The settings modal has three tabs:

* **Text → Image** — guidance scale, PAG scale, steps, prompt suffix override,
  negative override
* **AniGen** — CFG scales (SS and SLAT), joint density, skin smoothing
* **Hunyuan3D-2** — texture toggle, bg-removal toggle, polygon cap

Save persists to localStorage. Reset clears.

See [SETTINGS.md](SETTINGS.md) for what each field controls and recommended
values.

## Viewer interactions

| Action                         | Mouse / Key                                              |
| ------------------------------ | -------------------------------------------------------- |
| Orbit camera                   | left-drag                                                |
| Pan                            | right-drag                                               |
| Zoom                           | scroll                                                   |
| Frame on object                | click **Fit** in Camera section                          |
| Front / 3-4 / Side / Back / Top | click corresponding button                              |
| Toggle skeleton                | **Show skeleton** checkbox                               |
| Toggle classification overlay  | **Show classification** checkbox                         |
| Wireframe                      | **Wireframe** checkbox                                   |
| Hide grid                      | **Grid** checkbox                                        |
| Auto-rotate                    | **Auto-rotate** checkbox                                 |
| Switch lighting preset         | **Env preset** dropdown                                  |
| Tune lights manually           | sliders (Exposure / Ambient / Key / Fill / Rim)          |

### Animations

When AniGen produces a rigged GLB, the Animation dropdown is populated with
procedural clips:

| Clip      | Notes                                          | Requires      |
| --------- | ---------------------------------------------- | ------------- |
| Idle sway | Subtle root rotation, always available         | any rig       |
| Breathe   | Spine scale pulse                              | any rig       |
| Spin      | Full Y-axis rotation                           | any rig       |
| Walk      | Alternating leg swing + counter arm swing      | ≥2 legs detected |
| Run       | Faster walk, bigger amplitude                  | ≥2 legs detected |
| Wave      | Right arm raise + hand waggle                  | ≥1 arm detected |
| Sit       | Hip drop + knee bend (hold)                    | ≥2 legs detected |
| Crouch    | Partial squat (hold)                           | ≥2 legs detected |
| Jump      | Crouch → leap → land                           | ≥2 legs detected |
| Wiggle    | Per-bone phase-offset sine — fallback          | any rig       |
| Dance     | Combined hip + arm + body shimmy               | any rig       |

Pick "(bind pose)" to stop animation and see the rest pose.

### Rig editor

If a generated rig has a misplaced bone, you can fix it directly in the viewer:

1. Toggle **Show classification** on. Colored spheres appear on each bone.
2. Click a sphere to select that bone — a translate gizmo appears.
3. Drag the colored arrows to move the bone. The mesh deforms live via skinning.
4. Press **R** to switch the gizmo to rotation mode; **T** for translate.
5. Press **Esc** to deselect.
6. When happy, click **Download edited GLB** in the Rig editor section to
   save your corrected rig as a new GLB file.

This is hugely valuable for AniGen outputs where, e.g., legs collapse to a
single center column under a long dress — drag them outward to separate.

### Export

* **Download GLB** (in Export section) — the original model from the server
* **Download edited GLB** (in Rig editor section) — your in-browser edited version

Both are standard binary glTF (`.glb`) files compatible with Blender, Maya,
Three.js, Babylon.js, Unity, Unreal.

## Tips

* **For best AniGen rigs**: pick the **T-pose** ControlNet pose AND the
  **Character** style preset. The T-pose forces the 2D image into a clean
  T-pose; the Character style adds prompt cues that emphasize visible legs
  and front-facing composition.

* **Avoid floor-length clothing** in your prompts (dresses, robes, gowns,
  long coats). AniGen needs to see two distinct legs to rig them; if both
  are hidden under fabric, you get a single-leg-stack rig that animates
  poorly. The bone editor can fix this manually.

* **Upload a turnaround sheet** as your front view in multi-view mode and
  let Zero123++ infer the others — often produces cleaner results than
  generating front from scratch.

* **For non-humanoid creatures** (snakes, blobs, props): Walk/Wave/Sit
  won't appear in the dropdown because limbs aren't detected. Use Spin or
  Wiggle instead.

* **For static renders/exports**: turn off **enable_texture** in Hunyuan3D-2
  settings to skip paint and save 60 seconds. The mesh comes out gray; you
  can color it externally.

* **If the 3D viewer feels dim**: bump **Exposure** in Lighting up to ~2.0
  or pick the **Neutral** env preset (more even lighting than Studio).
