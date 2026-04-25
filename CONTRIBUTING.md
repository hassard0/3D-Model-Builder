# Contributing

This is a personal / research project. PRs welcome but expect a high bar
because the dependency stack is fragile and easy to break.

## Development loop

The fastest iteration loop:

1. Edit the local repo.
2. `pscp` (or rsync, or scp) the changed file to the server's
   `~/AniGen/server/` directory.
3. For Python changes: `pkill -9 -f 'server/app.py' && nohup /tmp/run_server.sh > /tmp/anigen_server.log 2>&1 &` to restart. ~2 minute cold boot.
4. For HTML/JS/CSS: just hard-refresh the browser (Ctrl+F5). No server restart needed because they're served as static files.

## Code organization

* `app.py` — keep server-side concerns here. Pipeline loading, route
  handlers, run_job functions, image preprocessing helpers.
* `prompt_harness.py` — only style presets and string-shaping logic. No
  pipeline code.
* `pose_gallery.py` — only the pose-skeleton drawing logic. The output PNGs
  are generated artifacts and should be regenerated on each install.
* `static/index.html` — UI shell, modal HTML, CSS.
* `static/viewer.js` — three.js scene, controls, animations, rig editor,
  fetch logic. Single file by design — splitting into modules adds build
  complexity without enough payoff.

## Style

* Python: stdlib formatting, no autoformatter on commit. PEP 8 with 100-col
  preference.
* JavaScript: vanilla ES6+ modules loaded via `<script type="module">` and
  three.js's import map. No bundler, no transpiler.

## Testing

There's no test suite. Manual checklist for any change touching pipelines:

- [ ] Server starts without errors (`grep -E 'FAILED|Traceback' /tmp/anigen_server.log` returns nothing fatal)
- [ ] `GET /api/health` shows all four model families ready
- [ ] Text-to-AniGen end-to-end produces a rigged GLB
- [ ] Text-to-Hunyuan3D end-to-end produces a textured GLB
- [ ] Text-to-Hunyuan3D-2mv with auto Zero123 produces a textured GLB
- [ ] Image upload → AniGen works
- [ ] Pose ControlNet selection actually changes the 2D output (toggle T-pose vs free)
- [ ] Cancel button stops a job at the next stage boundary
- [ ] Settings persist across page reload
- [ ] Bone-drag rig editor moves bones and exports a valid GLB

## Adding a new model

1. Decide which GPU it lives on. See [ARCHITECTURE.md](ARCHITECTURE.md) §
   "GPU placement strategy".
2. Add globals at the top of `app.py` (`my_model_pipe = None`, `my_model_error = None`).
3. Add a load block in `startup()` inside `try/except` so the server boots
   even if your model fails. Wrap with `with torch.cuda.device(IDX):` if
   the loader doesn't honor `device=`.
4. Update `GET /api/health` to expose readiness.
5. Add to the `model` enum in `GenerateRequest` if it's user-selectable.
6. Add a `_my_model(...)` helper and route to it in `run_job`.
7. Add to the model dropdown in `index.html`.
8. Document in [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md),
   and [API.md](API.md).

## Adding a new pose to the gallery

1. Edit `pose_gallery.py`. Define a function returning an 18-element list of
   normalized `(x, y)` keypoints. Use `tpose()` as a reference.
2. Add an entry to the `POSES` dict.
3. Re-run `python pose_gallery.py static/poses` on the server.
4. The UI picks up new poses on next page load (no server restart needed).

## Reporting issues

Include:

* The browser console output (F12) — especially `[skel]`, `[ground]`,
  `[rig-edit]` lines.
* The server log around the time of the failure (`tail -200 /tmp/anigen_server.log`).
* The job's input image and `mesh.glb` if reproducible.
* `nvidia-smi` output during the issue.
* Browser version and OS.
