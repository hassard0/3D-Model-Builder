# Deploying 3D Model Builder

Tested target: **Ubuntu 24.04, 4× V100-SXM2 16 GB (Volta, SM 7.0), CUDA 12.x**.
Other configurations should work but expect to revisit version pins.

## 1. Conda env

Conda 26 gates the default channels behind a Terms-of-Service accept; either
accept it or use `conda-forge` only:

```bash
conda create -n anigen -c conda-forge --override-channels python=3.10 -y
conda activate anigen
```

## 2. Clone AniGen and run its installer (no `--new-env`, no `--flash-attn`)

```bash
git clone --recurse-submodules https://github.com/VAST-AI-Research/AniGen.git
cd AniGen
source ./setup.sh --torch --basic --demo --xformers
```

`--all` would include `--flash-attn`, which **does not work on Volta** —
flash-attn-2 requires SM 8.0+. xformers + SDPA cover attention fine on V100.

## 3. Volta-specific rebuilds

The prebuilt nvdiffrast wheel that AniGen's setup pulls is Ampere-only. On
V100 it fails at `cudaFuncGetAttributes(&attr, (void*)fineRasterKernel)`
(CUDA error 209). Rebuild from source — the source path uses JIT compilation
which auto-targets your GPU's SM:

```bash
pip uninstall nvdiffrast -y
pip install https://github.com/NVlabs/nvdiffrast/archive/refs/tags/v0.3.3.tar.gz \
            --no-build-isolation
```

## 4. Hunyuan3D-2

```bash
git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git ~/Hunyuan3D-2
cd ~/Hunyuan3D-2
pip install -r requirements.txt
pip install -e .
cd hy3dgen/texgen/custom_rasterizer
TORCH_CUDA_ARCH_LIST=7.0 python setup.py install
cd ../differentiable_renderer
TORCH_CUDA_ARCH_LIST=7.0 python setup.py install
```

Hunyuan3D-2's `requirements.txt` may pull torch 2.4.0; restore the version
xformers/pytorch3d expect:

```bash
pip install torch==2.5.1 torchvision==0.20.1 \
            --index-url https://download.pytorch.org/whl/cu121
# Rebuild Hunyuan3D's extensions against the upgraded torch
cd ~/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer && TORCH_CUDA_ARCH_LIST=7.0 python setup.py install
cd ~/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer && TORCH_CUDA_ARCH_LIST=7.0 python setup.py install
```

## 5. Server-only Python deps

```bash
pip install fastapi 'uvicorn[standard]' diffusers accelerate \
            python-multipart safetensors rtree tiktoken sentencepiece
```

## 6. Pre-cache HuggingFace weights (optional but speeds first start)

```bash
python -c "from huggingface_hub import snapshot_download; \
  [snapshot_download(r) for r in [
    'VAST-AI/AniGen',
    'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers',
    'Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Pose',
    'tencent/Hunyuan3D-2',
    'tencent/Hunyuan3D-2mv',
    'sudo-ai/zero123plus-v1.2',
    'briaai/RMBG-1.4',
  ]]"
```

About 75 GB of weights total.

## 7. Drop the server files in place

Copy this repo's contents into AniGen/server/:

```
~/AniGen/server/
  app.py
  prompt_harness.py
  pose_gallery.py
  static/
    index.html
    viewer.js
```

Generate the pose gallery once:

```bash
cd ~/AniGen/server
python pose_gallery.py static/poses
```

## 8. Run

```bash
cd ~/AniGen
python server/app.py
```

Or use a launcher script that activates conda + execs the server, plus
nohup it for the background:

```bash
cat > /tmp/run_server.sh <<'EOF'
#!/bin/bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate anigen
cd $HOME/AniGen
exec python server/app.py
EOF
chmod +x /tmp/run_server.sh
nohup /tmp/run_server.sh > /tmp/anigen_server.log 2>&1 &
```

Cold start is ~2 minutes. Watch `/tmp/anigen_server.log` for `[startup] READY`.

## Troubleshooting

| Symptom                                    | Cause                                                                                              | Fix                                                                                            |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `CUDA error 209` from nvdiffrast           | Prebuilt wheel was Ampere-only                                                                     | Rebuild nvdiffrast from source (step 3)                                                        |
| `tiktoken is required` loading HunyuanDiT  | `tiktoken` missing                                                                                 | `pip install tiktoken`                                                                         |
| `Error parsing line b'\x0e' in spiece.model` | T5 tokenizer routed through tiktoken converter because sentencepiece missing                       | `pip install sentencepiece`                                                                    |
| `Due to a serious vulnerability in torch.load` | transformers ≥5.5 requires torch ≥2.6 to load `.bin` files; we're on 2.5.1                         | `app.py` already monkey-patches `check_torch_load_is_safe`                                     |
| `'BriaRMBG' object has no attribute 'all_tied_weights_keys'` | RMBG-1.4 custom code predates the new transformers API                                             | `app.py` adds a class-level default on `PreTrainedModel`                                       |
| `OutOfMemoryError` on cuda:1 at startup     | HunyuanDiT plain + ControlNet variant don't both fit                                               | We removed plain HunyuanDiT; only the ControlNet pipe is loaded. If you re-add: separate cards |
| Hy3D paint exports floating-point alpha textures the browser shows wrong | Browser issue — open in Blender/Maya to verify. Alt: turn off texture in settings.            |                                                                                                |
| Procedural animations rotate the wrong limb | Rig classification got fooled by an unusual topology                                               | Toggle "Show classification" overlay; wrong colors mean the classifier got confused on this rig |
| Mesh feet sink into grid                    | Bbox bottom is hair/cape, not feet                                                                 | Viewer grounds by ankle bones when present; if no legs detected, falls back to bbox              |
