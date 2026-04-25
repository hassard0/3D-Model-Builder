#!/usr/bin/env bash
# Container entrypoint — does first-run install on /data, then execs server.
set -euo pipefail

log() { printf '\n\033[1;34m[entrypoint]\033[0m %s\n' "$*"; }

# /data is the persistent volume. AniGen + Hunyuan3D + HF weights all live
# there so the image stays small and rebuilds don't lose installed deps.
DATA=/data
mkdir -p "$DATA"
export HF_HOME="$DATA/huggingface"
export TRANSFORMERS_CACHE="$DATA/huggingface"

# GPU compute capability for arch-specific builds.
SM_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
[[ -z "$SM_CAP" ]] && SM_CAP=86
export TORCH_CUDA_ARCH_LIST="$(echo "$SM_CAP" | sed 's/\(.\)\(.\)/\1.\2/')"
log "GPU compute capability: $TORCH_CUDA_ARCH_LIST (SM $SM_CAP)"

# ── AniGen ────────────────────────────────────────────────────────────────
if [[ ! -d "$DATA/AniGen" ]]; then
  log "first-run install: cloning AniGen into /data..."
  git clone --recurse-submodules https://github.com/VAST-AI-Research/AniGen.git "$DATA/AniGen"
fi
if ! python -c "import anigen" >/dev/null 2>&1; then
  log "installing AniGen Python package..."
  cd "$DATA/AniGen"
  # No flash-attn — Volta (and our base torch) don't need it.
  bash -c "source ./setup.sh --torch --basic --demo --xformers"
fi

# ── nvdiffrast for current SM ─────────────────────────────────────────────
if ! python -c "import nvdiffrast.torch as dr; import torch; dr.RasterizeCudaContext(device='cuda:0')" >/dev/null 2>&1; then
  log "rebuilding nvdiffrast from source for SM $SM_CAP..."
  pip uninstall nvdiffrast -y >/dev/null 2>&1 || true
  pip install "https://github.com/NVlabs/nvdiffrast/archive/refs/tags/v0.3.3.tar.gz" --no-build-isolation
fi

# ── Hunyuan3D-2 ───────────────────────────────────────────────────────────
if [[ ! -d "$DATA/Hunyuan3D-2" ]]; then
  log "first-run install: cloning Hunyuan3D-2 into /data..."
  git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git "$DATA/Hunyuan3D-2"
fi
if ! python -c "import hy3dgen" >/dev/null 2>&1; then
  cd "$DATA/Hunyuan3D-2"
  pip install -r requirements.txt
  pip install -e .
  cd hy3dgen/texgen/custom_rasterizer
  python setup.py install
  cd ../differentiable_renderer
  python setup.py install
fi

# Re-pin torch in case Hunyuan3D-2's deps downgraded it.
TORCH_V=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [[ ! "$TORCH_V" =~ ^2\.5\. ]]; then
  log "torch is $TORCH_V, restoring to 2.5.1..."
  pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
  cd "$DATA/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer"
  python setup.py install
  cd "$DATA/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer"
  python setup.py install
fi

# ── HF weight prefetch (skip via SKIP_WEIGHTS=1) ──────────────────────────
if [[ "${SKIP_WEIGHTS:-0}" != "1" ]] && [[ ! -f "$DATA/.weights_cached" ]]; then
  log "pre-fetching HF weights into $HF_HOME (~75 GB)..."
  python - <<'PY'
from huggingface_hub import snapshot_download
import sys
for r in [
    "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers",
    "Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Pose",
    "tencent/Hunyuan3D-2",
    "tencent/Hunyuan3D-2mv",
    "sudo-ai/zero123plus-v1.2",
    "briaai/RMBG-1.4",
]:
    print(f"  {r}", flush=True)
    try:
        snapshot_download(r)
    except Exception as e:
        print(f"    WARN {type(e).__name__}: {e}", file=sys.stderr)
PY
  touch "$DATA/.weights_cached"
fi

# ── Pose gallery ──────────────────────────────────────────────────────────
mkdir -p /app/static/poses
if [[ ! -f /app/static/poses/tpose.png ]]; then
  log "rendering pose gallery..."
  cd /app && python pose_gallery.py static/poses
fi

# ── Run ───────────────────────────────────────────────────────────────────
log "running preflight..."
cd /app && python preflight.py || log "preflight returned non-zero — server may still start; review the warnings above"

log "starting server on 0.0.0.0:9000..."
exec python /app/app.py
