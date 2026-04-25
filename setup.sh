#!/usr/bin/env bash
# 3D Model Builder — one-command install
#
# Usage:
#   ./setup.sh                    # full install + weight prefetch
#   ./setup.sh --skip-weights     # install, skip the ~75 GB HF download
#   ./setup.sh --skip-anigen      # if AniGen is already installed
#   ./setup.sh --skip-hunyuan3d   # if Hunyuan3D-2 is already installed
#
# Tested on Ubuntu 24.04 + CUDA 12.x + V100 (Volta SM 7.0) and confirmed to
# work on Ampere+ as well. Detects GPU compute capability and rebuilds CUDA
# extensions accordingly.
#
# Re-runnable. Each step checks for prior completion before doing work.

set -euo pipefail

SKIP_WEIGHTS=0
SKIP_ANIGEN=0
SKIP_HUNYUAN3D=0
ENV_NAME="anigen"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-weights)   SKIP_WEIGHTS=1 ;;
    --skip-anigen)    SKIP_ANIGEN=1 ;;
    --skip-hunyuan3d) SKIP_HUNYUAN3D=1 ;;
    --env)            ENV_NAME="$2"; shift ;;
    -h|--help)
      sed -n '2,15p' "$0"; exit 0 ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
  shift
done

log() { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[error]\033[0m %s\n' "$*"; exit 1; }

# ─── Preflight ──────────────────────────────────────────────────────────────

log "checking prerequisites..."
command -v conda  >/dev/null 2>&1 || die "conda not found. Install miniconda or anaconda first."
command -v git    >/dev/null 2>&1 || die "git not found."
command -v nvcc   >/dev/null 2>&1 || warn "nvcc not on PATH — CUDA Toolkit may not be properly installed. Continuing."
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. Need NVIDIA driver."

if [[ -z "${CONDA_EXE:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  CONDA_BASE=$(conda info --base)
else
  CONDA_BASE=$(dirname "$(dirname "$(readlink -f "${CONDA_EXE:-$(which conda)}")")")
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Detect GPU compute capability for arch-specific builds.
SM_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
if [[ -z "$SM_CAP" ]]; then
  SM_CAP=86  # safe Ampere default if query fails
fi
TORCH_CUDA_ARCH_LIST="$(echo "$SM_CAP" | sed 's/\(.\)\(.\)/\1.\2/')"
log "detected GPU compute capability: $TORCH_CUDA_ARCH_LIST (SM $SM_CAP)"
export TORCH_CUDA_ARCH_LIST

# ─── Conda env ──────────────────────────────────────────────────────────────

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "conda env '$ENV_NAME' already exists"
else
  log "creating conda env '$ENV_NAME' (python 3.10, conda-forge)..."
  # Use conda-forge to avoid the conda 26 default-channel TOS gate.
  conda create -n "$ENV_NAME" -c conda-forge --override-channels python=3.10 -y
fi
conda activate "$ENV_NAME"
log "active env: $(python --version) at $(which python)"

# ─── AniGen ─────────────────────────────────────────────────────────────────

if [[ "$SKIP_ANIGEN" -eq 0 ]]; then
  if [[ ! -d "$HOME/AniGen" ]]; then
    log "cloning AniGen..."
    git clone --recurse-submodules https://github.com/VAST-AI-Research/AniGen.git "$HOME/AniGen"
  else
    log "AniGen already cloned at ~/AniGen"
  fi
  cd "$HOME/AniGen"
  if ! python -c "import anigen" >/dev/null 2>&1; then
    log "installing AniGen deps (torch + basic + demo + xformers, NO flash-attn)..."
    # Flash-attn 2 isn't supported on Volta; xformers + SDPA cover attention.
    # shellcheck disable=SC1091
    source ./setup.sh --torch --basic --demo --xformers
  else
    log "AniGen Python package already importable"
  fi
fi

# ─── nvdiffrast: arch-specific rebuild ──────────────────────────────────────

if ! python -c "import nvdiffrast.torch as dr; import torch; ctx = dr.RasterizeCudaContext(device='cuda:0')" >/dev/null 2>&1; then
  log "rebuilding nvdiffrast from source for SM $SM_CAP..."
  pip uninstall nvdiffrast -y >/dev/null 2>&1 || true
  pip install "https://github.com/NVlabs/nvdiffrast/archive/refs/tags/v0.3.3.tar.gz" --no-build-isolation
else
  log "nvdiffrast OK (kernel loaded for current GPU)"
fi

# ─── Hunyuan3D-2 ────────────────────────────────────────────────────────────

if [[ "$SKIP_HUNYUAN3D" -eq 0 ]]; then
  if [[ ! -d "$HOME/Hunyuan3D-2" ]]; then
    log "cloning Hunyuan3D-2..."
    git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git "$HOME/Hunyuan3D-2"
  fi
  cd "$HOME/Hunyuan3D-2"
  if ! python -c "import hy3dgen" >/dev/null 2>&1; then
    log "installing Hunyuan3D-2 Python package..."
    pip install -r requirements.txt
    pip install -e .
  fi
  # Custom CUDA extensions: must be built against the *currently active* torch.
  log "building Hunyuan3D-2 custom CUDA extensions for SM $SM_CAP..."
  cd hy3dgen/texgen/custom_rasterizer
  TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" python setup.py install
  cd ../differentiable_renderer
  TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" python setup.py install
  cd "$HOME/Hunyuan3D-2"

  # Hunyuan3D-2 may have downgraded torch — re-pin to the version xformers/
  # pytorch3d expect. Then rebuild Hy3D extensions against the upgraded torch.
  TORCH_INSTALLED=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
  if [[ ! "$TORCH_INSTALLED" =~ ^2\.5\. ]]; then
    log "torch is $TORCH_INSTALLED, upgrading to 2.5.1+cu121..."
    pip install torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cu121
    log "rebuilding Hunyuan3D-2 extensions against torch 2.5.1..."
    cd "$HOME/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer"
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" python setup.py install
    cd "$HOME/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer"
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" python setup.py install
  fi
fi

# ─── Server-only deps from requirements.txt ─────────────────────────────────

log "installing server-only deps from requirements.txt..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "$SCRIPT_DIR/requirements.txt"

# ─── HF weight prefetch ─────────────────────────────────────────────────────

if [[ "$SKIP_WEIGHTS" -eq 0 ]]; then
  log "pre-fetching HuggingFace weights (~75 GB total, this can take a while)..."
  python - <<'PY'
from huggingface_hub import snapshot_download
import sys
repos = [
    "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers",
    "Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Pose",
    "tencent/Hunyuan3D-2",
    "tencent/Hunyuan3D-2mv",
    "sudo-ai/zero123plus-v1.2",
    "briaai/RMBG-1.4",
]
for r in repos:
    print(f"  fetching {r}...", flush=True)
    try:
        snapshot_download(r)
    except Exception as e:
        print(f"    WARNING: {type(e).__name__}: {e}", file=sys.stderr)
print("done.")
PY
else
  log "skipping HF weight prefetch (will download on first server start)"
fi

# ─── Pose gallery ───────────────────────────────────────────────────────────

if [[ -f "$SCRIPT_DIR/pose_gallery.py" ]]; then
  log "rendering pose gallery..."
  cd "$SCRIPT_DIR"
  mkdir -p static/poses
  python pose_gallery.py static/poses
fi

# ─── Done ───────────────────────────────────────────────────────────────────

log "done. Verify the install with: python preflight.py"
log "to launch the server: python app.py"
echo
echo "Server will bind 0.0.0.0:9000. Open http://localhost:9000 in a browser."
