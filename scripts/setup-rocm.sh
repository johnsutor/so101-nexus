#!/usr/bin/env bash
# setup-rocm.sh — create .venv-rocm with ROCm 7.2 PyTorch
#
# Usage:
#   ./scripts/setup-rocm.sh          # create venv and install all deps
#   ./scripts/setup-rocm.sh --sync   # re-sync only (venv must exist)
set -euo pipefail

ROCM_VENV=".venv-rocm"
PYTHON="${PYTHON:-python3.12}"

info()  { printf '\033[1;34m%s\033[0m\n' "$*"; }
die()   { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

command -v uv >/dev/null || die "uv not found — https://docs.astral.sh/uv/getting-started/installation/"

SYNC_ONLY=false
[[ "${1:-}" == "--sync" ]] && SYNC_ONLY=true

# ── create venv if needed ───────────────────────────────────────────────
if [[ ! -d "$ROCM_VENV" ]]; then
    info "Creating $ROCM_VENV ($PYTHON) ..."
    uv venv "$ROCM_VENV" --python "$PYTHON"
fi

PY="$ROCM_VENV/bin/python"

# ── install project + extras (no-deps first to avoid pulling CUDA torch) ──
info "Installing so101-nexus[teleop,train,warp] (editable, no-deps) ..."
uv pip install --python "$PY" --no-deps -e ".[teleop,train,warp]"

# ── install remaining deps (excluding torch) ────────────────────────────
info "Installing remaining dependencies ..."
uv pip install --python "$PY" \
    numpy huggingface_hub trimesh scipy "mujoco>=3.1.3" "gymnasium>=1.0.0" "tyro>=0.9.0" \
    "tensorboard>=2.0.0" "wandb[media]>=0.16.0" \
    "mujoco-warp>=3.9.0.1,<3.10" \
    "lerobot[feetech]>=0.5.0,<0.6" "gradio>=5.0.0" "plotly>=6.0.0" "opencv-python>=4.8.0" \
    "transformers>=5.8" "accelerate"

# ── install torch (ROCm 7.2) LAST, force-reinstall to override any CUDA ─
info "Installing torch via --torch-backend rocm7.2 ..."
uv pip install --python "$PY" --torch-backend rocm7.2 --force-reinstall torch torchvision

# ── install dev/test deps ──────────────────────────────────────────────
info "Installing dev/test deps ..."
uv pip install --python "$PY" \
    pytest pytest-cov hypothesis "imageio[ffmpeg]" Pillow \
    ruff ty litellm

# ── verify ──────────────────────────────────────────────────────────────
info "Verifying torch ..."
"$PY" -c "
import torch
v = torch.__version__
hip = getattr(torch.version, 'hip', None)
print(f'torch {v}' + (f'  ROCm/HIP {hip}' if hip else '  WARNING: HIP not detected'))
"

info "Done.  Activate:  source $ROCM_VENV/bin/activate"
