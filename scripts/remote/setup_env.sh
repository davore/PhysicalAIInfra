#!/usr/bin/env bash
# One-time (or repair) remote environment for robogate replay.
# Intended to run on microbo-gpu after sync.sh push.
set -euo pipefail

ROOT="${ROBOGATE_ROOT:-/root/work/PhysicalAIInfra}"
PREFIX="${ROBOGATE_CONDA_PREFIX:-/root/autodl-tmp/conda-envs/robogate}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/root/autodl-tmp/conda-pkgs}"
export PIP_NO_CACHE_DIR=1
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/root/autodl-tmp/hf/lerobot}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONNOUSERSITE=1

source /root/miniconda3/etc/profile.d/conda.sh

if [[ ! -x "${PREFIX}/bin/python" ]]; then
  conda create -y -p "${PREFIX}" python=3.11
fi
# shellcheck disable=SC1091
conda activate "${PREFIX}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  conda install -y -c conda-forge ffmpeg
fi

if ! python - <<'PY'
import importlib.util
import sys
if importlib.util.find_spec("torch") is None:
    sys.exit(1)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
then
  pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

cd "${ROOT}"
# Editable install first. Do NOT `pip install -e ".[lerobot]"` on this box:
# lerobot's extras pin torch<2.11, which breaks RTX 5090 (sm_120) and fills the overlay.
pip install --no-cache-dir -e .
python - <<'PY' || pip install --no-cache-dir --no-deps "lerobot>=0.4"
import lerobot
print("lerobot already present")
PY
pip install --no-cache-dir --no-deps torchcodec || true
pip cache purge || true
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
import lerobot
print("lerobot", getattr(lerobot, "__version__", "ok"))
PY
df -h / /root/autodl-tmp || true
