#!/usr/bin/env bash
# Train ACT on ep10-49 and snapshot 2k / 10k / 30k (or ACT_SAVE_STEPS).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
PREFIX="${ROBOGATE_CONDA_PREFIX:-/root/autodl-tmp/conda-envs/robogate}"
export PATH="${PREFIX}/bin:${PATH}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/root/autodl-tmp/hf/lerobot}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONNOUSERSITE=1
export ACT_SAVE_STEPS="${ACT_SAVE_STEPS:-2000,10000,30000}"
export ACT_OUT="${ACT_OUT:-/root/autodl-tmp/ckpts/v2}"
export ACT_WORKERS="${ACT_WORKERS:-4}"
export ACT_BATCH="${ACT_BATCH:-8}"
export ACT_BUFFER="${ACT_BUFFER:-64}"
export ACT_SKIP_LEROBOT_TRAIN="${ACT_SKIP_LEROBOT_TRAIN:-1}"

ID="$(date -u +%Y%m%dT%H%M%SZ)-train-$$"
JOB="${ROOT}/jobs/${ID}"
mkdir -p "${JOB}"
ln -sfn "${JOB}" "${ROOT}/jobs/current"

if [[ "${ACT_SKIP_LEROBOT_TRAIN}" != "1" ]] && python - <<'PY'
import sys
try:
    import lerobot.scripts.train  # noqa: F401
except Exception as exc:
    print("lerobot-train import failed:", type(exc).__name__, exc)
    sys.exit(1)
print("lerobot-train import ok")
PY
then
  set +e
  lerobot-train \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_static_coffee \
    --dataset.root="${HF_LEROBOT_HOME}/lerobot/aloha_static_coffee" \
    --dataset.episodes="[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]" \
    --wandb.enable=false \
    --save_freq=2000 \
    --steps=30000 \
    --output_dir="${ACT_OUT}/act-coffee-lerobot-train" \
    > >(tee "${JOB}/train.log") 2>&1
  CODE=$?
  set -e
  if [[ "${CODE}" -eq 0 ]]; then
    echo "[train] lerobot-train finished"
    exit 0
  fi
  echo "[train] lerobot-train failed (${CODE}); falling back to train_act.py"
fi

python "${ROOT}/scripts/remote/train_act.py" > >(tee "${JOB}/train.log") 2>&1
echo "[train] done"
