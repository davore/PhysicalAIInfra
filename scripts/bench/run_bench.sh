#!/usr/bin/env bash
# Run GPU evals for the gate benchmark. Intended on microbo-gpu after train_act.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
PREFIX="${ROBOGATE_CONDA_PREFIX:-/root/autodl-tmp/conda-envs/robogate}"
export PATH="${PREFIX}/bin:${PATH}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/root/autodl-tmp/hf/lerobot}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONNOUSERSITE=1

SUITE="${SUITE:-scenarios/real}"
RUNS="${RUNS:-runs}"
OUT="${OUT:-results}"
SLICE="${SLICE:-slices}"
DEVICE="${DEVICE:-cuda}"
PUBLISHED="${PUBLISHED:-gozdebaydogmus/act-aloha-static-coffee-test}"
PUB_REV="${PUB_REV:-646846823c8473712f689f59bdd03f798a688f6a}"
T2K="${T2K:-/root/autodl-tmp/ckpts/v2/act-coffee-2000}"
T10K="${T10K:-/root/autodl-tmp/ckpts/v2/act-coffee-10000}"
T30K="${T30K:-/root/autodl-tmp/ckpts/v2/act-coffee-30000}"
BASE_CKPT="${BASE_CKPT:-${PUBLISHED}}"
BASE_REV="${BASE_REV:-${PUB_REV}}"

eval_one() {
  local id="$1"
  shift
  echo "[bench] eval ${id}"
  robogate eval "${SUITE}" --runs "${RUNS}" --out "${OUT}" --slice-dir "${SLICE}" \
    --eval-id "${id}" "$@"
}

# Replay trained checkpoints first so tracking_judge can pick B' vs published B.
if [[ -d "${T2K}" ]]; then
  eval_one T2k --replay --device "${DEVICE}" --adapter lerobot --checkpoint "${T2K}"
fi
if [[ -d "${T10K}" ]]; then
  eval_one T10k --replay --device "${DEVICE}" --adapter lerobot --checkpoint "${T10K}"
fi
if [[ -d "${T30K}" ]]; then
  eval_one T30k --replay --device "${DEVICE}" --adapter lerobot --checkpoint "${T30K}"
  python scripts/bench/tracking_judge.py "${SUITE}" --runs "${RUNS}" \
    --version-contains "act-coffee-30000" | tee "${OUT}/tracking.json"
  if python - "${OUT}/tracking.json" <<'PY'
import json
import sys
from pathlib import Path
raise SystemExit(0 if json.loads(Path(sys.argv[1]).read_text())["ok"] else 1)
PY
  then
    BASE_CKPT="${T30K}"
    BASE_REV=""
    echo "[bench] tracking ok; baseline B' = ${T30K}"
  else
    echo "[bench] T30k does not track; baseline stays published B"
  fi
fi

# Baseline B (or B') replay. Thresholds may still be extract defaults.
if [[ -n "${BASE_REV}" ]]; then
  eval_one B --replay --device "${DEVICE}" --adapter lerobot \
    --checkpoint "${BASE_CKPT}" --revision "${BASE_REV}"
else
  eval_one B --replay --device "${DEVICE}" --adapter lerobot \
    --checkpoint "${BASE_CKPT}"
fi

python scripts/bench/write_thresholds.py "${SUITE}" --eval "${OUT}/B.parquet" \
  --runs "${RUNS}" --slices "${SLICE}" --eval-id B

# Re-assert B with calibrated thresholds (no replay).
eval_one B-calib --device "${DEVICE}" --adapter lerobot

# False-alarm: same checkpoint replayed again.
if [[ -n "${BASE_REV}" ]]; then
  eval_one B-rerun --replay --device "${DEVICE}" --adapter lerobot \
    --checkpoint "${BASE_CKPT}" --revision "${BASE_REV}"
else
  eval_one B-rerun --replay --device "${DEVICE}" --adapter lerobot \
    --checkpoint "${BASE_CKPT}"
fi

# Known-worse input perturbations on the chosen baseline.
if [[ -n "${BASE_REV}" ]]; then
  eval_one B-drop-cam --replay --device "${DEVICE}" --adapter lerobot \
    --checkpoint "${BASE_CKPT}" --revision "${BASE_REV}" \
    --perturb drop_camera=cam_high
  eval_one B-state-noise --replay --device "${DEVICE}" --adapter lerobot \
    --checkpoint "${BASE_CKPT}" --revision "${BASE_REV}" \
    --perturb state_noise=0.05
else
  eval_one B-drop-cam --replay --device "${DEVICE}" --adapter lerobot \
    --checkpoint "${BASE_CKPT}" \
    --perturb drop_camera=cam_high
  eval_one B-state-noise --replay --device "${DEVICE}" --adapter lerobot \
    --checkpoint "${BASE_CKPT}" \
    --perturb state_noise=0.05
fi

# Always report the published checkpoint if it is not the baseline.
if [[ "${BASE_CKPT}" != "${PUBLISHED}" ]]; then
  eval_one B-published --replay --device "${DEVICE}" --adapter lerobot \
    --checkpoint "${PUBLISHED}" --revision "${PUB_REV}"
fi

echo "ROBOGATE_BENCH_EVAL_DONE"
