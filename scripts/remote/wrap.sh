#!/usr/bin/env bash
# Start a remote robogate module, write jobs/<id>/status.json, and point jobs/current.
# Usage: bash scripts/remote/wrap.sh replay <scenario> [args...]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: wrap.sh <module> [args...]" >&2
  exit 2
fi

MODULE="$1"
shift
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

ID="$(date -u +%Y%m%dT%H%M%SZ)-${MODULE}-$$"
JOB="${ROOT}/jobs/${ID}"
mkdir -p "${JOB}"
ln -sfn "${JOB}" "${ROOT}/jobs/current"

STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PREFIX="${ROBOGATE_CONDA_PREFIX:-/root/autodl-tmp/conda-envs/robogate}"
if [[ -x "${PREFIX}/bin/python" ]]; then
  PYBIN="${PREFIX}/bin/python"
elif [[ -x /root/miniconda3/bin/python ]]; then
  PYBIN=/root/miniconda3/bin/python
else
  PYBIN=python3
fi
write_status() {
  local state="$1" finished="$2" code="$3"
  shift 3
  "${PYBIN}" - "${state}" "${MODULE}" "${STARTED}" "${finished}" "${code}" "$@" <<'PY'
import json
import sys

state, module, started, finished, code, *argv = sys.argv[1:]
payload = {
    "state": state,
    "module": module,
    "started": started,
    "finished": None if finished == "null" else finished,
    "exit_code": None if code == "null" else int(code),
    "argv": argv,
}
print(json.dumps(payload, indent=2))
PY
}
write_status running null null "$@" > "${JOB}/status.json"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
fi
if [[ -x "${PREFIX}/bin/python" ]]; then
  conda activate "${PREFIX}"
elif conda env list 2>/dev/null | grep -qE '^robogate\s'; then
  conda activate robogate
fi
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/root/autodl-tmp/hf/lerobot}"
# AutoDL often has IPv6 to huggingface.co unreachable; mirror stays on IPv4.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# Xet CAS talks to huggingface.co and 401s on AutoDL; force regular Hub downloads.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONNOUSERSITE=1

LOG="${JOB}/${MODULE}.log"
set +e
# robogate is on PATH after conda activate + editable install
robogate "${MODULE}" "$@" > >(tee "${LOG}") 2>&1
CODE=$?
set -e
FINISHED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STATE="failed"
if [[ "${CODE}" -eq 0 ]]; then STATE="ok"; fi
write_status "${STATE}" "${FINISHED}" "${CODE}" "$@" > "${JOB}/status.json"

case "${MODULE}" in
  replay) echo "ROBOGATE_REPLAY_DONE" | tee -a "${LOG}" ;;
  eval) echo "ROBOGATE_EVAL_DONE" | tee -a "${LOG}" ;;
  sim) echo "ROBOGATE_SIM_DONE" | tee -a "${LOG}" ;;
  cluster) echo "ROBOGATE_CLUSTER_DONE" | tee -a "${LOG}" ;;
esac

exit "${CODE}"
