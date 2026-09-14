#!/usr/bin/env bash
# Wait for trained checkpoints, then run GPU evals.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
T30K="${T30K:-/root/autodl-tmp/ckpts/act-coffee-30000}"
echo "[bench] waiting for ${T30K}"
while [[ ! -f "${T30K}/config.json" ]]; do
  sleep 60
done
sleep 20
echo "[bench] checkpoints ready"
bash "${ROOT}/scripts/bench/run_bench.sh"
echo "ROBOGATE_BENCH_DONE"
