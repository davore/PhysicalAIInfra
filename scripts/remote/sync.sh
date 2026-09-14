#!/usr/bin/env bash
# Push the repo to microbo-gpu or pull run directories back.
# Usage:
#   bash scripts/remote/sync.sh push
#   bash scripts/remote/sync.sh pull [runs/<id>]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HOST="${REMOTE_HOST:-microbo-gpu}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/work/PhysicalAIInfra}"

mode="${1:-}"
if [[ "${mode}" != "push" && "${mode}" != "pull" ]]; then
  echo "usage: sync.sh push|pull [remote-path]" >&2
  exit 2
fi

RSYNC=(rsync -az --human-readable)
SSH=(ssh -o BatchMode=yes)

if [[ "${mode}" == "push" ]]; then
  "${SSH[@]}" "${HOST}" "mkdir -p '${REMOTE_ROOT}'"
  "${RSYNC[@]}" \
    --delete \
    --exclude '.git/' \
    --exclude '.cache/' \
    --exclude '.pytest_cache/' \
    --exclude '.ruff_cache/' \
    --exclude 'robogate.egg-info/' \
    --exclude '__pycache__/' \
    --exclude 'runs/' \
    --exclude 'jobs/' \
    --exclude 'slices/' \
    --exclude 'results/' \
    --exclude 'untitled folder/' \
    "${ROOT}/" "${HOST}:${REMOTE_ROOT}/"
  echo "pushed ${ROOT} -> ${HOST}:${REMOTE_ROOT}"
  exit 0
fi

src="${2:-runs/}"
src="${src%/}/"
dest="${ROOT}/${src}"
mkdir -p "${dest}"
"${RSYNC[@]}" "${HOST}:${REMOTE_ROOT}/${src}" "${dest}"
echo "pulled ${HOST}:${REMOTE_ROOT}/${src} -> ${dest}"
