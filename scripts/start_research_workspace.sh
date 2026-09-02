#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint="$repo_root/runs/glioma-pilot--cuda-4060--brats--20260828--e100/best.pt"
expected_checkpoint_sha256="121422a861bbe7affaa5e161058e69eea737b2390651c3c03ea20256969e99e5"

if [[ -x "$repo_root/.tools/node/bin/node" ]]; then
  export PATH="$repo_root/.tools/node/bin:$PATH"
fi

if [[ ! -x "$repo_root/.venv/bin/python" ]]; then
  echo "Pinned Python environment is unavailable; run SETUP_CUDA=1 ./setup.sh first." >&2
  exit 69
fi
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "Node.js and npm are unavailable; install the pinned local Node runtime before starting the workspace." >&2
  exit 69
fi
if [[ ! -f "$checkpoint" ]]; then
  echo "Fixed research checkpoint is unavailable: $checkpoint" >&2
  exit 66
fi
observed_checkpoint_sha256="$(sha256sum "$checkpoint" | awk '{print $1}')"
if [[ "$observed_checkpoint_sha256" != "$expected_checkpoint_sha256" ]]; then
  echo "Fixed research checkpoint digest mismatch." >&2
  exit 74
fi

cd "$repo_root/monitor"
exec npm start
