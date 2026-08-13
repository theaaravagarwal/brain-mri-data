#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "This installer requires x86-64 Linux." >&2
  exit 1
fi

if [[ ! -r /etc/os-release ]]; then
  echo "Cannot identify the Linux distribution." >&2
  exit 1
fi

source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
  echo "Supported host: Ubuntu 24.04. Detected: ${PRETTY_NAME:-unknown}." >&2
  exit 1
fi

for command_name in uv rocminfo; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing $command_name. Complete docs/amd-rocm-setup.md first." >&2
    exit 1
  fi
done

if ! rocminfo 2>/dev/null | grep -q "gfx1100"; then
  echo "ROCm does not expose the RX 7900 XT (expected gfx1100)." >&2
  exit 1
fi

uv python install 3.12
uv venv .venv-train --python 3.12

TRAIN_PYTHON=".venv-train/bin/python"
ROCM_WHEEL_ROOT="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1"

uv pip install --python "$TRAIN_PYTHON" -e ".[qc]" "numpy==1.26.4"
uv pip install --python "$TRAIN_PYTHON" \
  "$ROCM_WHEEL_ROOT/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl" \
  "$ROCM_WHEEL_ROOT/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl"
uv pip install --python "$TRAIN_PYTHON" "monai==1.6.0"

"$TRAIN_PYTHON" scripts/verify_amd_training.py
