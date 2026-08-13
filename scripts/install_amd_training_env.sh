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
  echo "Supported WSL distribution: Ubuntu 24.04. Detected: ${PRETTY_NAME:-unknown}." >&2
  exit 1
fi

if ! grep -qi microsoft /proc/sys/kernel/osrelease; then
  echo "This installer now targets WSL2. Native Linux users need a separate host setup." >&2
  exit 1
fi

if [[ ! -e /dev/dxg ]]; then
  echo "WSL does not expose /dev/dxg. Update WSL and install AMD's matching Adrenalin WSL2 driver in Windows." >&2
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
ROCM_WHEEL_ROOT="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2"

uv pip install --python "$TRAIN_PYTHON" -e ".[qc]" "numpy==1.26.4"
uv pip install --python "$TRAIN_PYTHON" \
  "$ROCM_WHEEL_ROOT/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl" \
  "$ROCM_WHEEL_ROOT/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl"
uv pip install --python "$TRAIN_PYTHON" "monai==1.6.0"

# AMD's WSL instructions require the Windows-provided HSA runtime instead of
# the runtime bundled inside the Linux PyTorch wheel.
TORCH_LIB_DIR="$($TRAIN_PYTHON -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("torch").origin).parent / "lib")')"
find "$TORCH_LIB_DIR" -maxdepth 1 -type f -name 'libhsa-runtime64.so*' -delete

"$TRAIN_PYTHON" scripts/verify_amd_training.py
