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

if ! dpkg-query -W -f='${Status}' hsa-runtime-rocr4wsl-amdgpu 2>/dev/null \
  | grep -q '^install ok installed$'; then
  echo "Missing AMD's WSL-specific HSA runtime: hsa-runtime-rocr4wsl-amdgpu." >&2
  echo "A native-Linux ROCm install will detect WSL but fail hsa_init." >&2
  echo "Repair the ROCm host packages using docs/amd-rocm-setup.md before continuing." >&2
  exit 1
fi

if ! rocminfo 2>/dev/null | grep -q "gfx1100"; then
  echo "ROCm does not expose the RX 7900 XT/XTX (expected gfx1100)." >&2
  exit 1
fi

uv python install 3.12
uv sync --extra amd --python 3.12

TRAIN_PYTHON=".venv/bin/python"

# AMD's WSL instructions require the WSL-compatible HSA runtime instead of the
# native Linux runtime bundled inside the PyTorch wheel.
TORCH_LIB_DIR="$($TRAIN_PYTHON -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("torch").origin).parent / "lib")')"
WSL_HSA_RUNTIME="$(readlink -f /opt/rocm/lib/libhsa-runtime64.so 2>/dev/null || true)"
if [[ -z "$WSL_HSA_RUNTIME" || ! -f "$WSL_HSA_RUNTIME" ]]; then
  echo "Missing WSL HSA runtime under /opt/rocm/lib/." >&2
  echo "Install the WSL ROCm use case before creating the Python environment." >&2
  exit 1
fi
find "$TORCH_LIB_DIR" -maxdepth 1 -type f -name 'libhsa-runtime64.so*' -delete
cp "$WSL_HSA_RUNTIME" "$TORCH_LIB_DIR/libhsa-runtime64.so"

"$TRAIN_PYTHON" scripts/verify_amd_training.py
