#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "This installer requires x86-64 Linux or a Linux WSL2 distribution." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable. Install a current NVIDIA driver before creating the CUDA environment." >&2
  exit 1
fi

if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "RTX 3060"; then
  echo "Expected an RTX 3060 worker; inspect the selected CUDA training profile before continuing." >&2
  exit 1
fi

uv python install 3.12
uv sync --extra cuda --python 3.12
.venv/bin/python scripts/verify_cuda_training.py
