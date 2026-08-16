#!/usr/bin/env bash
set -euo pipefail

echo "AMD CNN pilots are disabled for this study. Use ./scripts/train_cuda_pilot.sh on the RTX 3060." >&2
echo "The AMD worker is reserved for the bounded language-model benchmark layer." >&2
exit 64
