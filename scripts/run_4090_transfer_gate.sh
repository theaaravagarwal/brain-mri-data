#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
while ! .venv/bin/python scripts/verify_4090_transfer.py; do
  sleep 60
done
