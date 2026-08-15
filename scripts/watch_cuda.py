#!/usr/bin/env bash
# Backward-compatible shortcut for the interactive project CLI.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/brain-mri-data monitor "$@"
