#!/usr/bin/env bash
# Claim and run exactly one pre-registered CUDA glioma study job.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: ./scripts/run_glioma_job.sh ARM SEED" >&2
  exit 2
fi
ARM="$1"
SEED="$2"
PROFILE="cuda"
MATRIX="config/run-matrix/glioma.yaml"
STUDY="data/manifests/glioma.locked.json"
RUN_ID="glioma--${PROFILE}--${ARM}--${SEED}"
RUN="runs/${RUN_ID}"

if [[ ! -x .venv/bin/brain-mri-data || ! -x .venv/bin/python ]]; then
  echo "Run: uv sync --extra cuda" >&2
  exit 1
fi
if [[ ! -f "$STUDY" ]]; then
  echo "Missing frozen study: $STUDY" >&2
  exit 1
fi
if [[ -e "$RUN" ]]; then
  echo "Refusing to reuse existing run directory: $RUN" >&2
  exit 1
fi

.venv/bin/brain-mri-data runs claim "$MATRIX" "$RUN_ID" --profile "$PROFILE"
exec .venv/bin/python training/train_glioma.py \
  --study "$STUDY" --data-root data --profile training/profiles/cuda.yaml \
  --arm "$ARM" --seed "$SEED" --output "$RUN"
