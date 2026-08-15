#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-1}"
SEED="${2:-20260813}"
STUDY="data/manifests/glioma.pilot.json"
RUN="runs/glioma-pilot--cuda--brats--${SEED}"
RUN+="${RUN_SUFFIX:-}"

if [[ ! -x .venv/bin/brain-mri-data || ! -x .venv/bin/python ]]; then
  echo "Run: uv sync --extra cuda" >&2
  exit 1
fi

for manifest in data/manifests/brats2020_kaggle.cases.jsonl data/manifests/brats2020_kaggle.qc.jsonl; do
  if [[ ! -s "$manifest" ]]; then
    echo "Missing $manifest. Run: ./scripts/sync_cuda_pilot.sh" >&2
    exit 1
  fi
done

if [[ ! -f "$STUDY" ]]; then
  .venv/bin/brain-mri-data build-study config/studies/glioma-pilot.yaml --output "${STUDY#data/manifests/}"
fi

exec .venv/bin/python training/train_glioma.py \
  --study "$STUDY" --profile training/profiles/cuda.yaml --arm brats \
  --seed "$SEED" --epochs "$EPOCHS" --output "$RUN"
