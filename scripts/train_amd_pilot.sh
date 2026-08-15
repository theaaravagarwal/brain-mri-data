#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-1}"
STUDY="data/manifests/glioma.pilot.json"
RUN="runs/glioma-pilot--amd--brats--20260812"

if [[ ! -x .venv/bin/brain-mri-data || ! -x .venv/bin/python ]]; then
  echo "Run: uv sync --extra amd" >&2
  exit 1
fi

for manifest in data/manifests/brats2020_kaggle.cases.jsonl data/manifests/brats2020_kaggle.qc.jsonl; do
  if [[ ! -s "$manifest" ]]; then
    echo "Missing $manifest. Run: brain-mri-data index brats2020_kaggle && brain-mri-data validate brats2020_kaggle" >&2
    exit 1
  fi
done

if [[ ! -f "$STUDY" ]]; then
  .venv/bin/brain-mri-data build-study config/studies/glioma-pilot.yaml --output "${STUDY#data/manifests/}"
fi

exec .venv/bin/python training/train_glioma.py \
  --study "$STUDY" --profile training/profiles/amd.yaml --arm brats \
  --seed 20260812 --epochs "$EPOCHS" --output "$RUN"
