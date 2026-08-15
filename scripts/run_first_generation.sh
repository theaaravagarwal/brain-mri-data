#!/usr/bin/env bash
# Build and run the first full-data CUDA pilot without making an external-test claim.
#
# This queue is deliberately finite: it imports only missing official BraTS files,
# re-indexes and QC-gates them, locks a new full-data pilot manifest, trains three
# deterministic seeds, and writes a stability report.  It does not tune the model
# after observing results and it does not manufacture an external test set.
set -euo pipefail

cd "$(dirname "$0")/.."

STAGE_DIR="${1:?usage: $0 STAGED_BRATS_DIRECTORY [epochs]}"
EPOCHS="${2:-10}"
RAW_DIR="data/raw/brats2020_kaggle"
STUDY="data/manifests/glioma.pilot.full.json"
REPORT="runs/glioma-pilot-full--cuda--first-generation.json"
SEEDS=(20260812 20260813 20260814)

if [[ ! -x .venv/bin/brain-mri-data || ! -x .venv/bin/python ]]; then
  echo "Missing CUDA environment. Run: uv sync --extra cuda" >&2
  exit 1
fi
if [[ ! -d "$STAGE_DIR/BraTS2020_TrainingData" || ! -d "$STAGE_DIR/BraTS2020_ValidationData" ]]; then
  echo "Stage does not look like a completed official BraTS 2020 download: $STAGE_DIR" >&2
  exit 1
fi
if [[ "$EPOCHS" -lt 1 ]]; then
  echo "epochs must be positive" >&2
  exit 2
fi

mkdir -p runs

# A current pilot owns the single RTX 3060.  Waiting prevents two jobs competing
# for its 12 GB of VRAM and keeps this queue safe to start immediately.
while tmux has-session -t cuda-curves 2>/dev/null; do
  echo "$(date -Is) waiting for existing cuda-curves session"
  sleep 60
done

echo "$(date -Is) importing missing official BraTS files"
for collection in BraTS2020_TrainingData BraTS2020_ValidationData; do
  rsync -a --ignore-existing --info=progress2 "$STAGE_DIR/$collection/" "$RAW_DIR/$collection/"
done

echo "$(date -Is) indexing and validating the full local source"
.venv/bin/brain-mri-data index brats2020_kaggle
.venv/bin/brain-mri-data validate brats2020_kaggle
.venv/bin/brain-mri-data split brats2020_kaggle --seed 20260812
.venv/bin/brain-mri-data export-monai brats2020_kaggle

if [[ -e "$STUDY" ]]; then
  echo "Refusing to overwrite immutable full-data study: $STUDY" >&2
  exit 1
fi
.venv/bin/brain-mri-data build-study config/studies/glioma-pilot.yaml --output "${STUDY#data/manifests/}"

for seed in "${SEEDS[@]}"; do
  run="runs/glioma-pilot-full--cuda--brats--${seed}--e${EPOCHS}"
  if [[ -e "$run" ]]; then
    echo "Refusing to reuse a run directory: $run" >&2
    exit 1
  fi
  echo "$(date -Is) training full-data baseline seed=$seed epochs=$EPOCHS"
  .venv/bin/python training/train_glioma.py \
    --study "$STUDY" --data-root data --profile training/profiles/cuda.yaml --arm brats \
    --seed "$seed" --epochs "$EPOCHS" --output "$run"
done

.venv/bin/python scripts/summarize_first_generation.py \
  --study "$STUDY" --output "$REPORT" \
  "runs/glioma-pilot-full--cuda--brats--20260812--e${EPOCHS}" \
  "runs/glioma-pilot-full--cuda--brats--20260813--e${EPOCHS}" \
  "runs/glioma-pilot-full--cuda--brats--20260814--e${EPOCHS}"

echo "$(date -Is) first generation complete: $REPORT"
echo "External testing remains locked: acquire and provenance-review an independent source before testing."
