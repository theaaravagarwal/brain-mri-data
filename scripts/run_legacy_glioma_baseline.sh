#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export HSA_ENABLE_DXG_DETECTION=1

exec .venv/bin/python training/train_brats_segmentation.py \
  --profile training/profiles/amd.yaml \
  --datalist data/manifests/brats2020_kaggle.monai.json \
  --external-datalist data/manifests/brats2023_ssa_hf.external.monai.json \
  --external-label-schema brats_africa_123 \
  --output runs/legacy-glioma-baseline-v2-seed20260812 \
  --epochs 30 \
  --batch-size 1 \
  --accumulation-steps 4 \
  --init-filters 40 \
  --num-workers auto \
  --max-workers 6 \
  --seed 20260812
