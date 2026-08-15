#!/usr/bin/env bash
set -euo pipefail

AMD_HOST="${AMD_HOST:-b@100.64.0.5}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/b/brain-mri-data}"

mkdir -p data/raw/brats2020_kaggle data/manifests
rsync -a --partial --append-verify --info=progress2 \
  "${AMD_HOST}:${REMOTE_ROOT}/data/raw/brats2020_kaggle/" data/raw/brats2020_kaggle/
rsync -a "${AMD_HOST}:${REMOTE_ROOT}/data/manifests/brats2020_kaggle.cases.jsonl" data/manifests/
rsync -a "${AMD_HOST}:${REMOTE_ROOT}/data/manifests/brats2020_kaggle.qc.jsonl" data/manifests/

uv sync --extra cuda
exec ./scripts/train_cuda_pilot.sh 1 20260813
