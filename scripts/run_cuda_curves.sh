#!/usr/bin/env bash
set -euo pipefail

for seed in 20260812 20260813 20260814; do
  RUN_SUFFIX=--e10 ./scripts/train_cuda_pilot.sh 10 "$seed"
done
