#!/usr/bin/env bash
# Produce internal-only evidence after the finite RTX 3060 queue completes.
set -euo pipefail

cd "$(dirname "$0")/.."

study="data/manifests/glioma.pilot.full.v3.batch4.json"
report="runs/prototype-cnn-rtx3060-16h--three-seed-summary.json"
failures="analyses/pilots/glioma-3060-16h/failure-analysis.json"
runs=(
  "runs/prototype-cnn-rtx3060-16h--brats--20260904--e100"
  "runs/prototype-cnn-rtx3060-16h--brats--20260905--e100"
  "runs/prototype-cnn-rtx3060-16h--brats--20260906--e100"
)

while systemctl --user is-active --quiet brain-mri-cnn-3060-16h.service; do
  sleep 60
done
for run in "${runs[@]}"; do
  [[ -f "$run/best.pt" && -f "$run/last.pt" && -f "$run/external.json" ]] || {
    echo "Incomplete analysis run: $run" >&2
    exit 1
  }
done

.venv/bin/python scripts/summarize_first_generation.py \
  --study "$study" --output "$report" "${runs[@]}"
.venv/bin/python scripts/analyze_first_generation_failures.py \
  --study "$study" --data-root data --output "$failures" "${runs[@]}"
echo "$(date -Is) postprocess complete: $report and $failures"
