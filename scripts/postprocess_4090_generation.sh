#!/usr/bin/env bash
# Wait for the final full-data RTX 4090 seeds, then write internal-only evidence.
set -euo pipefail

cd "$(dirname "$0")/.."

study="data/manifests/glioma.pilot.4060.json"
report="runs/prototype-cnn-rtx4090-dual-a--three-seed-summary.json"
failures="analyses/pilots/glioma-4090-dual-a/failure-analysis.json"
runs=(
  "runs/prototype-cnn-rtx4090-dual-a--brats--20260907--e100"
  "runs/prototype-cnn-rtx4090-dual-a--brats--20260908--e100"
  "runs/prototype-cnn-rtx4090-dual-a--brats--20260909--e100"
)

mkdir -p runs/queue-logs
exec > >(tee -a runs/queue-logs/4090-postprocess.log) 2>&1

complete() {
  [[ -f "$1/best.pt" && -f "$1/last.pt" && -f "$1/external.json" ]]
}

while true; do
  ready=true
  for run in "${runs[@]}"; do
    complete "$run" || ready=false
  done
  "$ready" && break
  if ! systemctl --user is-active --quiet brain-mri-cnn-4090-queue.service; then
    echo "$(date -Is) queue ended before all three analysis runs completed" >&2
    exit 1
  fi
  sleep 60
done

.venv/bin/python scripts/summarize_first_generation.py \
  --study "$study" --output "$report" "${runs[@]}"
.venv/bin/python scripts/analyze_first_generation_failures.py \
  --study "$study" --data-root data --output "$failures" "${runs[@]}"
echo "$(date -Is) postprocess complete: $report and $failures"
