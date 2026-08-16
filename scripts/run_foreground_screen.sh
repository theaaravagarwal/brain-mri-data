#!/usr/bin/env bash
# Run the predeclared, single-seed foreground-patch screen and stop for review.
set -euo pipefail

cd "$(dirname "$0")/.."

PROFILE="training/profiles/cuda-batch4.yaml"
SOURCE_CACHE="data/cache/glioma-pilot-full-repaired--chunk20-v1/cache.json"
INDEXED_CACHE="data/cache/glioma-pilot-full-repaired--chunk20-v1/cache.foreground-chunks-v1.json"
FAILURE_ANALYSIS="analyses/pilots/glioma-v3/failure-analysis.json"
BASELINE="runs/glioma-pilot-full-v3-batch4--cuda--brats--20260812--e10"
OUTPUT_ROOT="analyses/pilots/glioma-v4-foreground-screen"
SEED=20260812
EPOCHS=10
VARIANTS=(fg25 fg50 fg75)

declare -A CONFIGS=(
  [fg25]="config/studies/glioma-pilot-v4-fg25.yaml"
  [fg50]="config/studies/glioma-pilot-v4-fg50.yaml"
  [fg75]="config/studies/glioma-pilot-v4-fg75.yaml"
)
declare -A STUDIES=(
  [fg25]="data/manifests/glioma.pilot.v4.fg25.json"
  [fg50]="data/manifests/glioma.pilot.v4.fg50.json"
  [fg75]="data/manifests/glioma.pilot.v4.fg75.json"
)

if [[ ! -x .venv/bin/brain-mri-data || ! -x .venv/bin/python ]]; then
  echo "Missing CUDA environment" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_CACHE" || ! -f "$FAILURE_ANALYSIS" || ! -d "$BASELINE" ]]; then
  echo "Foreground screen prerequisites are incomplete" >&2
  exit 1
fi
available_kb="$(df --output=avail -k . | tail -1 | tr -d ' ')"
if (( available_kb < 100 * 1024 * 1024 )); then
  echo "Foreground screen requires at least 100 GiB free" >&2
  exit 1
fi

for variant in "${VARIANTS[@]}"; do
  study="${STUDIES[$variant]}"
  if [[ ! -f "$study" ]]; then
    .venv/bin/brain-mri-data build-study "${CONFIGS[$variant]}" --output "${study#data/manifests/}"
  fi
done

if [[ ! -f "$INDEXED_CACHE" ]]; then
  .venv/bin/python scripts/build_foreground_cache_index.py \
    --study "${STUDIES[fg25]}" --data-root data --source-cache "$SOURCE_CACHE" \
    --output "$INDEXED_CACHE" --arm brats
fi

for variant in "${VARIANTS[@]}"; do
  run="runs/glioma-pilot-v4-${variant}--cuda--brats--${SEED}--e${EPOCHS}"
  if [[ -e "$run" ]]; then
    echo "Refusing to reuse screen run: $run" >&2
    exit 1
  fi
  echo "$(date -Is) starting foreground screen variant=$variant seed=$SEED"
  .venv/bin/python training/train_glioma.py \
    --study "${STUDIES[$variant]}" --data-root data --profile "$PROFILE" --arm brats \
    --seed "$SEED" --epochs "$EPOCHS" --output "$run" --training-cache "$INDEXED_CACHE"
done

.venv/bin/python scripts/summarize_foreground_screen.py \
  --failure-analysis "$FAILURE_ANALYSIS" --baseline "$BASELINE" \
  --output-json "$OUTPUT_ROOT/results.json" --output-markdown "$OUTPUT_ROOT/results.md" \
  "runs/glioma-pilot-v4-fg25--cuda--brats--${SEED}--e${EPOCHS}" \
  "runs/glioma-pilot-v4-fg50--cuda--brats--${SEED}--e${EPOCHS}" \
  "runs/glioma-pilot-v4-fg75--cuda--brats--${SEED}--e${EPOCHS}"

if [[ -f runs/language-transport/brain_mri_language_ed25519 ]]; then
  scripts/export_and_push_foreground_summary.sh
else
  echo "Restricted language-transfer key is absent; aggregate export remains pending" >&2
fi

echo "$(date -Is) foreground screen complete; stop for human review: $OUTPUT_ROOT/results.md"
