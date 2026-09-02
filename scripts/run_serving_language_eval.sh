#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
revision="${BRAIN_MRI_SOURCE_REVISION:-$(git rev-parse --short=12 HEAD)}"
output_dir="runs/language/serving-metadata-v1--${revision}--$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p runs/language
mkdir "$output_dir"
sha256sum config/language-serving-eval.yaml benchmarks/language/serving-metadata-v1.jsonl \
  > "$output_dir/inputs.sha256"

failures=0
for model in qwen3:0.6b qwen3:4b; do
  slug="${model//[:.]/-}"
  if ! .venv/bin/python scripts/run_serving_language_eval.py \
    --fixtures benchmarks/language/serving-metadata-v1.jsonl \
    --output "$output_dir/${slug}.json" --model "$model" --max-wall-seconds 5; then
    failures=$((failures + 1))
  fi
  ollama ps > "$output_dir/${slug}.ollama-ps.txt"
  if ! awk -v model="$model" '$1 == model && $0 ~ /100% GPU/ { found=1 } END { exit !found }' \
    "$output_dir/${slug}.ollama-ps.txt"; then
    failures=$((failures + 1))
  fi
done

echo "Serving language evaluation complete: $output_dir"
exit "$failures"
