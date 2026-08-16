#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
revision="$(git rev-parse --short=12 HEAD)"
output_dir="runs/language/language-adversarial-v1--${revision}"
mkdir -p runs/language
mkdir "$output_dir"

sha256sum config/language-adversarial-v1.yaml \
  benchmarks/language/planner-adversarial-v1.jsonl \
  config/language.yaml > "$output_dir/inputs.sha256"

.venv/bin/python scripts/run_language_adversarial.py \
  --fixtures benchmarks/language/planner-adversarial-v1.jsonl \
  --output "$output_dir/results.json" \
  --model qwen3-coder:30b | tee "$output_dir/summary.json"

ollama ps > "$output_dir/ollama-ps.txt"
awk '$1 == "qwen3-coder:30b" && $0 ~ /100% GPU/ { found=1 } END { exit !found }' \
  "$output_dir/ollama-ps.txt"

echo "Adversarial evaluation complete: $output_dir"
