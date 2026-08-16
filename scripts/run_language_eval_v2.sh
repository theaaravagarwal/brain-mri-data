#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

revision="$(git rev-parse --short=12 HEAD)"
output_dir="runs/language/language-v2--${revision}"
mkdir -p runs/language
mkdir "$output_dir"

sha256sum \
  config/language-eval-v2.yaml \
  benchmarks/language/structured-v2.jsonl \
  benchmarks/language/evidence-v2.jsonl \
  benchmarks/language/evidence-sources-v2.json \
  benchmarks/language/planner.jsonl \
  config/language.yaml > "$output_dir/inputs.sha256"

run_and_verify_gpu() {
  local model="$1"
  local kind="$2"
  shift 2
  ./.venv/bin/python scripts/run_language_benchmark.py "$kind" --model "$model" "$@" \
    | tee "$output_dir/${kind}.summary.json"
  ollama ps > "$output_dir/${kind}.ollama-ps.txt"
  awk -v model="$model" '$1 == model && $0 ~ /100% GPU/ { found=1 } END { exit !found }' \
    "$output_dir/${kind}.ollama-ps.txt"
}

run_and_verify_gpu qwen3:14b structured \
  --fixtures benchmarks/language/structured-v2.jsonl \
  --output "$output_dir/structured.responses.jsonl"

run_and_verify_gpu qwen3:14b evidence \
  --fixtures benchmarks/language/evidence-v2.jsonl \
  --evidence benchmarks/language/evidence-sources-v2.json \
  --output "$output_dir/evidence.responses.jsonl"

run_and_verify_gpu qwen3-coder:30b planner \
  --fixtures benchmarks/language/planner.jsonl \
  --output "$output_dir/planner.responses.jsonl"

echo "Language evaluation complete: $output_dir"
