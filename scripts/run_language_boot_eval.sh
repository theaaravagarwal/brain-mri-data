#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
for _ in {1..60}; do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null; then
    for model in qwen3:0.6b qwen3:4b; do
      printf '{"model":"%s","prompt":"Return only: ready","stream":false,"think":false,"keep_alive":"10m"}' "$model" |
        curl -fsS --max-time 120 -H 'Content-Type: application/json' --data-binary @- http://127.0.0.1:11434/api/generate >/dev/null
    done
    status=0
    scripts/run_serving_language_eval.sh || status=$?
    for model in qwen3:0.6b qwen3:4b; do
      printf '{"model":"%s","keep_alive":0}' "$model" |
        curl -fsS --max-time 30 -H 'Content-Type: application/json' --data-binary @- http://127.0.0.1:11434/api/generate >/dev/null || true
    done
    exit "$status"
  fi
  sleep 1
done
echo "Ollama did not become ready within 60 seconds" >&2
exit 1
