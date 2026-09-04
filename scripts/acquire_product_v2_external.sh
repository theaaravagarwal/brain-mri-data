#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
source_id="brats2023_ssa_hf"

cd "$repo_root"

python="$repo_root/.venv/bin/python"
if [[ ! -x "$python" ]]; then
  echo "missing QC environment; run: uv sync --frozen --extra qc" >&2
  exit 69
fi
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_XET=1

echo "Acquiring the immutable 60-case BraTS 2023 SSA external cohort."
echo "This is a public mirror/subset, not the full 146-subject TCIA collection."

for attempt in 1 2 3 4 5; do
  if "$python" -m brain_mri_data.cli fetch "$source_id" --resume; then
    break
  fi
  if [[ "$attempt" == 5 ]]; then
    echo "external download failed after five resumable attempts" >&2
    exit 1
  fi
  sleep "$((attempt * 60))"
done
"$python" -m brain_mri_data.cli discover "$source_id"
"$python" -m brain_mri_data.cli index "$source_id"
"$python" -m brain_mri_data.cli verify-files "$source_id"
"$python" -m brain_mri_data.cli validate "$source_id"

echo "External source acquisition, indexing, fingerprint verification, and QC complete."
