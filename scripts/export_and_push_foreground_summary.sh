#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source_summary="analyses/pilots/glioma-v4-foreground-screen/results.json"
outbox="runs/language-outbox"
identity="${LANGUAGE_TRANSFER_IDENTITY:-runs/language-transport/brain_mri_language_ed25519}"

if [[ ! -f "$source_summary" ]]; then
  echo "Foreground-screen summary is not complete: $source_summary" >&2
  exit 2
fi
if [[ ! -f "$identity" ]]; then
  echo "Restricted language-transfer identity is not configured: $identity" >&2
  exit 2
fi

mkdir -p "$outbox"
pending=()
completed=()
while IFS= read -r envelope; do
  receipt="${envelope%.json}.transfer-receipt.json"
  if [[ -f "$receipt" ]]; then
    completed+=("$envelope")
  else
    pending+=("$envelope")
  fi
done < <(
  find "$outbox" -maxdepth 1 -type f -name '*.json' \
    ! -name '*.receipt.json' ! -name '*.transfer-receipt.json' -print | sort
)
if (( ${#completed[@]} > 0 && ${#pending[@]} > 0 )); then
  echo "Mixed completed and pending language exports require review" >&2
  exit 2
elif (( ${#completed[@]} > 0 )); then
  echo "Foreground summary already transferred: ${completed[0]}"
  exit 0
elif (( ${#pending[@]} > 1 )); then
  echo "Multiple pending language exports require review" >&2
  exit 2
elif (( ${#pending[@]} == 1 )); then
  export_path="${pending[0]}"
else
  export_json="$(.venv/bin/brain-mri-data language export-run-summary \
    "$source_summary" --runs-root runs --outbox "$outbox" \
    --run-group-id glioma-v4-foreground-screen)"
  export_path="$(.venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["path"])' <<<"$export_json")"
fi

.venv/bin/brain-mri-data language push "$export_path" --identity "$identity"
