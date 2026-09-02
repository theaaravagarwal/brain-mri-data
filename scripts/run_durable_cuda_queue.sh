#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 QUEUE_ID STUDY PROFILE EPOCHS ARM:SEED..." >&2
  exit 64
fi

queue_id="$1"
study="$2"
profile="$3"
epochs="$4"
shift 4
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! "$queue_id" =~ ^[a-z0-9-]+$ || ! "$epochs" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid queue id or epoch count" >&2
  exit 64
fi
cd "$repo_root"
if [[ ! -f "$study" || ! -f "$profile" || ! -x .venv/bin/python ]]; then
  echo "queue prerequisites are unavailable" >&2
  exit 66
fi

jobs=("$@")
runs=()
for job in "${jobs[@]}"; do
  arm="${job%%:*}"
  seed="${job##*:}"
  if [[ ! "$arm" =~ ^(brats|pooled|pamc)$ || ! "$seed" =~ ^[0-9]{8}$ ]]; then
    echo "invalid frozen job: ${job}" >&2
    exit 64
  fi
  runs+=("runs/${queue_id}--${arm}--${seed}--e${epochs}")
done

mkdir -p runs/queue-logs
exec 9>"runs/queue-logs/${queue_id}.lock"
if ! flock -n 9; then
  echo "$(date -Is) queue ${queue_id} is already active"
  exit 0
fi
exec > >(tee -a "runs/queue-logs/${queue_id}.log") 2>&1
status_path="runs/queue-logs/${queue_id}.status.json"

write_status() {
  local state="$1" current="${2:-}" last_error="${3:-}" completed=0 failed=0 separator=""
  local temporary="${status_path}.tmp"
  for run in "${runs[@]}"; do
    if [[ -f "$run/external.json" ]]; then
      completed=$((completed + 1))
    elif [[ -e "$run" && "$run" != "$current" ]]; then
      failed=$((failed + 1))
    fi
  done
  {
    printf '{"schemaVersion":"research-training-queue/v1","queueId":"%s","state":"%s","currentRun":' "$queue_id" "$state"
    [[ -n "$current" ]] && printf '"%s"' "${current#runs/}" || printf 'null'
    printf ',"queuedRuns":['
    for run in "${runs[@]}"; do
      if [[ ! -e "$run" ]]; then
        printf '%s"%s"' "$separator" "${run#runs/}"
        separator=','
      fi
    done
    printf '],"completedCount":%d,"totalCount":%d,"failedCount":%d,"lastError":' "$completed" "${#runs[@]}" "$failed"
    [[ -n "$last_error" ]] && printf '"%s"' "$last_error" || printf 'null'
    printf ',"updatedAt":"%s"}\n' "$(date -Is)"
  } > "$temporary"
  mv "$temporary" "$status_path"
}

write_status waiting
if [[ -n "${QUEUE_WAIT_FOR_UNIT:-}" ]]; then
  if [[ ! "$QUEUE_WAIT_FOR_UNIT" =~ ^[a-zA-Z0-9@_.:-]+\.service$ ]]; then
    echo "invalid QUEUE_WAIT_FOR_UNIT" >&2
    exit 64
  fi
  while systemctl --user is-active --quiet "$QUEUE_WAIT_FOR_UNIT"; do
    echo "$(date -Is) waiting for ${QUEUE_WAIT_FOR_UNIT} to finish"
    sleep 30
  done
fi

failures=0
for index in "${!jobs[@]}"; do
  job="${jobs[$index]}"
  arm="${job%%:*}"
  seed="${job##*:}"
  run="${runs[$index]}"
  if [[ -f "$run/external.json" ]]; then
    echo "$(date -Is) already complete: ${run}"
    continue
  fi
  if [[ -e "$run" ]]; then
    echo "$(date -Is) preserving incomplete run without overwrite: ${run}"
    failures=$((failures + 1))
    continue
  fi
  echo "$(date -Is) starting ${run}"
  write_status running "$run"
  if ! .venv/bin/python training/train_glioma.py \
      --study "$study" --data-root data --profile "$profile" \
      --arm "$arm" --seed "$seed" --epochs "$epochs" --output "$run"; then
    echo "$(date -Is) failed and preserved: ${run}"
    failures=$((failures + 1))
    write_status attention "" "preserved_failure"
  else
    write_status waiting
  fi
done

echo "$(date -Is) queue ${queue_id} finished with ${failures} preserved failure(s)"
if (( failures )); then
  write_status attention "" "preserved_failure"
else
  write_status complete
fi
exit 0
