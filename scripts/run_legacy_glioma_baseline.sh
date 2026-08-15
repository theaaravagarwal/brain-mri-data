#!/usr/bin/env bash
set -euo pipefail

echo "The legacy glioma baseline is archived and disabled; it is not part of the frozen ISEF study." >&2
echo "Use ./scripts/train_cuda_pilot.sh or ./scripts/run_glioma_job.sh instead." >&2
exit 64
