#!/usr/bin/env bash
set -euo pipefail
umask 077

expected_command="cd /home/b/brain-mri-data && .venv/bin/brain-mri-data language ingest --inbox runs/language-inbox"
if [[ -t 0 || "${SSH_ORIGINAL_COMMAND:-}" != "$expected_command" ]]; then
  echo '{"status":"rejected","reason_code":"invalid_transport_command"}'
  exit 2
fi
unset PYTHONHOME PYTHONPATH LD_PRELOAD LD_LIBRARY_PATH

cd /home/b/brain-mri-data
exec .venv/bin/brain-mri-data language ingest --inbox runs/language-inbox
