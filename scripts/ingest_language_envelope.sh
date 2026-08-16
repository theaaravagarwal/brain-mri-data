#!/usr/bin/env bash
set -euo pipefail

cd /home/b/brain-mri-data
exec .venv/bin/brain-mri-data language ingest --inbox runs/language-inbox
