#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sudo -v
sudo powerprofilesctl set performance
sudo install -m 0644 "$repo_root/scripts/systemd/brain-mri-gpu-performance.service" /etc/systemd/system/brain-mri-gpu-performance.service
sudo systemctl daemon-reload
sudo systemctl enable --now brain-mri-gpu-performance.service
nvidia-smi -q -d POWER | grep -E "Average Power Draw|Current Power Limit|Max Power Limit" | head -3
