#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unit="$script_dir/systemd/brain-mri-gpu-performance.service"
if [[ ! -f "$unit" ]]; then
  unit="$HOME/Documents/.aa/brain/scripts/systemd/brain-mri-gpu-performance.service"
fi
if [[ ! -f "$unit" ]]; then
  echo "brain-mri-gpu-performance.service is unavailable" >&2
  exit 66
fi
sudo -v
sudo install -m 0644 "$unit" /etc/systemd/system/brain-mri-gpu-performance.service
sudo systemctl daemon-reload
sudo systemctl enable --now brain-mri-gpu-performance.service
printf 'Platform profile: %s\n' "$(powerprofilesctl get)"
nvidia-smi -q -d POWER | grep -E "Average Power Draw|Current Power Limit|Max Power Limit" | head -3
