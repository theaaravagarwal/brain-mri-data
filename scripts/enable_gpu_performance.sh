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
powerd_unit="$(find /usr/share/doc -maxdepth 2 -path '*/nvidia-kernel-common-*/nvidia-powerd.service' -print | sort -V | tail -1)"
if [[ ! -x /usr/bin/nvidia-powerd || ! -f "$powerd_unit" ]]; then
  echo "NVIDIA Dynamic Boost components are unavailable" >&2
  exit 66
fi
sudo -v
sudo install -m 0644 "$unit" /etc/systemd/system/brain-mri-gpu-performance.service
sudo install -m 0644 "$powerd_unit" /etc/systemd/system/nvidia-powerd.service
sudo systemctl daemon-reload
sudo systemctl enable --now brain-mri-gpu-performance.service nvidia-powerd.service
printf 'Platform profile: %s\n' "$(powerprofilesctl get)"
printf 'Battery charge mode: %s\n' "$(cat /sys/class/power_supply/BAT0/charge_types)"
printf 'Dynamic Boost: %s\n' "$(systemctl is-active nvidia-powerd.service)"
nvidia-smi -q -d POWER | grep -E "Average Power Draw|Current Power Limit|Max Power Limit" | head -3
