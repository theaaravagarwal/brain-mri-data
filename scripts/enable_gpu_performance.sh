#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unit="$script_dir/systemd/brain-mri-gpu-performance.service"
dbus_policy="$script_dir/systemd/nvidia-dbus.conf"
if [[ ! -f "$unit" ]]; then
  unit="$HOME/Documents/.aa/brain/scripts/systemd/brain-mri-gpu-performance.service"
  dbus_policy="$HOME/Documents/.aa/brain/scripts/systemd/nvidia-dbus.conf"
fi
if [[ ! -f "$unit" || ! -f "$dbus_policy" ]]; then
  echo "GPU performance service files are unavailable" >&2
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
sudo install -m 0644 "$dbus_policy" /etc/dbus-1/system.d/nvidia-dbus.conf
sudo systemctl daemon-reload
sudo systemctl reload dbus.service
sudo systemctl enable brain-mri-gpu-performance.service
sudo systemctl restart brain-mri-gpu-performance.service
sudo systemctl enable nvidia-powerd.service
sudo systemctl restart nvidia-powerd.service
sleep 1
if ! busctl --system status nvidia.powerd.server >/dev/null 2>&1; then
  sudo journalctl -u nvidia-powerd.service -n 10 --no-pager >&2
  echo "NVIDIA Dynamic Boost failed to acquire its D-Bus service name" >&2
  exit 1
fi
printf 'Platform profile: %s\n' "$(powerprofilesctl get)"
printf 'Battery charge mode: %s\n' "$(cat /sys/class/power_supply/BAT0/charge_types)"
printf 'Dynamic Boost: active (D-Bus ownership verified)\n'
nvidia-smi -q -d POWER | grep -E "Average Power Draw|Current Power Limit|Max Power Limit" | head -3
