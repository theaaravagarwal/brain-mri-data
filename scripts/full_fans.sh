#!/usr/bin/env bash
set -euo pipefail

fan=/sys/bus/platform/drivers/ideapad_acpi/VPC2004:00/fan_mode
unit=brain-mri-full-fans.service

if [[ "${1:-}" == "--worker" ]]; then
  (( EUID == 0 )) || exit 77
  trap 'printf 0 > "$fan"' EXIT INT TERM
  while true; do
    printf 2 > "$fan"
    sleep 20
  done
fi

script="$(readlink -f "$0")"
sudo systemctl stop "$unit" 2>/dev/null || true
sudo systemd-run --unit="${unit%.service}" --collect \
  --property=Restart=on-failure --property=RestartSec=2 \
  "$script" --worker
sleep 1
sudo systemctl is-active --quiet "$unit"
echo "Forced high-fan mode is active. Run ~/auto-fans.sh to restore automatic control."
