#!/usr/bin/env bash
set -euo pipefail

fan=/sys/bus/platform/drivers/ideapad_acpi/VPC2004:00/fan_mode
sudo systemctl stop brain-mri-full-fans.service 2>/dev/null || true
printf 0 | sudo tee "$fan" >/dev/null
echo "Automatic fan control restored."
