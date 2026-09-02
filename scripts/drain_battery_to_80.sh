#!/usr/bin/env bash
set -euo pipefail

target=80
battery=/sys/class/power_supply/BAT0
fan=/sys/bus/platform/drivers/ideapad_acpi/VPC2004:00/fan_mode

if (( EUID != 0 )); then
  exec sudo "$0" "$@"
fi
if [[ ! -r "$battery/capacity" || ! -w "$fan" ]]; then
  echo "Required Legion battery or fan control is unavailable" >&2
  exit 66
fi
if [[ "$(<"$battery/status")" != "Discharging" ]]; then
  echo "Unplug AC before running this helper" >&2
  exit 1
fi

restore_fans() {
  printf 0 > "$fan"
}
trap restore_fans EXIT INT TERM

while (( $(<"$battery/capacity") > target )); do
  printf 2 > "$fan"
  capacity="$(<"$battery/capacity")"
  watts="$(awk '{printf "%.1f", $1 / 1000000}' "$battery/power_now")"
  printf '%(%H:%M:%S)T battery=%s%% draw=%sW fan_mode=2\n' -1 "$capacity" "$watts"
  sleep 30
done

restore_fans
trap - EXIT INT TERM
printf '\aBattery reached %s%%. Fan control is automatic again; reconnect AC now.\n' "$(<"$battery/capacity")"
