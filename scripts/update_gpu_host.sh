#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID} -eq 0 ]]; then
  echo "Run this as your normal user; it will request sudo once." >&2
  exit 64
fi

state_dir="${HOME}/.local/state/drupd"
mkdir -p "$state_dir"
chmod 700 "$state_dir"
rm -f "${state_dir}/postboot.json"
exec 9>"${state_dir}/update.lock"
flock -n 9 || { echo "Another driver update is already running." >&2; exit 75; }

log="${state_dir}/update-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$log") 2>&1
trap 'code=$?; printf "{\"status\":\"failed\",\"exitCode\":%d,\"updatedAt\":\"%s\"}\n" "$code" "$(date -u +%FT%TZ)" > "${state_dir}/preboot.json"; exit "$code"' ERR

. /etc/os-release
[[ ${ID:-} == ubuntu && ${VERSION_ID:-} == 24.04 ]] || {
  echo "This updater is pinned to Ubuntu 24.04." >&2
  exit 65
}
if command -v on_ac_power >/dev/null && ! on_ac_power; then
  echo "Connect the laptop to AC power before updating." >&2
  exit 69
fi

echo "Authenticating once for the complete maintenance run..."
sudo -v
(while sleep 45; do sudo -n true || exit; done) &
sudo_keeper=$!
trap 'kill "$sudo_keeper" 2>/dev/null || true' EXIT

target_user="$(id -un)"
target_home="$HOME"
current_kernel="$(uname -r)"
printf '{"status":"updating","kernel":"%s","updatedAt":"%s"}\n' \
  "$current_kernel" "$(date -u +%FT%TZ)" > "${state_dir}/preboot.json"

echo "Repairing package state and applying all Ubuntu updates..."
sudo dpkg --configure -a
sudo apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get -y \
  -o Dpkg::Options::=--force-confold --fix-broken install
sudo env DEBIAN_FRONTEND=noninteractive apt-get -y \
  -o Dpkg::Options::=--force-confold full-upgrade
sudo env DEBIAN_FRONTEND=noninteractive apt-get -y install \
  --install-recommends linux-generic-hwe-24.04 linux-headers-generic-hwe-24.04 ubuntu-drivers-common
if command -v snap >/dev/null; then sudo snap refresh; fi

echo "Installing Ubuntu's recommended NVIDIA driver and signed kernel modules..."
ubuntu-drivers devices
sudo ubuntu-drivers install
sudo env DEBIAN_FRONTEND=noninteractive apt-get -y --fix-broken install
sudo depmod -a
sudo update-initramfs -u -k all
sudo update-grub
sudo apt-get check
audit="$(sudo dpkg --audit)"
[[ -z "$audit" ]] || { printf '%s\n' "$audit" >&2; exit 70; }

echo "Refreshing firmware metadata (firmware flashing remains manual)..."
sudo fwupdmgr refresh --force || true
fwupdmgr get-updates || true

postboot="$(mktemp)"
unit="$(mktemp)"
trap 'rm -f "$postboot" "$unit"; kill "$sudo_keeper" 2>/dev/null || true' EXIT
cat > "$postboot" <<'POSTBOOT'
#!/usr/bin/env bash
set -uo pipefail
state_dir="${DRUPD_TARGET_HOME}/.local/state/drupd"
install -d -m 0700 -o "$DRUPD_TARGET_USER" -g "$DRUPD_TARGET_USER" "$state_dir"
if nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >/dev/null 2>&1; then
  status=ready
  gpu=true
else
  status=gpu_unavailable
  gpu=false
fi
temporary="${state_dir}/postboot.json.tmp"
printf '{"status":"%s","gpuAccess":%s,"kernel":"%s","bootId":"%s","checkedAt":"%s"}\n' \
  "$status" "$gpu" "$(uname -r)" "$(cat /proc/sys/kernel/random/boot_id)" "$(date -u +%FT%TZ)" > "$temporary"
chown "$DRUPD_TARGET_USER:$DRUPD_TARGET_USER" "$temporary"
chmod 0600 "$temporary"
mv "$temporary" "${state_dir}/postboot.json"
POSTBOOT
cat > "$unit" <<EOF
[Unit]
Description=Validate NVIDIA GPU after drupd maintenance reboot
After=multi-user.target

[Service]
Type=oneshot
Environment=DRUPD_TARGET_USER=${target_user}
Environment=DRUPD_TARGET_HOME=${target_home}
ExecStart=/usr/local/sbin/drupd-postboot

[Install]
WantedBy=multi-user.target
EOF
sudo install -m 0755 "$postboot" /usr/local/sbin/drupd-postboot
sudo install -m 0644 "$unit" /etc/systemd/system/drupd-postboot.service
sudo systemctl daemon-reload
sudo systemctl enable drupd-postboot.service

module_kernel="$(basename "$(readlink -f /boot/vmlinuz)")"
module_kernel="${module_kernel#vmlinuz-}"
modinfo -k "$module_kernel" nvidia >/dev/null
vermagic="$(modinfo -k "$module_kernel" -F vermagic nvidia)"
[[ $vermagic == "$module_kernel "* ]]
if command -v mokutil >/dev/null && mokutil --sb-state 2>/dev/null | grep -qi enabled; then
  [[ -n "$(modinfo -k "$module_kernel" -F signer nvidia)" ]] || {
    echo "The NVIDIA module is unsigned while Secure Boot is enabled." >&2
    exit 71
  }
fi
printf '{"status":"rebooting","kernelBefore":"%s","moduleKernel":"%s","updatedAt":"%s"}\n' \
  "$current_kernel" "$module_kernel" "$(date -u +%FT%TZ)" > "${state_dir}/preboot.json"
sync
echo "Updates complete. Rebooting in 10 seconds; post-boot GPU validation is installed."
sleep 10
sudo systemctl reboot
