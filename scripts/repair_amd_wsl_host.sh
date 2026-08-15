#!/usr/bin/env bash
set -euo pipefail

if ! grep -qi microsoft /proc/sys/kernel/osrelease || [[ ! -e /dev/dxg ]]; then
  echo "This repair script requires a WSL2 guest with /dev/dxg exposed by Windows." >&2
  exit 1
fi

source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
  echo "This repair script is pinned to Ubuntu 24.04 WSL2." >&2
  exit 1
fi

sudo -v
if dpkg-query -W -f='${Status}' hsa-rocr 2>/dev/null | grep -q '^install ok installed$'; then
  echo "Removing the incompatible native ROCm stack before installing the WSL runtime."
  sudo amdgpu-uninstall --rocmrelease=all
  sudo apt purge -y amdgpu-install
  sudo apt autoremove -y
fi
REPAIR_TMP="$(mktemp -d)"
trap 'rm -rf "$REPAIR_TMP"' EXIT
INSTALLER_DEB="$REPAIR_TMP/amdgpu-install_7.2.70200-1_all.deb"

curl --fail --location --retry 3 \
  --output "$INSTALLER_DEB" \
  "https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb"
sudo apt update
sudo apt install -y --allow-downgrades "$INSTALLER_DEB"
sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms
rocminfo | grep -E "Name:|Marketing Name:" | head -n 12
