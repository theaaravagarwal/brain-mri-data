# AMD RX 7900 XT training environment

This setup targets **WSL2** with x86-64 Ubuntu 24.04, AMD Radeon RX 7900 XT/XTX,
ROCm 7.2, Python 3.12, and AMD's production-supported PyTorch 2.9.1
wheel. It does not install CUDA or any NVIDIA packages.

The version pins reflect AMD's supported Radeon matrix as of 2026-08-12. Check
AMD's current Radeon compatibility matrix before reproducing the environment at
a later date.

## 1. Clone the project

```bash
git clone https://github.com/theaaravagarwal/brain-mri-data.git
cd brain-mri-data
```

## 2. Make Windows expose the GPU to WSL2

The Windows host must use AMD's matching **Adrenalin Edition 26.1.1 for WSL2**
driver for the ROCm 7.2 stack. Install that Windows driver and reboot Windows.

Then run in an elevated PowerShell terminal:

```powershell
wsl --update
wsl --shutdown
```

Restart the Ubuntu distribution. Confirm WSL exposes the GPU interface:

```bash
grep -i microsoft /proc/sys/kernel/osrelease
ls -l /dev/dxg
```

If `/dev/dxg` is missing, stop. ROCm packages inside Ubuntu cannot create GPU
passthrough; the Windows driver/WSL layer is not ready.

Useful diagnostics from elevated PowerShell:

```powershell
wsl --version
wsl --list --verbose
Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion
```

The Ubuntu distribution must show WSL version `2`. If Windows sees the Radeon
but `/dev/dxg` remains absent after installing the matching AMD WSL driver,
reinstall/update the WSL distribution rather than installing native Linux DKMS
drivers inside it.

## 3. Install the WSL-specific ROCm user space

If `rocminfo` prints `WSL environment detected` followed by `hsa_init Failed`,
check the installed runtime:

```bash
dpkg-query -W hsa-runtime-rocr4wsl-amdgpu hsa-rocr 2>&1
```

The WSL stack requires `hsa-runtime-rocr4wsl-amdgpu`. If only `hsa-rocr` is
installed, the native-Linux ROCm repositories were used accidentally. Remove
that ROCm installation before installing the WSL stack:

```bash
sudo amdgpu-uninstall
sudo apt purge -y amdgpu-install
sudo apt autoremove -y
```

These commands remove AMD/ROCm packages from the Ubuntu distribution; they do
not remove the Windows Radeon driver or personal files.

Run inside the Ubuntu WSL distribution:

```bash
sudo apt update
sudo apt install -y python3-setuptools python3-wheel
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
sudo apt install -y ./amdgpu-install_7.2.70200-1_all.deb
sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms
```

Do not install DKMS or the native Linux graphics driver inside WSL. Close WSL
from PowerShell with `wsl --shutdown`, reopen Ubuntu, then verify:

```bash
rocminfo | grep -E 'Name:|Marketing Name:'
```

The RX 7900 XT/XTX should appear as architecture `gfx1100`. Stop if it does not.
Also confirm the WSL runtime is installed:

```bash
dpkg-query -W hsa-runtime-rocr4wsl-amdgpu
```

## 4. Install uv

If `uv` is not already installed, use the installation method published at
https://docs.astral.sh/uv/getting-started/installation/ and then open a new
shell.

## 5. Create the isolated training environment

```bash
./scripts/install_amd_training_env.sh
```

This creates `.venv-train`, installs the project/QC dependencies, installs only
AMD's WSL-compatible ROCm 7.2 PyTorch and Triton wheels, removes the Linux
wheel's bundled HSA runtime as AMD requires for WSL, installs MONAI, and runs an
actual matrix multiplication on the GPU.

Use the environment explicitly:

```bash
source .venv-train/bin/activate
python scripts/verify_amd_training.py
brain-mri-data --help
```

PyTorch intentionally calls the ROCm device through APIs such as
`torch.cuda.is_available()` and the device string `cuda:0`. Those historical
API names are also used by ROCm builds; the verifier requires `torch.version.hip`
and rejects a build with a CUDA runtime.
