# AMD RX 7900 XTX training environment

This setup targets **WSL2** with x86-64 Ubuntu 24.04, AMD Radeon RX 7900 XTX,
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
```

Reboot the complete Windows host so the configured startup task brings WSL,
systemd, and Tailscale back together. Never issue `wsl --shutdown` through SSH;
it can remove remote access without starting the distribution again. After the
host returns, confirm WSL exposes the GPU interface:

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

For a host with the wrong native-Linux AMD installer or a missing WSL HSA
runtime, run this helper from an interactive terminal on the AMD worker. It
prompts for your sudo password, removes only an incompatible native ROCm stack
when detected, and installs AMD's official WSL 7.2 use case. Do not run it
from a non-interactive SSH job:

```bash
./scripts/repair_amd_wsl_host.sh
```

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

Do not install DKMS or the native Linux graphics driver inside WSL. If the new
runtime genuinely requires a restart, notify the owner and reboot the complete
Windows host. Do not use remote `wsl --shutdown`. After the host returns, verify:

```bash
rocminfo | grep -E 'Name:|Marketing Name:'
```

The RX 7900 XT/XTX should appear as architecture `gfx1100`. Stop if it does not.
Also confirm the WSL runtime is installed:

```bash
dpkg-query -W hsa-runtime-rocr4wsl-amdgpu
```

Do **not** set `HSA_ENABLE_DXG_DETECTION=1` in `~/.bashrc`, the Ollama service,
or another persistent environment file. With this ROCm 7.2 WSL runtime it loads
an incompatible `librocdxg` ABI and makes both `rocminfo` and PyTorch crash. A
clean shell with that variable unset must enumerate `gfx1100`.

## 4. Install uv

If `uv` is not already installed, use the installation method published at
https://docs.astral.sh/uv/getting-started/installation/ and then open a new
shell.

## 5. Create the isolated training environment

```bash
./scripts/install_amd_training_env.sh
```

This runs `uv sync --extra amd` into `.venv`, installs AMD's WSL-compatible
ROCm 7.2 PyTorch and Triton wheels, replaces the Linux wheel's bundled HSA
runtime with `/opt/rocm/lib/libhsa-runtime64.so.1.2` as AMD requires for WSL,
installs MONAI, and runs an actual matrix multiplication on the GPU. Do not
install the `cuda` extra in this environment.

Use the environment explicitly:

```bash
source .venv/bin/activate
python scripts/verify_amd_training.py
brain-mri-data --help
```

PyTorch intentionally calls the ROCm device through APIs such as
`torch.cuda.is_available()` and the device string `cuda:0`. Those historical
API names are also used by ROCm builds; the verifier requires `torch.version.hip`
and rejects a build with a CUDA runtime.

## Monitoring a WSL training run

WSL2 intentionally does not load the native Linux `amdgpu` kernel driver, so
`rocm-smi` cannot report utilization there. Windows owns that driver and exposes
the GPU to WSL through `/dev/dxg`. Use the project monitor instead:

```bash
./scripts/watch_amd.py
```

It combines Linux trainer CPU/RAM with Windows GPU adapter memory and compute
engine counters. Press `Ctrl-C` to stop monitoring; it does not stop training.

## Storage accounting

`df /` reports free blocks inside the virtual ext4 filesystem, not free space
on the Windows drive that stores the VHDX. Query Windows directly when deciding
whether another model or dataset fits:

```bash
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe \
  -NoProfile -NonInteractive -Command \
  'Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | Select-Object DeviceID,Size,FreeSpace'
```

On 2026-08-16, Windows `C:` had about 269.5 GiB free and the WSL VHDX occupied
about 274.4 GiB. Cleanup removed the AMD copies of six CNN raw datasets, old
AMD CNN run directories, and 28.2 GiB of regenerable uv cache. WSL usage fell
from about 274 GiB to 104 GiB, but the VHDX did not shrink online. Do not stop
WSL or attempt remote VHDX compaction. Windows Downloads (about 206 GiB) and
the 17 GiB root-owned APT archive cache were measured but intentionally left
untouched.

## Restarting WSL from Windows

Never run `wsl --shutdown` through the AMD worker's SSH connection: it closes
the connection that is executing the command and may leave the distribution
stopped. If a restart is genuinely required, notify the owner and reboot the
complete Windows host from Windows so the existing startup task restores WSL.
From an authorized Windows PowerShell session, the full-host command is:

```powershell
shutdown.exe /r /t 0
```

Do not invoke that command unless a restart is necessary and the owner has been
notified. Normal Ollama repair below does not require a restart.

## Optional: Ollama research-language worker

Ollama is optional and is only for the constrained research-language benchmarks
in this project. It never receives images, paths, patient identifiers, or the
power to route a model, diagnose, or recommend treatment.

The RX 7900 XTX is an officially supported Ollama ROCm target. On this WSL2
host, Ollama's bundled native-Linux HSA runtime enumerates only CPU even though
PyTorch sees `gfx1100`. The helper installs a user-scoped service that preloads
the generic symlink to the installed WSL HSA runtime, keeps
`HSA_ENABLE_DXG_DETECTION` unset, enables user lingering, and fails closed if
the new service log does not report `library=ROCm` and `compute=gfx1100`:

```bash
./scripts/repair_amd_ollama_wsl.sh
ollama run qwen3:14b "Reply with exactly: OK"
ollama ps
```

`ollama ps` must say `100% GPU` in the `PROCESSOR` column. If it says `CPU`,
stop the model immediately (`ollama stop qwen3:14b`) rather than consuming the
thermally constrained CPU, then capture the user-service log:

```bash
systemctl --user status ollama.service
tail -n 120 runs/logs/ollama-amd.log
```

The helper intentionally uses an 8k context and five-minute keep-alive for the
first verification. Increase either only after a GPU-backed benchmark has
passed and its resource use is recorded.
