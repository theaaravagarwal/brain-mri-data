# AMD RX 7900 XT training environment

This setup targets x86-64 **Ubuntu 24.04.4**, AMD Radeon RX 7900 XT, ROCm
7.2.1, Python 3.12, and AMD's production-supported PyTorch 2.9.1 wheel. It does
not install CUDA or any NVIDIA packages.

The version pins reflect AMD's supported Radeon matrix as of 2026-08-12. Check
AMD's current Radeon compatibility matrix before reproducing the environment at
a later date.

## 1. Clone the project

```bash
git clone https://github.com/theaaravagarwal/brain-mri-data.git
cd brain-mri-data
```

## 2. Install the AMD driver and ROCm on the host

Run these commands on the Ubuntu training computer, not on a development Mac:

```bash
sudo apt update
sudo apt install -y python3-setuptools python3-wheel
wget https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb
sudo apt install ./amdgpu-install_7.2.1.70201-1_all.deb
amdgpu-install -y --usecase=graphics,rocm
sudo usermod -a -G render,video "$LOGNAME"
sudo reboot
```

After reconnecting:

```bash
groups
rocminfo | grep -E 'Name:|Marketing Name:'
```

The RX 7900 XT should appear as architecture `gfx1100`. Stop if it does not.

## 3. Install uv

If `uv` is not already installed, use the installation method published at
https://docs.astral.sh/uv/getting-started/installation/ and then open a new
shell.

## 4. Create the isolated training environment

```bash
./scripts/install_amd_training_env.sh
```

This creates `.venv-train`, installs the project/QC dependencies, installs only
AMD's ROCm PyTorch and Triton wheels, installs MONAI, and runs an actual matrix
multiplication on the GPU.

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
