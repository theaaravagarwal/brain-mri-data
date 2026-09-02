# NVIDIA CUDA environment

This setup supports the approved RTX 3060 or RTX 4060 CUDA workers. The current
external-inference worker is an RTX 4060 with 8 GB VRAM; do not reuse a 12 GB
training profile there without a fresh memory validation.
It is intentionally a distinct environment from the AMD ROCm worker: do not
install both training extras in the same virtual environment.

## Install and verify

Install a current NVIDIA driver on the host, confirm the GPU is visible to the
Linux or WSL2 guest, then run:

```bash
nvidia-smi
./scripts/install_cuda_training_env.sh
```

The installer creates the normal project `.venv` with Python 3.12 and runs:

```bash
uv sync --extra cuda --python 3.12
```

It uses the official PyTorch 2.9.1 CUDA 12.8 Linux wheel and MONAI 1.6.0, then
verifies that PyTorch sees an approved CUDA worker and completes a GPU matrix
product.

## Runtime profile

Use `training/profiles/cuda.yaml`. Its batch size is one, its
patch size is 80^3, and gradient accumulation reaches the study's effective
batch size of four. These values are deliberately shared with the ROCm profile
so hardware scheduling does not change the scientific configuration.
