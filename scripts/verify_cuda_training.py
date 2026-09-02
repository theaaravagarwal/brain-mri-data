from __future__ import annotations

import sys


def fail(message: str) -> None:
    raise SystemExit(f"CUDA training environment check failed: {message}")


try:
    import monai
    import torch
except ImportError as error:
    fail(str(error))

if torch.version.cuda is None or torch.version.hip is not None:
    fail("PyTorch is not a CUDA-only build")
if not torch.cuda.is_available():
    fail("CUDA GPU is unavailable to PyTorch")

device_name = torch.cuda.get_device_name(0)
supported_gpus = ("RTX 3060", "RTX 4060")
if not any(expected in device_name.upper() for expected in supported_gpus):
    fail(
        "device 0 is not an approved CUDA worker "
        f"({', '.join(supported_gpus)}): {device_name}"
    )

total_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
if total_memory_gib < 7.5:
    fail(f"device 0 has insufficient VRAM ({total_memory_gib:.2f} GiB; need at least 7.5 GiB)")

device = torch.device("cuda:0")
left = torch.randn((1024, 1024), device=device)
right = torch.randn((1024, 1024), device=device)
result = left @ right
torch.cuda.synchronize()

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"MONAI: {monai.__version__}")
print(f"GPU: {device_name}")
print(f"GPU memory: {total_memory_gib:.2f} GiB")
print(f"GPU tensor check: {tuple(result.shape)}")
