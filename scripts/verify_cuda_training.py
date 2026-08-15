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
if "RTX 3060" not in device_name.upper():
    fail(f"device 0 is not an RTX 3060: {device_name}")

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
print(f"GPU tensor check: {tuple(result.shape)}")
