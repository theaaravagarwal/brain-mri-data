from __future__ import annotations

import sys


def fail(message: str) -> None:
    raise SystemExit(f"AMD training environment check failed: {message}")


try:
    import monai
    import torch
except ImportError as error:
    fail(str(error))

if torch.version.hip is None:
    fail("PyTorch is not a ROCm/HIP build")
if torch.version.cuda is not None:
    fail(f"unexpected CUDA runtime detected: {torch.version.cuda}")
if not torch.cuda.is_available():
    fail("ROCm GPU is unavailable to PyTorch")

device_name = torch.cuda.get_device_name(0)
if "AMD" not in device_name.upper() and "RADEON" not in device_name.upper():
    fail(f"device 0 is not identified as AMD Radeon: {device_name}")

device = torch.device("cuda:0")
left = torch.randn((1024, 1024), device=device)
right = torch.randn((1024, 1024), device=device)
result = left @ right
torch.cuda.synchronize()

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"ROCm/HIP: {torch.version.hip}")
print(f"MONAI: {monai.__version__}")
print(f"GPU: {device_name}")
print(f"GPU tensor check: {tuple(result.shape)}")
