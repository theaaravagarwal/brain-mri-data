# Compute-host runbook

This file is the canonical operational reference for the project's two GPU
workers. Keep raw MRI data and machine-specific environments local to each
worker; synchronize only code and approved non-identifying artifacts.

## NVIDIA CNN worker

- SSH target: `theaa@10.0.0.65`
- Repository: `/home/theaa/Documents/brain-mri-data`
- GPU: NVIDIA RTX 3060, 12 GB VRAM
- Environment: `uv sync --extra cuda`
- Role: the sole CNN training worker, including frozen studies and bounded
  patch-sampling screens
- Current production loader profile: batch 4, eight workers, prefetch factor 2,
  persistent workers, and the indexed chunk cache
- Language-transfer role: construct aggregate-only JSON envelopes and push them
  one-way to AMD. Never transfer MRI data, case-level metrics, paths, or free text.

## AMD research-language worker

- SSH target: `b@100.64.0.5`
- Repository: `~/brain-mri-data` (`/home/b/brain-mri-data`)
- GPU: AMD Radeon RX 7900 XTX, 24 GB VRAM
- Environment: `uv sync --extra amd`; never combine it with the CUDA extra
- Role: primarily GPU-backed research-language experiments, not CNN study arms
- Language inbox: `runs/language-inbox`; it accepts only the strict aggregate
  envelope through a forced, restricted SSH command. AMD never pulls from or
  holds credentials for the NVIDIA worker.
- Constraint: the host CPU is severely throttled. Avoid CPU-heavy preprocessing,
  large loader-worker counts, CPU inference fallback, and unbounded compilation.
  Verify that a workload is actually GPU-backed before leaving it running.

The AMD environment was last checked on 2026-08-15 with PyTorch
`2.9.1+rocm7.2.0.git7e1940d4`; it reported HIP 7.2 and enumerated one RX 7900
XTX through `torch.cuda`'s ROCm-compatible API. Re-run the lightweight verifier
after changing the lockfile or environment rather than assuming this remains
true.

## WSL restart safety rule

Never run `wsl --shutdown` on either remotely administered worker. If the AMD
worker's WSL instance genuinely must be restarted, reboot the complete Windows
host instead; WSL is registered as a startup task and should return after the
host boots. Notify the owner and record the reason before rebooting. A host
reboot is an exceptional recovery action, not a routine training step.

## Language inbox service

After code and environment validation, install the versioned user units with
`scripts/install_language_inbox_service.sh`. This writes only user-systemd
configuration and starts `brain-mri-language.path`; it does not restart WSL or
Windows. The service is a low-CPU, event-driven oneshot and leaves Ollama as the
only persistent language process.
