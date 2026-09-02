# Compute-host runbook

This is the canonical operational map for the private Tailscale prototype.
Raw MRI data stays on its existing worker. Synchronize code and approved,
non-identifying aggregate artifacts only.

## Application and language host

- SSH: `software@100.64.0.7`
- Repository: `/home/software/Documents/.aa/brain`
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU, 16 GB VRAM
- Services: `brain-mri-prototype.service` and `brain-mri-ollama.service`
- Role: application, fixed-checkpoint CNN inference, and metadata-only language
  explanation. The LLM never receives MRI bytes, paths, or identifiers.
- Serving checkpoint changes require frozen evaluation evidence and human review.

The application binds to `100.64.0.7:4173`. The stable user entrypoint remains
`http://100.64.0.1:4173` through the fixed Tailscale proxy.

## RTX 4060 CNN worker

- SSH: `software@100.64.0.1`
- Repository: `/home/software/Documents/.aarav/brain`
- GPU: NVIDIA GeForce RTX 4060, 8 GB VRAM
- Role: reproducible BraTS CNN training with the `cuda-4060-safe` profile
- Queue: `brain-mri-cnn-4060-queue.service`
- Proxy: `brain-mri-tailnet-proxy.service`

`/home/software/Documents/.aa/brain` does not exist on this host. Do not deploy
training or proxy files there.

## RTX 3060 CNN worker

- SSH: `theaa@100.64.0.3`
- Repository: `/home/theaa/Documents/brain-mri-data`
- GPU: NVIDIA GeForce RTX 3060, 12 GB VRAM
- Role: independent BraTS, pooled, and PAMC training/evaluation
- Queue: `brain-mri-cnn-3060-queue.service`

The worker may not have `nvidia-smi` on its interactive shell `PATH`; the
collector resolves the WSL NVIDIA binary explicitly.

## Operating rules

- Run independent experiments per host; do not use cross-host DDP.
- Keep run directories immutable and preserve incomplete failures.
- Never promote a checkpoint automatically or tune against a locked external
  cohort.
- Keep the serving checkpoint available while experimental queues run.
- Do not reboot a worker for routine recovery. Record and report any exceptional
  host restart before taking it.
