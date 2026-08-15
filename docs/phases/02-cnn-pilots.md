# Phase 02 — CNN pilots

Status: in progress.

- Completed AMD one-epoch BraTS-only runs: seeds `20260812` and `20260814`.
  They are engineering checks only and are excluded from the scientific study.
- CUDA is assigned the independent `20260813` seed after resumable data synchronization.
- Next acceptance point: three 10-epoch BraTS-only learning curves with runtime, checkpoint, telemetry, and internal-validation records.
- These are engineering pilots, not external-performance results.
- All subsequent CNN study jobs run on CUDA; AMD is reserved for the bounded
  language layer after its GPU check passes.
