# Architecture decision: adult-glioma MRI research system

## Research scope

The active study asks whether hierarchical nesting regularization improves
adult-glioma MRI segmentation across seeds. The completed internal comparison
rejected the candidate; the next stage is a six-checkpoint, inference-only
evaluation on one locked external cohort. Meningioma, pediatric glioma,
metastasis, routing, and clinical deployment are outside this study.

Input is exactly four co-registered channels: `T1`, `T1ce`, `T2`, and `FLAIR`.
Every evaluated case requires a reviewed voxel mask. Whole-lesion boxes are
derived from the mask and never used as an independent label.

## Frozen scientific configuration

The external stage uses the six completed nnU-Net checkpoints: baseline and
nesting-penalty arms for seeds `20260821`, `20260822`, and `20260823`.
BraTS-Africa is locked before inference. No external result may select a seed,
change a checkpoint, tune preprocessing, or remove a case.

## Independent training workers

| Worker | Runtime preset | Settings |
| --- | --- | --- |
| NVIDIA RTX 4060, 8 GB VRAM | `cuda` | current primary CUDA host; profile must be re-benchmarked before any future training |
| AMD RX 7900 XTX, 24 GB VRAM, CPU-throttled host | `amd`; language only | bounded, GPU-verified structured-output/evidence benchmarks; avoid CPU-heavy work |

The completed confirmation models were trained on the AMD worker. The current
primary CUDA host is the RTX 4060 worker recorded in the compute runbook; no new
training is part of the external evaluation. The constrained research-language
layer never accesses images or clinical decisions. The
controller locks manifests and claims, while workers retain their own legal
local raw-data copy. The language boundary is narrower than general experiment
synchronization: NVIDIA constructs a strict aggregate-only research envelope
and pushes it one-way to AMD; AMD has no access to MRI data, paths, case-level
results, or NVIDIA credentials. The AMD planner sees only a validated status
snapshot intersected with the frozen run matrix, and can emit only a
non-executing proposal for human review.

The importer environment has no training framework. Use `uv sync --extra cuda`
or `uv sync --extra amd` in a separate Python 3.12 environment, never both.
See [the compute-host runbook](compute-hosts.md) for canonical SSH targets,
repository paths, workload placement, and the WSL restart rule.
