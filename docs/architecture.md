# Architecture decision: ISEF adult-glioma MRI study

## Competition scope

The ISEF project is one adult-glioma research question: whether provenance-
audited source-diverse training with PAMC (provenance-aware modality
consistency) improves external whole-lesion segmentation. Meningioma, pediatric glioma, metastasis, LLM
explanations, routing, and clinical deployment are outside this study.

Input is exactly four co-registered channels: `T1`, `T1ce`, `T2`, and `FLAIR`.
Every evaluated case requires a reviewed voxel mask. Whole-lesion boxes are
derived from the mask and never used as an independent label.

## Fixed scientific configuration

All three arms use a MONAI SegResNet, the same preprocessing, 80^3 patches,
batch size one, effective batch size four, fixed source manifests, and the same
three seeds. The arms are BraTS-only baseline, provenance-audited pooled
baseline, and PAMC extension. BraTS-Africa is locked before
training as the primary external test set.

## Independent training workers

| Worker | Runtime preset | Settings |
| --- | --- | --- |
| NVIDIA RTX 3060, 12 GB VRAM | `cuda` | CUDA FP16, batch 4, eight loader workers, prefetch 2, 80^3 patch, indexed chunk cache |
| AMD RX 7900 XTX, 24 GB VRAM, CPU-throttled host | `amd`; language only | bounded, GPU-verified structured-output/evidence benchmarks; avoid CPU-heavy work |

The RTX 3060 trains every frozen CNN `(arm, seed)` job. The AMD worker must not
run CNN study arms: it is reserved for the separately evaluated, constrained
research-language layer and never accesses images or clinical decisions. The
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
