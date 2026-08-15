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
| NVIDIA RTX 3060, 12 GB VRAM | `cuda` | CUDA FP16, two loader workers, 80^3 patch |
| AMD RX 7900 XTX, CPU-limited host | language only | bounded, structured-output/evidence benchmarks after GPU verification |

The RTX 3060 trains every frozen CNN `(arm, seed)` job. The AMD worker must not
run CNN study arms: it is reserved for the separately evaluated, constrained
research-language layer and never accesses images or clinical decisions. The
controller locks manifests and claims, while workers retain their own legal
local raw-data copy. Synchronize only configs, metrics, checkpoints, and
non-identifying prediction artifacts over Tailscale.

The importer environment has no training framework. Use `uv sync --extra cuda`
or `uv sync --extra amd` in a separate Python 3.12 environment, never both.
