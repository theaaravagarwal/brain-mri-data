# Architecture decision: `glioma_4seq_v1`

## First prototype

Input is exactly four co-registered channels: `T1, T1ce, T2, FLAIR`.
Preprocess with orientation/spacing checks, nonzero z-score normalization per
channel, and fixed-size 3D patches. The output is a three-region BraTS mask
(whole tumour, tumour core, enhancing tumour); the clinical localization output
is the axis-aligned 3D bounding box of the predicted whole-tumour component.

Start with MONAI SegResNet. It is substantially cheaper than a transformer and
fits a 20-GB card with mixed precision and 96^3 patches. Add a global
classification/presence head from encoder features after the baseline is sound.
The box head is an experiment, not the source of truth: enforce consistency
with the mask-derived box loss.

| Hardware | Baseline settings |
| --- | --- |
| AMD RX 7900 XT (20 GB) | PyTorch ROCm build, 96^3 patch, batch 1, AMP, gradient accumulation 4 |
| One RTX A4000 (16 GB) | 96^3 patch, batch 1, AMP, gradient accumulation 4 |
| Four RTX A4000s | DDP, per-GPU batch 1--2, 96^3 or 128^3 after profiling |

Validate the exact ROCm/PyTorch/MONAI versions on the AMD machine before
committing to it; CUDA multi-GPU has the lower-friction training path.

## Novel-but-defensible extension

After the baseline, add modality-aware cross-attention plus a joint
segmentation/presence/box-consistency objective. This is publishable only with
ablations against SegResNet/nnU-Net-style baselines and a true external holdout;
architecture novelty alone is not enough.

## Modular future scope

Meningioma and pituitary models may later be added as separate, protocol-bound
modules, each with its own four-sequence data, label ontology, calibration, and
test set. A deterministic input protocol router may select a module. An LLM is
restricted to explaining structured outputs and uncertainty flags; it is not a
medical decision-maker or model router.
