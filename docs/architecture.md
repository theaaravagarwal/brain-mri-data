# Architecture decision: multi-tumor MRI on AMD ROCm

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

Validate the exact accelerator/PyTorch/MONAI combination on the training
machine before installing it. The dataset aggregator deliberately installs no
ROCm, PyTorch, or MONAI packages, and the project has no CUDA path.

## Novel-but-defensible extension

After the baseline, add modality-aware cross-attention plus a joint
segmentation/presence/box-consistency objective. This is publishable only with
ablations against SegResNet/nnU-Net-style baselines and a true external holdout;
architecture novelty alone is not enough.

## Modular future scope

Adult glioma, meningioma, pediatric glioma, and metastasis models are separate,
protocol-bound modules, each with its own four-sequence data, label ontology,
calibration, and test set. A deterministic input protocol router may select a
module. An LLM is restricted to explaining structured outputs and uncertainty
flags; it is not a medical decision-maker or model router.
