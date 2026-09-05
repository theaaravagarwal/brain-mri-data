# Research model card: glioma-segresnet-20260828

Evidence snapshot: September 4, 2026. This card describes the frozen serving
artifact; it is not a reconstructed training report.

| Field | Recorded contract |
| --- | --- |
| Task | Binary whole-lesion segmentation of compatible four-modality brain MRI |
| Model ID | `glioma-segresnet-20260828` |
| Checkpoint SHA-256 | `121422a861bbe7affaa5e161058e69eea737b2390651c3c03ea20256969e99e5` |
| Implementation | `PamcSegResNet(init_filters=32, source_count=1)` |
| Segmenter | MONAI SegResNet, 3D, 4 input channels, 1 output channel, dropout parameter 0.2 |
| Serving mode | Evaluation mode; segmentation logits used; source-head output unused |
| Inputs | T1, T1ce, T2, FLAIR; finite 3D NIfTI volumes with matching geometry |
| Normalization | Per modality, z-score nonzero voxels; zero background retained |
| Frozen benchmark inference | 80 × 80 × 80 windows; overlap 0.5; window batch 1; CUDA FP16 autocast |
| Output | Sigmoid threshold ≥0.5; binary uint8 NIfTI, preserved input geometry |
| Reference interpretation | Supported nonzero labels collapsed to one foreground class |
| Current language renderer | Local qwen3:4b; exact renderer identity recorded in result evidence |

Sources: [network](../training/pamc.py), [runner](../scripts/run_4060_research_inference.py),
[benchmark plan](../config/analysis/fixed-segresnet-external-benchmark.json).
The class contains a source adversary; its presence does not establish which
training arm produced the checkpoint. Verify training records separately.

## Metric definitions and known limitations

With foreground voxel counts TP, FP, FN: Dice = 2TP/(2TP+FP+FN),
IoU = TP/(TP+FP+FN), precision = TP/(TP+FP), recall = TP/(TP+FN).
Consult the runner for denominator edge cases. It rejects empty reference masks.
An empty prediction against a nonempty reference receives zero overlap.

HD95 uses the 95th percentile of concatenated bidirectional nearest-surface
distances with voxel spacing in millimeters. It is unavailable when either
surface is empty. This exact implementation should be named when comparing
against other tools with different HD95 conventions.

The model's recorded 60-case mean Dice is 0.8778012350; one output is empty.
See the [synopsis](project-synopsis.md) for distributions and limitations.
Neither subregion segmentation, disease classification, calibrated confidence,
nor tumor-negative screening performance is established. Matching affine
metadata alone does not prove anatomical alignment. Raw clinical acquisition
compatibility is not demonstrated by results on prepared research volumes.

## Provenance still needed for a complete training record

- Exact training manifest/hash, patient split/hash, source citations and licenses.
- Overlap audit linking development and external source identities, including
  copies or derived versions of the same patient scans.
- Training code commit and configuration/hash: seed, loss, optimizer, epochs,
  sampling, augmentation, selected epoch, and checkpoint selection criterion.
- Actual training hardware, runtime, and package versions for this checkpoint.
- Model-weight distribution terms, checked separately from software licenses.

These fields are unresolved in this card. Existing historical experiments must
be linked to this checkpoint by evidence before their settings are attributed
to it. Model identity is verified more strongly than training reconstruction.

## Use and change control

Researchers can inspect and export an initial outline for review. Corrections,
diagnostic decisions, and confidence judgments are not produced by this model.
Original inputs and sanitized viewing data follow the application retention
contract; the LLM receives metadata only.

The frozen benchmark may be analyzed descriptively but must not select a new
threshold or model. A proposed change needs development-only evidence, a new
card and checkpoint hash, and a separately frozen evaluation cohort. Application
rollback scripts exist; replacing a checkpoint also requires updating and
testing the explicit frozen-model contract. Arbitrary checkpoint swapping is
not currently a supported user workflow.
