# Audit: expanding from glioma-only to multi-tumor MRI

## Verdict

Retain the importer, provenance manifests, NIfTI QC, and mask-to-3D-box logic.
Replace the single `glioma_4seq_v1` research protocol with a **shared
four-sequence contract plus task-specific modules**. Do not concatenate cases
from different tumor challenges into one segmentation target.

## What remains unchanged

- Required input contract: co-registered `T1`, `T1ce`, `T2`, `FLAIR` volumes.
- Case-level provenance, hashes, geometry/QC checks, patient-level splits.
- Whole-lesion binary mask and its derived 3D bounding box.
- PyTorch/MONAI training infrastructure and 3D patching strategy.

## What changes

| Area | Glioma-only | Multi-tumor program |
| --- | --- | --- |
| Dataset registry | One label ontology | Per-source ontology and explicit mapping to a shared binary `whole_lesion` target |
| Segmentation | BraTS glioma subregions | One decoder/head per module; optional shared encoder only after ablation evidence |
| Classification | Glioma presence/subregion | Dataset-supported diagnosis head only; never infer a histologic class from segmentation labels alone |
| Boxes | One whole-tumor box | Same derivation, but allow multiple connected-component boxes for metastases/multifocal disease |
| Splits | Internal + glioma external | Locked external holdout per module and no cross-module claims |
| Routing | Fixed glioma protocol | Deterministic protocol selector based on declared task/validated metadata; not an LLM decision |

## Modules that fit the mandatory four-sequence rule

- **Adult glioma:** BraTS, UTSW-Glioma, UCSF-PDGM, UPENN-GBM, BraTS-Africa.
- **Adult meningioma:** BraTS MEN has four sequences and dense masks.
- **Pediatric tumor / pediatric high-grade glioma:** BraTS-PED has four sequences,
  but its label ontology and population differ from adults; keep separate.
- **Metastases:** add only after confirming a source has all four required
  sequences and compatible masks; lesion count requires multi-box evaluation.

## Current hard gap: pituitary tumors

The cataloged Kaggle pituitary data are 2D classification images, not matched
four-sequence 3D volumes with masks. They cannot train or validate this protocol.
Do not fabricate 3D samples by stacking unrelated 2D images. A pituitary module
is blocked until a suitable volumetric, annotated source is identified and its
license/label definitions are reviewed.

## Recommended staged deliverable

1. Make the registry multi-protocol and map every mask to `whole_lesion`.
2. Train adult-glioma and meningioma modules independently with their own
   subregion heads and locked source holdouts.
3. Add pediatric and metastasis modules only after source-specific label/QC
   adapters pass review.
4. Evaluate a shared-encoder/multi-head model against independent modules. Keep
   it only if it improves each module's locked external test performance.

## LLM role

An LLM may render a research-oriented explanation from a structured, validated
result (`module`, input-QC, mask metrics, box coordinates, uncertainty flags).
It may not choose the module from images, re-label a scan, override a model, or
make medical claims. The actual router is deterministic and fails closed when
the required four sequences or declared protocol are absent.
