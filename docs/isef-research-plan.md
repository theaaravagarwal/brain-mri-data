# ISEF research plan: provenance-robust adult-glioma MRI segmentation

## Research question and hypothesis

Can provenance-audited, source-diverse four-sequence MRI training with
provenance-aware modality consistency (PAMC) improve external adult-glioma
segmentation relative to a BraTS-only baseline? PAMC uses a source-adversarial
encoder and requires its prediction after one intentionally masked MRI sequence
to agree with its full four-sequence prediction. The hypothesis is that PAMC
will improve whole-lesion Dice and retain more performance under a controlled
single-sequence corruption on the locked BraTS-Africa cohort.

## Design

All arms use the same SegResNet, preprocessing, 80^3 patches, effective batch
size, optimization budget, and three fixed seeds. The independent variable is
the training arm: BraTS-only baseline, pooled-source baseline, or PAMC. The
primary dependent variable is per-case external whole-lesion Dice; HD95,
mask-derived box IoU, and clean-to-corrupted Dice retention are secondary
outcomes.

BraTS 2020, UTSW-Glioma, and UCSF-PDGM are development sources only. BraTS-
Africa is locked before training and is never used for preprocessing choices,
hyperparameter selection, or early stopping. UPENN-GBM is excluded from the
primary claim until its declared potential overlap with the BraTS ecosystem is
resolved by a documented audit.

## Provenance, statistics, and limits

Every case must have all four required sequences, a reviewed canonical
whole-lesion mapping, passing geometry QC, source file hashes, and a frozen
study manifest. The primary comparison uses paired patient-level bootstrap
confidence intervals and a paired permutation test; results from CUDA and ROCm
are reported separately.

This is retrospective research on public, de-identified data. It collects no
new participant data, makes no diagnoses or treatment recommendations, and
does not include an LLM or public-facing clinical application. Confirm the
documentation route with the affiliated-fair SRC before data work; local fairs
can impose additional requirements.

# First execution: internal pilot

Use the BraTS 2020-only pilot only to verify the full training pipeline and
measure runtime. Its validation score is **not** an external-test result and
must not be used for final model selection claims.

```bash
./scripts/train_amd_pilot.sh 1
```

The script creates `data/manifests/glioma.pilot.json` once and records output
under `runs/glioma-pilot--amd--brats--20260812`. The full study remains blocked
until its independent external cohort, label mapping, and manual provenance
review are ready.
