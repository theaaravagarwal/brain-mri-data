# Plan: Four-sequence brain MRI dataset aggregation

**Generated:** 2026-08-12
**Complexity:** High

## Overview

Create a provenance-first corpus for a glioma-only 3D MRI
segmentation and mask-derived bounding-box system. Keep raw sources independent
and make training composition a manifest choice.

## Prerequisites

- Kaggle credentials for Kaggle sources; Hugging Face authentication only for a gated source.
- A Python 3.11--3.13 environment and 1 TB usable storage. Budget about 251 GB
  for the recommended raw corpus, 150--300 GB for preprocessed/cache artifacts,
  and 100 GB for experiments/checkpoints; monitor actual disk use.
- Confirm data-use terms at download time; catalog metadata is not a legal determination.

## Sprint 1: Acquire and inventory

**Goal:** Download one source at a time and produce immutable case manifests.

### Task 1.1: Fetch an approved source
- **Location:** `src/brain_mri_data/cli.py`
- **Complexity:** 4
- **Acceptance:** Provider downloader records resolved source directory and does not overwrite it.
- **Validation:** `brain-mri-data fetch brats2020_kaggle --dry-run`.

### Task 1.2: Index four-sequence cases
- **Location:** `src/brain_mri_data/indexer.py`
- **Complexity:** 7
- **Acceptance:** Each accepted record has all modalities, a mask, source ID, and source paths.
- **Validation:** `brain-mri-data index brats2020_kaggle`.

## Sprint 2: Quality gate and data split

**Goal:** Reject non-conforming cases and produce reproducible patient-level splits.

### Task 2.1: Validate NIfTI geometry and mask boxes
- **Location:** `src/brain_mri_data/qc.py`
- **Complexity:** 7
- **Acceptance:** Affine/shape mismatches, empty masks, and missing files are reported; boxes include voxel and world coordinates.
- **Validation:** `brain-mri-data validate brats2020_kaggle`.

### Task 2.2: Build source-aware splits
- **Location:** `src/brain_mri_data/splits.py`
- **Complexity:** 5
- **Acceptance:** No patient ID appears in more than one split.
- **Validation:** fixed-seed repeat produces byte-identical output.

## Sprint 3: Train and benchmark

**Goal:** Establish 3D SegResNet baseline, then test a multi-task extension.

### Task 3.1: Train segmentation baseline
- **Location:** future `training/`
- **Complexity:** 8
- **Acceptance:** Dice/HD95 and 3D box metrics are reported on locked test cases.

### Task 3.2: External validation
- **Dependencies:** provenance review of a non-overlapping external source.
- **Acceptance:** No source or patient overlap with training; metrics reported separately.

## Risks

- Kaggle/HF mirrors can overlap. Mitigation: never merge until case IDs/files are audited.
- 2D classification images are not 3D MRI volumes. Mitigation: catalog them separately.
- Storage is finite. Mitigation: retain raw data only once, keep a measured disk budget, and do not duplicate full-volume preprocessing exports.
