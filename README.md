# Brain MRI dataset aggregator

This project creates provenance-preserving manifests for four-sequence brain MRI
tumour datasets. It is designed for a **multi-module 3D protocol**: every
evaluated case requires T1, T1ce/T1c, T2, and FLAIR/T2f, plus a voxel mask.
3D bounding boxes are derived from the mask; they are never treated as a
separate, weaker ground truth.

The downloader stores each source once under `data/raw/`. Indexing writes
portable manifests under `data/manifests/` that point at those files, so a
dataset is not duplicated during standardization.

## Setup

Use `uv` with Python 3.12 on the eventual training machine (medical-imaging
packages may lag behind Python 3.14):

```bash
uv python install 3.12
uv sync --extra qc
```

This project intentionally does not install PyTorch, ROCm, or MONAI as part of
the data-import environment. Training targets the AMD RX 7900 XT exclusively;
the ROCm-enabled PyTorch stack will be installed separately inside WSL2 after
the Windows AMD WSL driver exposes the GPU. CUDA is not part of this project.

See [the AMD ROCm setup guide](docs/amd-rocm-setup.md) for the exact host and
Python-environment procedure.

Kaggle access requires `~/.kaggle/kaggle.json` or `KAGGLE_*` credentials.
Hugging Face access requires `huggingface-cli login` for gated repos.

## Core workflow

```bash
# See the approved catalog and storage-budget profiles.
uv run brain-mri-data catalog

# Download a source only after checking its license and terms.
uv run brain-mri-data fetch brats2020_kaggle

# Build case records (no image copying) and reject incomplete cases.
uv run brain-mri-data discover brats2020_kaggle  # inventory only; writes nothing
uv run brain-mri-data index brats2020_kaggle
uv run brain-mri-data validate brats2020_kaggle

# Emit a deterministic, patient-level split and MONAI datalist.
uv run brain-mri-data split brats2020_kaggle --seed 20260812
uv run brain-mri-data export-monai brats2020_kaggle
```

With the available 1-TB budget, start with `glioma_train_plus_locked_external_1tb`:
BraTS 2020, UTSW-Glioma, UCSF-PDGM, and BraTS-Africa for training/development;
keep UPENN-GBM locked as the final external test set. Its raw-data estimate is
about 251 GB, leaving room for downloads, preprocessing caches, model
checkpoints, and transfer packaging. Do not use an unverified mirror or a 2D
slice dataset as an independent test set: catalog it as `auxiliary_2d` only,
because it may overlap a primary BraTS release.

The catalog also includes `manual_tcia` sources. These are high-value datasets
whose official portals sometimes require TCIA tools or Aspera; download them
manually into the matching `data/raw/<source_id>/` directory, then use the same
`index`, `validate`, `split`, and `export-monai` commands. They are never
silently fetched by the CLI.

`discover` scans nested NIfTI layouts without modifying them. `index` recognizes
standard modality aliases (such as `t1n`, `t1c`, `t2w`, and `t2f`), refuses
ambiguous duplicates rather than guessing, hashes every accepted source file,
and emits a discovery report plus a detailed exclusion manifest.

Before any training split is accepted, audit the proposed source lineage:

```bash
uv run brain-mri-data audit-experiment \
  --train brats2020_kaggle utsw_glioma_tcia \
  --test upenn_gbm_tcia --strict
```

The audit blocks cross-module experiments, declared institution/challenge
lineage risks, unreviewed REMIND labels, and exact overlapping patient IDs once
case manifests exist. A passing audit still requires a documented manual
provenance review for publication.

## Model direction

Use task-specific 3D models for adult glioma, meningioma, pediatric glioma, and
metastases. Start each module with a MONAI `SegResNet` baseline and compare
shared-encoder or multi-task extensions only after the independent baselines
are sound. Boxes are computed from predicted whole-lesion masks; metastases can
produce multiple connected-component boxes.

An LLM should only turn validated, structured model outputs into a clearly
labelled research explanation. It must not select the diagnostic model, inspect
raw images, override a result, or make treatment claims. Model routing should
be deterministic from the protocol (`glioma_4seq_v1`), not agentic.

See `docs/architecture.md` for the training path and `docs/data-plan.md` for
the project plan. See `docs/multitumor-scope-audit.md` for the consequences of
expanding beyond glioma.
