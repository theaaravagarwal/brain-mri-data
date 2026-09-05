# Brain MRI dataset aggregator

## Working research prototype

Open <http://100.64.0.1:4173> from the tailnet. Choose **Use sample** or
upload T1, T1ce, T2, and FLAIR volumes, then run and download the outline.
The application runs on `.7`; `.1` provides the stable proxy. No training is
needed to use it, and `.3` is excluded from current operations.

See [the prototype operating guide](docs/prototype-operations.md) for acceptance
checks, model identity, service recovery, and the definition of completion.
The dataset and historical research workflows below remain available separately.

For project documentation, start with the [research synopsis](docs/project-synopsis.md),
[model card](docs/research-model-card.md), and [evaluation protocol](docs/research-evaluation-protocol.md).
These link recorded results to their evidence and identify the studies still needed.

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

This project intentionally does not install PyTorch, ROCm, CUDA, or MONAI in
the default data-import environment. Training uses one of two mutually
exclusive extras on separate workers:

```bash
# AMD RX 7900 XTX ROCm worker (WSL2/Linux)
uv sync --extra amd

# NVIDIA RTX 3060 12-GB CUDA worker (Linux/WSL2)
uv sync --extra cuda
```

Never enable both extras in one environment. They contain different PyTorch
builds. The CUDA profile is the sole CNN-study runtime. The AMD environment is
kept for ROCm validation and the separately bounded research-language worker;
it must not run frozen CNN study arms.

See [the AMD ROCm setup guide](docs/amd-rocm-setup.md) and
[the NVIDIA CUDA setup guide](docs/cuda-setup.md) for host checks and
verification. The canonical SSH targets, repository paths, workload roles, and
restart safety rules are recorded in [the compute-host runbook](docs/compute-hosts.md).

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
uv run brain-mri-data verify-files brats2020_kaggle
uv run brain-mri-data validate brats2020_kaggle

# Emit a deterministic, patient-level split and MONAI datalist.
uv run brain-mri-data split brats2020_kaggle --seed 20260812
uv run brain-mri-data export-monai brats2020_kaggle
```

With the available 1-TB budget, `glioma_train_plus_locked_external_1tb` is a
useful acquisition-budget profile (about 251 GB raw), leaving room for
downloads, preprocessing caches, checkpoints, and transfer packaging. It is
not an experimental allocation. It belongs to the historical PAMC plan and is
not the current study allocation. The proposed current extension evaluates six
already frozen Product V2 checkpoints on one locked BraTS-Africa cohort; it
does not add a training source or permit retraining. Do not use an unverified
mirror or a 2D slice dataset as an independent test set: catalog it as
`auxiliary_2d` only, because it may overlap a primary BraTS release.

The catalog also includes `manual_tcia` sources. These are high-value datasets
whose official portals sometimes require TCIA tools or Aspera; download them
manually into the matching `data/raw/<source_id>/` directory, then use the same
`index`, `validate`, `split`, and `export-monai` commands. They are never
silently fetched by the CLI.

`discover` scans nested NIfTI layouts without modifying them. `index` recognizes
standard modality aliases (such as `t1n`, `t1c`, `t2w`, and `t2f`), refuses
ambiguous duplicates rather than guessing, hashes every accepted source file,
and emits a discovery report plus a detailed exclusion manifest. New manifests
store source-relative paths, allowing the same indexed manifest to resolve
against each worker's local `data/raw/` copy; SHA-256 hashes preserve identity.

Before any training split is accepted, audit the proposed source lineage:

```bash
uv run brain-mri-data audit-experiment \
  --train brats2020_kaggle utsw_glioma_tcia \
  --test brats_africa_tcia --strict
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

The first language prototype is aggregate-only: a completed NVIDIA research
screen is exported as strict canonical JSON, pushed one-way to the AMD worker,
validated again, and explained by local Ollama. Individual MRI cases and paths
never cross this boundary. See [Phase 04](docs/phases/04-language-layer.md) for
the contract, commands, automation, and human-review gate.

See `docs/architecture.md` for the training path and `docs/data-plan.md` for
the project plan. See `docs/multitumor-scope-audit.md` for the consequences of
expanding beyond glioma.

## Frozen external evaluation workflow

The current study asks whether a hierarchical nesting penalty improves
adult-glioma segmentation across three seeds. The internal result rejected that
candidate. See the [study scope](docs/product-v2-study-scope.md).

No additional training is part of this track. Before external inference, run
the readiness report:

```bash
brain-mri-data external-readiness \
  config/analysis/product-v2-external-readiness.yaml --strict
```

The command fails until the label mapping, provenance review, source manifests,
QC, and immutable external lock are present. The complete sequence is in the
[external-evaluation runbook](docs/product-v2-external-evaluation-runbook.md).
It is an inference-only research comparison, not a diagnosis system, clinical
validation, or LLM study.
# CUDA pilot run

Use CUDA for every CNN pilot and frozen CNN study run. The AMD worker is not a
CNN fallback.

```bash
# CUDA worker: resumably copy only BraTS 2020, then run seed 20260813
./scripts/sync_cuda_pilot.sh
```

For a live CUDA health/telemetry view, run:

```bash
brain-mri-data monitor

# Noninteractive or script-friendly snapshots:
brain-mri-data monitor --once
brain-mri-data monitor --json --once
```

For runs started after the live-progress update, the dashboard also shows the
current training/validation phase, epoch, batch or case count, running loss,
and elapsed epoch time. The training pane itself uses `tqdm` progress bars.
