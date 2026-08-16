# Aggregate language pipeline review — 2026-08-16

## Decision

The aggregate CNN-to-language path is ready for internal research review. A
completed NVIDIA foreground-sampling summary was reduced to a 3.9-KB strict
envelope, transferred to AMD, automatically explained by `qwen3:14b` at 100%
GPU, and independently validated. No MRI data, case-level metrics, paths,
filenames, free text, or native subject identifiers crossed the boundary.

This is not clinical validation, guaranteed anonymization, autonomous job
execution, or approval to begin LoRA. Every artifact retains
`human_review_required`, `automatic_promotion: false`, and `executed: false`.

## Reproducibility and evidence

- Model/evaluation revision: `6c55d8ab25b199bc3827bb8a7296f8f787016224`
- Final transport-hardening revision: `12a402f32be98ad3b92758cc81bd7478a6458f84`
- CNN screen summary revision: `7b9f95ec09c82f32164e33ebf1d3f073dffa0a92`
- CNN summary SHA-256: `8b8cf77c5029028092874a5ded5c7ce5540cb5dc0d152b2a05cc40a202caccc4`
- Export UUID: `8d566a70-1466-4089-8c81-7be48aa5f399`
- NVIDIA envelope SHA-256: `0297369626ccbd0d6fbf32d6c9d509855ceefbe1617d3cf47a20da7e3c5e5d38`
- AMD explanation SHA-256: `25741c1893545b685fd53c071845ef59acc67141b739e131970d57352651cbd6`
- Explainer digest: `bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`
- Planner digest: `06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca`
- Host-local raw evidence remains under `runs/language/` and
  `runs/language-inbox/`; those directories are intentionally Git-ignored.

The AMD user path unit is enabled and active. It consumed the real envelope as
a oneshot job, wrote immutable JSON and escaped Markdown, and reported one
processed / zero quarantined. Ollama reported 100% GPU for both tested models.

## Verification results

| Gate | Result | Evidence |
| --- | ---: | --- |
| Full repository unit suite on AMD | 60/60 | Current ROCm environment |
| New deterministic language tests | 29/29 | Strict schemas, allowlist export, transport, replay, evidence, and authorization |
| Frozen v2 structured explainer | 6/6 | SHA-256 `ff9f9382c7def3d6dc0486fc25a15d6dac39936819c1acfaf0b1a72a6801854e` |
| Frozen v2 evidence answers | 8/8 | SHA-256 `b5d87f5f94fc3f1b25323c1e33be40a246de9fc4cabbb82ad4c3395656b6628a` |
| Frozen v2 planner | 6/6 | SHA-256 `62a4765e36bd902e4afc8da08e1330f3b3d6d94b747b0edc52245d0a208e1bbf` |
| Frozen adversarial planner | 12/12 | SHA-256 `496df0bf8f13a77e3831a32c9bff33960dc1ada0a5b91bb74e87f4388c416068` |

A final security pass pinned the only transfer destination, removed arbitrary
remote-command selection, enforced private key permissions, added bounded inbox
retention, made ingest failures generic, cleared loader/Python environment
overrides in the forced command, and sandboxed the systemd consumer. The
language contract/gateway suite passed 32/32 after these changes, and the
hardened service processed another synthetic envelope with zero quarantines.
A complete all-unavailable job-status envelope also crossed the same boundary
with SHA-256 `98a1be9f601947cf3b49b9a36381ce8a49d4813165f9e2cdd441fc12edbd03a0`;
it triggered no GPU work and authorized no proposal.

The first adversarial run scored 11/12 because the planner over-abstained on a
safe request to propose a job for human review. Clarifying the allowed action
fixed that case but exposed one JSON tool-injection miss. The final design moved
exact matches and obvious execution/tool/path/instruction-override requests to
deterministic preflight. The unchanged suite then passed 12/12. This supports
hardening the deterministic boundary rather than fine-tuning.

## CNN result carried by the envelope

All three foreground candidates passed the predeclared single-seed screen gate,
but they trade off metrics differently. `fg25` had the best small-lesion Dice,
box IoU, and HD95 among the candidates; `fg75` had the best overall Dice. These
are internal-validation, one-seed results only. Selecting a setting for the two
remaining fixed seeds is a separate human scientific decision and no run was
claimed or launched here.

## Storage and Git hygiene

- AMD WSL: 104 GB used, 853 GB available after removing 5.5 GiB of regenerable
  `uv` wheel cache created during lock verification.
- NVIDIA WSL: 486 GB used, 471 GB available.
- Language evaluation and inbox artifacts together use less than 140 KiB.
- Runtime artifacts, logs, partial transfers, caches, credentials, and local
  agent state are ignored; schemas, fixtures, service templates, summaries, and
  this review record remain tracked.
- The existing uncommitted NVIDIA PowerShell edit remains untouched.

## Required human review

The tested transfer used the pre-existing NVIDIA-to-AMD SSH credential. The
repository includes the forced ingest command and expects a dedicated identity,
but installing a `restrict,command=...` entry in AMD `authorized_keys` changes
state outside the repository-only work boundary. Future unattended NVIDIA push
is therefore fail-closed until the owner authorizes that one SSH configuration
change. AMD-side automatic explanation is already active.

Do not begin LoRA. First review the SSH credential hardening and the scientific
choice among `fg25`, `fg50`, and `fg75`.
