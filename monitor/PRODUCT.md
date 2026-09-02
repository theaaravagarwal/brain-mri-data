# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Delegated: React, TypeScript, and Vite for the interface; a dependency-light
Node.js gateway for local serving and fixed, read-only SSH collection.

## Users

The repository owner and research collaborators running one study at a time on
the private NVIDIA research worker.

## Product Purpose

Accept exactly four MRI modalities (T1, T1ce, T2, and FLAIR), validate their
identity and geometry, run one fixed CNN locally, and return a segmentation,
an exact model/checkpoint receipt, and a research-only metadata explanation.
Success means no inference can begin before validation passes, every result is
traceable to the frozen checkpoint, and the language model never receives scan
voxels or creates clinical conclusions.

## Positioning

The study workflow is the primary product. Existing worker telemetry, model
evidence, explanations, and review-only proposals remain secondary views for
research operations. Processing stays on the private NVIDIA worker and remains
useful without the optional local language model through a deterministic
validated explanation.

## Operating Context

The browser reaches a gateway bound only to the NVIDIA worker's exact Tailscale
address (or loopback for local development). The NVIDIA CNN worker is the sole inference target. The
AMD research worker may be offline and is not required for this workflow.
Operational telemetry can still show last-known values with explicit age and
error state.

## Capabilities and Constraints

- Bind only to loopback or an explicit Tailscale IPv4 address and expose no
  training controls; mutation routes reject non-tailnet clients and mismatched
  Host/Origin headers.
- Accept exactly one `.nii.gz` file for each of T1, T1ce, T2, and FLAIR, with
  streaming per-file and total-size limits.
- Require matching shape, affine geometry, finite voxels, and bounded volume
  dimensions before inference.
- Run only `glioma-segresnet-20260828` with the allowlisted checkpoint SHA-256;
  recheck the digest for every inference and permit one GPU job at a time.
- Return a binary research segmentation plus machine-readable receipt and
  explanation artifacts; preserve the source geometry in the output.
- Give the optional local LLM validated result metadata only. Reject its output
  unless it preserves the evidence fields and contains no clinical claim.
- Remove uploaded volumes after processing and expire result artifacts after
  24 hours; allow the user to clear results sooner.
- Use fixed, allowlisted SSH targets and commands; browser input never becomes
  a command, path, or host name.
- Never send MRI data to the language model or any external service.
- Keep network traffic and dependencies small; reuse SSH connections and poll
  conservatively.

## Brand Commitments

Plain, direct wording. Minimal and calm without generic AI-dashboard styling,
decorative gradients, glows, excessive rounded cards, or theatrical motion.

## Evidence on Hand

The repository's run manifests, progress JSON, metrics JSONL, terminal monitor,
compute-host runbook, and verified SSH access provide the source data. The tool
must not invent missing measurements.

## Product Principles

- Validation is a hard gate, not a visual hint.
- A result is inseparable from its exact model and checkpoint receipt.
- The deterministic explanation is always available; LLM rendering is
  optional and subordinate to validated metadata.
- Research limitations appear with the result, never behind a tooltip.
- Freshness remains part of every operational measurement.

## Accessibility & Inclusion

Keyboard-operable, responsive, semantic, high-contrast, color-independent
statuses with reduced-motion, reduced-transparency, and increased-contrast
support.
