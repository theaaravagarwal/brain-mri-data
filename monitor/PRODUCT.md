# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Delegated: React, TypeScript, and Vite for the interface; a dependency-light
Node.js gateway for local serving and fixed, read-only SSH collection.

## Users

The repository owner, checking two private GPU workers from the current Mac
during training and evaluation work.

## Product Purpose

Show the NVIDIA and AMD workers' current reachability, GPU load, storage,
training progress, queue state, and recorded metrics in one glance. Success
means the user can tell what is running, whether it is healthy, how far it has
progressed, and when the information was last confirmed.

## Positioning

The dashboard reads the project's existing provenance-rich run artifacts and
host telemetry through fixed Tailscale SSH targets, while remaining local,
read-only, and useful during intermittent connectivity.

## Operating Context

The dashboard runs on the repository owner's Mac. It observes the NVIDIA CNN
worker at `theaa@100.64.0.3` and the AMD research worker at `b@100.64.0.5`.
Connections may be unstable, so the last successful snapshot must remain
visible with an explicit age and error state.

## Capabilities and Constraints

- Bind only to localhost and expose no training controls.
- Use fixed, allowlisted SSH targets and commands; browser input never becomes
  a command, path, or host name.
- Never modify either worker, restart WSL, or transfer MRI data.
- Keep network traffic and dependencies small; reuse SSH connections and poll
  conservatively.
- Keep all implementation and documentation inside `monitor/` and leave it
  uncommitted.

## Brand Commitments

Plain, direct wording. Minimal and calm without generic AI-dashboard styling,
decorative gradients, glows, excessive rounded cards, or theatrical motion.

## Evidence on Hand

The repository's run manifests, progress JSON, metrics JSONL, terminal monitor,
compute-host runbook, and verified SSH access provide the source data. The tool
must not invent missing measurements.

## Product Principles

- Freshness is part of every measurement.
- Preserve the last known truth instead of replacing it with an empty error.
- Put active work and failures before historical detail.
- Prefer one dependable read-only path over operational controls.
- Make dense scientific values scan quickly without hiding their units.

## Accessibility & Inclusion

Keyboard-operable, responsive, semantic, high-contrast, color-independent
statuses with reduced-motion, reduced-transparency, and increased-contrast
support.
