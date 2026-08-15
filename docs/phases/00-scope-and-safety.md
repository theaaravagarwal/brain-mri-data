# Phase 00 — scope and safety

Status: complete when the implementation accepts only the boundaries below.

- Research-demo use on public, de-identified data only.
- CNN protocol selection is deterministic from declared, validated metadata.
- Language models receive structured, non-identifying outputs only; never MRI pixels, paths, or patient identifiers.
- They may not diagnose, make treatment recommendations, alter masks, or select a medical model.
- The planner may only propose fixed run-matrix jobs; policy code validates every proposal.
