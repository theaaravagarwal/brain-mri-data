# Phase 04 — constrained language layer

Status: implementation in progress; model evaluation starts after the first CNN learning-curve review.

- Planner: `qwen3-coder:30b`, read-only status/run-matrix tools and pre-approved job proposals only.
- Explainer: `qwen3:14b`, validated structured outputs plus frozen evidence cards only.
- Evaluate held-out structured faithfulness and evidence-grounded responses.
- A synthetic structured-output LoRA is allowed only if the frozen explainer misses the predeclared benchmark threshold; medical-literature fine-tuning is out of scope.
