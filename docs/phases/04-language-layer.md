# Phase 04 — constrained language layer

Status: bounded local evaluation in progress; language results remain separate from CNN claims.

- Planner: `qwen3-coder:30b`, read-only status/run-matrix tools and pre-approved job proposals only.
- `brain-mri-data language propose-job` validates a planner proposal against the frozen run matrix; execution remains a separate human-controlled step.
- Explainer: `qwen3:14b`, validated structured outputs plus frozen evidence cards only.
- Evaluate held-out structured faithfulness and evidence-grounded responses.
- The frozen planner benchmark includes exact allowed-job selection, ambiguity,
  unauthorized profiles, prompt injection, and execution-request abstention.
- Promotion requires every frozen case to pass and `ollama ps` to report 100% GPU.
- Benchmark outputs are immutable JSONL artifacts with the model name and
  per-case wall time, token counts, and generation throughput.
- A synthetic structured-output LoRA is allowed only if the frozen explainer misses the predeclared benchmark threshold; medical-literature fine-tuning is out of scope.
