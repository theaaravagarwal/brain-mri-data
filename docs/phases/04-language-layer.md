# Phase 04 — constrained language layer

Status: aggregate research-summary pipeline implementation; language results remain separate from CNN claims.

## Aggregate CNN-to-language boundary

The production prototype explains completed, aggregate internal-validation
screens only. It does not export individual cases, case metrics, native IDs,
paths, filenames, dates, images, masks, DICOM fields, or free text. Its artifact
is described as a **sanitized/direct-identifier-free research envelope**, not as
anonymous or HIPAA-compliant.

`ResearchRunSummaryEnvelopeV1` is the single strict contract on both workers.
The NVIDIA worker constructs it from an allowlist, writes canonical JSON and a
SHA-256 receipt once, and pushes only that envelope to AMD. AMD validates it
again, rejects duplicate keys, malformed or oversized inputs, replayed UUIDs,
symlinks, unknown fields, non-finite metrics, and inconsistent gates, then
atomically makes it ready for explanation.

```bash
# NVIDIA, only after the foreground-screen summary is complete:
brain-mri-data language export-run-summary \
  analyses/pilots/glioma-v4-foreground-screen/results.json \
  --runs-root runs --outbox runs/language-outbox \
  --run-group-id glioma-v4-foreground-screen

# The dedicated transfer key must be restricted to the ingest command on AMD.
brain-mri-data language push runs/language-outbox/<export-id>.json \
  --identity runs/language-transport/brain_mri_language_ed25519
```

`scripts/export_and_push_foreground_summary.sh` makes that operation
idempotent. Future foreground-screen queues call it automatically when the
restricted identity exists; an absent key leaves the completed CNN result
untouched and reports that transfer is pending.

The AMD `brain-mri-language.path` user unit activates a low-CPU oneshot
consumer when a validated envelope arrives. It serializes Ollama work, records
the exact model digest and hashes, independently validates every cited field and
value, and writes immutable JSON plus escaped Markdown. Invalid model output is
quarantined and never displayed.

The SSH key entry must use OpenSSH `restrict` plus the forced absolute command
`/home/b/brain-mri-data/scripts/ingest_language_envelope.sh`. The automation key
must not grant an interactive shell, PTY, forwarding, or arbitrary SFTP access.

Design references: [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs),
[OWASP prompt-injection defenses](https://genai.owasp.org/llmrisk/llm01-prompt-injection/),
[OpenSSH authorized-key restrictions](https://man.openbsd.org/sshd.8),
[NIST Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf),
and [HHS de-identification guidance](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html).

- Planner: `qwen3-coder:30b`, read-only status/run-matrix tools and pre-approved job proposals only.
- `brain-mri-data language propose` intersects a validated status snapshot with
  the frozen run matrix before prompting or validating. A proposal always has
  `executed: false`; execution remains a separate human-controlled step.
- `brain-mri-data language export-job-status` requires an explicit state for
  every frozen matrix job and derives `proposal_allowed` rather than trusting it
  as input. The checked-in example marks every job unavailable; a human or
  controller must deliberately create a new snapshot before any proposal is
  possible.
- Explainer: `qwen3:14b`, validated structured outputs plus frozen evidence cards only.
- Evaluate held-out structured faithfulness and evidence-grounded responses.
- The frozen planner benchmark includes exact allowed-job selection, ambiguity,
  unauthorized profiles, prompt injection, and execution-request abstention.
- Promotion requires every frozen case to pass and `ollama ps` to report 100% GPU.
- Benchmark outputs are immutable JSONL artifacts with the model name and
  per-case wall time, token counts, and generation throughput.
- Versioned v2 explainer fixtures exercise six structured-result cases and
  eight source-grounded safety/provenance questions without replacing the
  original smoke fixtures.
- `scripts/run_language_eval_v2.sh` runs the two explainer suites followed by
  the planner suite, records fixture hashes, refuses an existing revision
  directory, and fails unless each served model is observed at 100% GPU.
- `config/language-eval-v2.yaml` freezes the models, fixture versions, pass
  thresholds, generation settings, GPU requirement, and no-execution rule.
- `scripts/run_language_prototype.py explain` consumes only a validated CNN
  result envelope and validates the generated explanation before writing an
  immutable artifact. `plan` reads the frozen run matrix, treats the request as
  untrusted, validates any proposal, records `executed: false`, and never calls
  the run-claim or training paths.

Prototype smoke commands:

```bash
./.venv/bin/python scripts/run_language_prototype.py \
  --output runs/language/prototype-explanation.json explain \
  --result examples/language/cnn-result-envelope.json

./.venv/bin/python scripts/run_language_prototype.py \
  --output runs/language/prototype-plan.json plan \
  --request-file examples/language/planner-request.txt
```
- LoRA is not authorized by this phase. First pass the broader frozen
  adversarial suite. Only repeatable semantic failures—not schema, transport,
  leakage, authorization, or evidence failures—can justify a later human-reviewed
  fine-tuning proposal. Medical-literature fine-tuning remains out of scope.
