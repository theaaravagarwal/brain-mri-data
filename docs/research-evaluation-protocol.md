# Research evaluation protocol

Status: proposed studies, not completed experiments. Freeze a dated copy with
the selected tasks, sample, conditions, and analysis before collecting results.
Use the existing [cohort checks and private reviewer worksheet](independent-validation-review.md).

## 1. Review segmentation failures

Use all six recorded weak cases for descriptive review; do not present this
selected group as a representative cohort. For a new evaluation, freeze the
entire eligible manifest before inference and report exclusions and failures.

Ask a qualified reviewer to inspect MRI and overlays before revealing numerical
scores. Record missed regions, extra regions, boundary problems, reference
ambiguity, and review time. Freeze that assessment before revealing Dice/HD95.
Call this score-blinded review; model/reference labels are visible in the current
viewer, so it is not a fully blinded model comparison. The current page also
displays scores below the viewer: use a controlled viewing session that conceals
them, or record accidental exposure as a protocol deviation.

Use nonexclusive error labels: `empty_output`, `missed_small_region`,
`extra_region`, `boundary_disagreement`, `input_or_registration_problem`,
`reference_ambiguity`, and `unassessed`. Define any quantitative small-region
threshold in mm³ from development data before a new cohort is opened. An empty
prediction is observable; its cause requires investigation. Do not infer that
the reference is wrong merely because the model disagrees with it.

## 2. Test whether the workspace helps users

Compare the current workspace with a documented file-based workflow using the
same checkpoint outputs, scan viewer, reference information, and receipts.
Supply equivalent information in both conditions; record tool versions.

For an initial formative pilot, recruit 5–8 intended research users and report
their relevant experience. This sample is for finding usability problems, not
for claiming population-level superiority. Use matched case sets and alternate
condition order so participants do not answer the same remembered case twice.

Tasks: locate checkpoint identity; find a discrepancy on the scan; identify
whether a score has a reference; explain an empty-output warning; download the
correct result package. Measure correct completion, time, assistance, and a
brief difficulty rating. Prepare the answer key before sessions. Report paired
differences, failures, and individual observations; do not hide abandoned tasks.
Qualified imaging judgments and interface tasks require different expertise.

## 3. Measure the LLM's added value

Hold metadata and UI constant and compare deterministic text with the constrained
LLM rendering. Use separate or counterbalanced case sets. Ask users to identify
what was measured, whether accuracy was assessed, and what remains unknown.
Record comprehension and time; preference alone does not establish correctness.

For factual fidelity, freeze a metadata fixture set spanning reference present,
reference absent, empty/tiny output, unavailable boundary metric, and renderer
failure. Add adversarial text only in fields the actual schema accepts.
Run three generations per supported fixture, preserving model digest, prompt,
configuration, input hash, raw response, validator verdict, and displayed text.

Two reviewers independently mark each displayed factual assertion as supported,
contradicted, or absent from metadata, then reconcile disagreements. Report
unsupported assertions per assertion and responses with any unsupported claim
per response; report rejected generations separately. Rejection is not evidence
that the raw generation was factual. Zero observed violations applies only to
the tested set. Do not treat a claim validator's own verdict as independent proof.

## 4. Measure performance and recovery

Use permitted research cases and record host/GPU, code/checkpoint, input size,
dependency versions, and concurrent load. Measure upload, validation, inference,
explanation, first usable viewer frame, and complete result latency separately.
Run one explicitly cold request and at least five warm requests per selected
case for a formative benchmark. Report sample counts, median, range, and p95
with a warning about small-sample instability. Measure process/GPU peak memory;
GPU utilization screenshots alone do not measure efficiency.

Keep repeated timings separate from independent-case model metrics. GPU busy,
invalid input, expired results, and interrupted inference are distinct outcomes.
The next operating drill is an authenticated `.7` reboot with recovery checks,
then `.1` proxy reboot. Save actual outcomes; existing service restart checks
cannot substitute for this drill. Checkpoint replacement needs a separate
approved artifact and model-contract change before a rollback demonstration.

## 5. Preserve and publish evidence

Keep native identifiers, case tokens, access tokens, screenshots, and reviewer
identities in private review storage. A randomly assigned publication row ID is
pseudonymous, not proof of anonymization. The current public export policy is
aggregate-only; any per-case release needs a separate disclosure/permission
review. Publish aggregate failures and metric denominators in the meantime.

Private evaluation-table columns:

```text
study_id,cohort_manifest_sha256,checkpoint_sha256,app_commit,case_row_id,
validation_status,dice,iou,precision,recall,hd95_mm,hd95_available,
predicted_volume_mm3,reference_volume_mm3,error_labels,review_status,
upload_seconds,validation_seconds,inference_seconds,explanation_seconds,
viewer_seconds,total_seconds,peak_vram_mib,protocol_deviation
```

Missing values remain missing with a reason; unavailable HD95 is not zero.
Keep the identity mapping outside the analysis export. Each experiment log
should include question, frozen protocol/hash, data/hash, code/model versions,
change, findings, failures, decision, and next experiment.

## Completion criteria

The documentation package is ready when claims point to artifacts and unresolved
fields are explicit. Research hypotheses remain open until their studies finish.
Prioritize reviewer access and the provenance record, then the user/explanation
pilot and runtime study. Add an in-app reviewer form only after the worksheet
has been exercised; no reviewer form or new experiment is implemented by this
documentation change. Further training depends on a demonstrated development
failure and a declared experiment.
