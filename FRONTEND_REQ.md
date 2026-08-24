---
title: Brain MRI research product frontend requirements
meta:
  contentType: Requirements
  status: Draft for implementation
  lastUpdated: 2026-08-23
---

# Brain MRI research product frontend requirements

## Decision

Frontend work should begin now while the AMD confirmation campaign trains. The
current evidence is sufficient to stop broad CNN architecture exploration and
build an internal research prototype around stable, versioned contracts. It is
not sufficient to call the CNN clinically validated or externally generalizable.

The current training queue is the final planned confirmation campaign. After it
finishes, evaluate the frozen baseline and nesting-penalty runs across seeds and
promote neither unless the preregistered evidence package supports it. Start more
CNN training only when that analysis identifies a specific, testable failure.

## Product boundary

### Goal

Give the project owner one local, read-only place to answer four questions:

1. Are the compute workers and training queues healthy?
2. What evidence supports the current model candidate?
3. What does the deterministic and constrained-language layer say about that
   evidence?
4. What job, if any, may be proposed for human approval next?

The frontend is a research operations and evidence-review tool. It is not a
diagnostic viewer, radiology workstation, treatment tool, patient portal, or
autonomous training controller.

### Primary user

The repository owner, using the current Mac over an intermittent Tailscale
connection to inspect the private AMD and NVIDIA workers. The user is technical
and needs exact status, metrics, provenance, and failure context more than
marketing language.

### Release labels

- **Internal research prototype**: may be built and used now with fixtures and
  sanitized artifacts.
- **CNN candidate selected**: requires the completed frozen confirmation queue,
  paired multi-seed analysis, and an explicit review record.
- **Externally evaluated research model**: additionally requires a locked,
  independent adult-glioma test source and overlap/provenance audit.
- **Clinical use**: out of scope. The interface must never imply this status.

## Scientific truth the UI must preserve

- Current inputs are four co-registered MRI channels: T1, T1ce, T2, and FLAIR.
- Current model development is for adult glioma only. Pediatric glioma,
  meningioma, metastasis, and other tumor types are unsupported.
- Current reported training results are internal BraTS development/validation
  evidence, not independent external-test evidence.
- Mean Dice alone cannot select a winner. Review mean regional Dice,
  lowest-quartile whole-tumor Dice, HD95, derived box IoU, paired per-case
  differences with confidence intervals, seed stability, and worst cases.
- Pseudo-Dice shown during nnU-Net training is trajectory telemetry, not final
  validation performance.
- Automatic model promotion remains disabled.

Every evidence view must show its evaluation scope and freshness near the
headline metric, not behind a tooltip.

## Information architecture

Use four top-level views. On a small screen they may become a compact menu, but
the order remains the same.

1. **Overview** — worker health, active work, queue progress, disk headroom, and
   alerts.
2. **Model evidence** — frozen run comparison, metric distributions, paired
   results, provenance, and promotion-gate state.
3. **Explanations** — canonical deterministic explanation beside an optional,
   validated LLM rendering and its exact evidence fields.
4. **Proposals** — read-only job availability and non-executing orchestrator
   proposals awaiting human review.

The existing `monitor/` overview is the visual and behavioral starting point.
Do not replace it with a separate dashboard aesthetic.

## Functional requirements

### FR-1: Worker and training overview

The overview must:

- show AMD and NVIDIA reachability, collection latency, last successful update,
  stale-since time, and the last known values during an outage;
- show GPU utilization, VRAM, RAM, disk free space, active session, active run,
  epoch/batch/case progress, phase, live loss, and queue state when available;
- distinguish `training`, `reporting`, `stale`, `attention`, and `unavailable`
  with both words and color;
- show recent runs with host, run ID, seed, best epoch, Dice, box IoU, HD95, and
  status;
- identify pseudo metrics as live estimates and final validation metrics as
  completed results;
- expose no start, stop, pause, resume, reboot, WSL, shell, delete, or queue
  mutation control.

The frontend consumes the existing `DashboardSnapshot` contract in
`monitor/src/types.ts`. Schema changes require a version bump and an explicit
unknown-version error state.

### FR-2: Model evidence

The evidence view must support a baseline plus at least one candidate and three
seeds per arm. It must show:

- study ID, protocol, architecture/trainer, dataset/split label, seed, best
  epoch, code revision, and checkpoint/study/profile hashes;
- mean Dice by required tumor region when present;
- overall mean Dice, lowest-quartile mean Dice, mean HD95 in millimetres, and
  mean derived box IoU;
- candidate-minus-baseline deltas with directionality made explicit because
  lower HD95 is better while higher overlap scores are better;
- paired per-case estimates and confidence intervals when the final evaluation
  produces them;
- seed-level points rather than only an aggregate mean;
- preregistered gate results, missing evidence, and the final review decision;
- a clear `internal validation` or `independent test` scope label on every
  comparison.

Do not render a winner badge from a single metric, one seed, or a live training
estimate. When required evidence is incomplete, show `Promotion blocked` and
list the missing items.

### FR-3: Explanation comparison

The canonical facts come from a validated, sanitized
`ResearchRunSummaryEnvelopeV1`. The frontend must render them deterministically
even when the LLM service is offline.

For each explanation, show:

- the required disclaimer: “Research output only; not a diagnosis or treatment
  recommendation.”;
- deterministic summary and limitations;
- optional validated LLM summary and limitations;
- whether the LLM abstained;
- every evidence item as a field/value pair traceable to the source envelope;
- schema version, export ID, artifact hash, model name/digest when applicable,
  and generation time;
- validation status and fallback reason.

The LLM is a constrained renderer, not the source of truth. If its output fails
schema, evidence, safety, or provenance validation, the UI must omit its prose,
retain the deterministic rendering, and state `LLM explanation unavailable —
deterministic facts shown`. Never silently repair model output in the browser.

### FR-4: Orchestrator proposals

The proposal view consumes `JobStatusEnvelopeV1` and `JobProposalV1`. It must:

- show every frozen job's run ID, profile, state, reason code, and whether a
  proposal is allowed;
- allow display of only an exact, pre-approved `(run_id, profile)` match;
- display abstention and its reason for ambiguous, unavailable, unsafe, or
  unmatched requests;
- display `executed: false` prominently on every proposal;
- label the action as a proposal for human review, never as a started job;
- record an approval/rejection decision only in a local review record if that
  feature is later enabled.

Version 1 must not execute a proposal. A future execution path requires a
separate threat model, authenticated controller, confirmation design, audit log,
and explicit user authorization; it is not implied by an approval button.

### FR-5: Provenance and review record

Every result detail view must expose a compact provenance panel containing all
available immutable identifiers and hashes. Long hashes may be visually
truncated, but copying must return the full value.

Human review records, if added, must include reviewer, timestamp, artifact
hashes, decision, and optional structured note. They must not rewrite the source
run, envelope, explanation, or proposal artifacts.

## Data and privacy contract

### Allowed in the product frontend

- aggregate run metrics and preregistered gates;
- synthetic fixture data;
- random export UUIDs and safe run identifiers;
- code, study, profile, checkpoint, matrix, source-summary, and artifact hashes;
- worker telemetry and non-sensitive queue/session labels;
- allowlisted evidence field/value pairs from validated language artifacts.

### Forbidden in the product frontend

- MRI volumes, masks, derived slices, thumbnails, or embeddings in version 1;
- native subject or patient IDs, names, dates, MRNs, DICOM fields, filenames,
  local paths, host credentials, or free clinical text;
- case-level artifacts crossing from the CNN worker to the language-model host;
- diagnosis, prognosis, treatment, medication, or patient-specific claims;
- raw or unvalidated LLM output.

Describe the language artifact as a **sanitized/direct-identifier-free research
envelope**. Do not call it anonymous, de-identified, HIPAA-compliant, or safe
merely because it passed the local allowlist.

## Data contracts

The frontend must treat these checked-in Pydantic models as authoritative:

- `ResearchRunSummaryEnvelopeV1`
- `RunSummaryExplanationV1`
- `JobStatusEnvelopeV1`
- `JobProposalV1`

Unknown fields, duplicate JSON keys, non-finite numbers, invalid hashes,
unexpected schema versions, and inconsistent deterministic gates must be
rejected by the server before the browser receives data. The browser must still
handle a server-reported contract error without crashing or inventing values.

All frontend API responses need:

- `schemaVersion`;
- `generatedAt`;
- source freshness and collection status;
- an artifact or snapshot digest where the source is immutable;
- explicit `null` for unavailable measurements rather than fabricated zeroes.

## Network and backend requirements

- Bind the product gateway to `127.0.0.1` by default.
- Keep SSH targets, usernames, paths, and commands fixed and server-side.
- Never interpolate browser input into a shell command, remote path, or host.
- Collect independent hosts concurrently with hard timeouts and bounded retry
  backoff.
- Reuse multiplexed SSH connections and compression where they improve unstable
  Tailscale links.
- Preserve the last good snapshot across transient failures and expose its age.
- Support ETag/`If-None-Match`; unchanged polls should return `304` with no body.
- Poll active training around every 10 seconds and back off to at most once per
  minute during repeated failure. Exact intervals remain server-owned.
- Do not use browser-to-worker SSH, WebSockets that require persistent wide-area
  connectivity, external analytics, third-party fonts, or CDN assets.

Server-sent events may be considered later, but polling plus ETags is the
version-1 reliability baseline.

## Interaction and state requirements

Every view must define visible states for:

- initial loading;
- no data yet;
- fresh data;
- stale last-known data;
- partial host failure;
- complete outage;
- malformed or unsupported artifact;
- training running, validating, complete, failed, and idle;
- explanation valid, abstained, rejected, and unavailable;
- proposal exact match, abstained, unavailable, unsafe, and already complete;
- promotion pending, blocked, rejected, and selected for further research.

Errors remain close to the affected data. A gateway-wide banner is appropriate
only when the local gateway itself is failing. The interface must never replace
stale values with a success-looking empty state.

## Content requirements

Use direct research language:

- `Internal validation Dice`, not `Accuracy` or `Model confidence`.
- `Research segmentation model`, not `Tumor detector`.
- `Candidate selected for further research`, not `Approved model`.
- `Proposal awaiting human review`, not `AI action`.
- `Worker unavailable`, not `Something went wrong`.
- `No metric recorded`, not a displayed zero.

The persistent product notice is: `Internal research prototype. Not for
diagnosis, treatment, or clinical use.`

Avoid chat bubbles, simulated assistant personality, promotional claims, and
anthropomorphic wording. Explanations should read like an evidence memo.

## Visual and responsive direction

Extend the existing **Research Scheduler Docket** in `monitor/DESIGN.md`:

- cool paper surfaces, graphite text, thin rules, and sparse semantic accents;
- system sans typography and tabular numerals;
- flat sections and ruled tables rather than a mosaic of floating cards;
- no gradients, glows, decorative AI imagery, glass effects, or ambient motion;
- color paired with text and symbols for every status;
- desktop-first information density that remains usable at 320 CSS pixels;
- horizontally scrollable comparison tables with the key identity column kept
  understandable on narrow screens;
- dark mode, increased contrast, reduced motion, and reduced transparency.

Motion is limited to short state feedback such as progress changes and tab
selection. Charts must not auto-animate.

## Accessibility requirements

- Meet WCAG 2.2 AA contrast and interaction requirements.
- Use semantic landmarks, headings, tables, buttons, and progress elements.
- Make all controls keyboard operable with persistent visible focus.
- Do not communicate status or metric direction by color alone.
- Provide text alternatives or tabular equivalents for charts.
- Preserve meaningful reading order when layouts collapse.
- Respect `prefers-reduced-motion`, `prefers-contrast`, and
  `prefers-reduced-transparency` where supported.

## Performance requirements

- The overview must remain useful on unstable Wi-Fi and after either worker
  becomes unreachable.
- Keep the initial application shell small; target no more than 150 KiB of
  compressed first-party JavaScript for version 1.
- Prefer CSS, semantic HTML, and small SVG primitives over a large component or
  chart library. Reassess the current chart dependency before expanding it.
- Lazy-load detailed evidence/provenance views.
- Avoid images and external font downloads.
- Render the last cached snapshot immediately, then refresh in the background.
- Keep user interactions responsive while parsing the largest allowed 256 KiB
  language envelope.

## Implementation phases

### Phase 1 — stabilize the existing overview

- Preserve the current read-only monitor and its fixture mode.
- Add explicit research-use, metric-scope, and pseudo-metric labels.
- Harden contract-version, malformed-data, and stale-cache states.
- Add accessible table equivalents for charted values.

This phase can proceed immediately and must not touch either GPU workload.

### Phase 2 — evidence workspace

- Add frozen JSON fixtures for baseline/candidate multi-seed comparisons.
- Implement model evidence and provenance views.
- Add gate completeness and `Promotion blocked` logic.
- Integrate final paired metrics only after the analysis contract is frozen.

### Phase 3 — deterministic plus LLM explanations

- Render validated envelopes deterministically first.
- Add the independently validated LLM artifact as an optional comparison.
- Show field-level evidence and exact fallback reasons.
- Test every known adversarial and stale/malformed state.

### Phase 4 — proposal review

- Display validated job availability and non-executing proposals.
- Add local immutable human review records if needed.
- Keep all remote execution out of scope.

## Acceptance criteria for the first usable product

- The owner can identify the active host, run, progress, freshness, and disk
  headroom in under ten seconds.
- Disconnecting either host preserves and clearly marks its last-known values.
- The app cannot mutate a worker through any browser request.
- Fixture mode demonstrates fresh, stale, failed, empty, and completed states
  without network access.
- A completed candidate cannot show as promoted when any required multi-seed,
  paired, confidence-interval, provenance, or external-scope field is missing.
- Deterministic explanations render without Ollama.
- Invalid LLM output never appears as accepted prose.
- Every orchestrator artifact displays `executed: false`, and no frontend route
  can start a job.
- Automated tests cover contract rejection, stale caching, shell-input
  isolation, safety wording, and the major UI states.
- Keyboard-only and narrow-screen reviews pass without hidden evidence.

## Model-readiness exit criteria

Frontend completion and model completion are separate. The current CNN training
campaign is considered done enough to stop routine retraining when:

1. all frozen baseline and nesting confirmations finish or have a documented,
   scientifically justified exclusion;
2. immutable artifacts and hashes are complete;
3. the paired multi-seed evidence package reports every required metric and
   confidence interval;
4. worst-case and failure-cluster review is recorded;
5. the selection decision is explicit and does not rely on mean Dice alone.

External generalization remains unproven until a locked independent adult-
glioma test is evaluated. That is a later study gate, not a blocker for the
internal product prototype.

## Source of truth

These requirements consolidate the current repository contracts and plans:

- `monitor/PRODUCT.md` and `monitor/DESIGN.md`
- `monitor/src/types.ts`
- `docs/architecture.md`
- `docs/phases/04-language-layer.md`
- `src/brain_mri_data/language_contracts.py`
- `analyses/language/language-pipeline-review-20260816.md`
- `HANDOFF.md`

When these sources conflict, use the newest explicit user instruction for host
placement and operational state, the strict Pydantic models for language data,
and the frozen study artifact for scientific claims. Do not infer a clinical or
execution capability from a frontend requirement.
