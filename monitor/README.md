# Brain MRI research workspace

For the current deployed prototype, start with the
[operating guide](../docs/prototype-operations.md). The app is hosted on `.7`,
proxied through `.1:4173`, and uses `qwen3:4b`. The three visible tabs are
Try it, System status, and Model results; the resource endpoints below also
preserve historical research evidence.

A private, local workflow for validating four-volume adult-glioma MRI studies,
running one fixed CNN, and returning a research segmentation with exact
provenance. It binds either to loopback or the NVIDIA worker's exact Tailscale
address; uploaded image data remains on that worker and is never supplied to
the language model.

The workspace includes these product and historical evidence surfaces:

1. **New study** — select T1, T1ce, T2, and FLAIR NIfTI volumes, validate their
   geometry, run the fixed CNN, and download the segmentation, receipt, and
   explanation.
2. **Operations** — worker health, active training, queues, telemetry, and recent
   internal-validation results.
3. **Model evidence** — the completed three-seed baseline/candidate comparison,
   paired uncertainty, rejected-candidate decision, retained internal model,
   and immutable provenance.
4. **Explanations** — deterministic facts, optional validated LLM rendering,
   fallback reasons, evidence fields, and provenance.
5. **Proposals** — frozen job availability and non-executing proposals for
   human review. Every proposal remains `executed: false`.

## Start

Run the product on the NVIDIA worker. Prerequisites are Node.js 24, the pinned
CUDA Python environment, and this exact checkpoint:

```text
runs/glioma-pilot--cuda-4060--brats--20260828--e100/best.pt
SHA-256 121422a861bbe7affaa5e161058e69eea737b2390651c3c03ea20256969e99e5
```

```bash
SETUP_CUDA=1 ./setup.sh
scripts/install_node_runtime.sh
PATH="$PWD/.tools/node/bin:$PATH" npm --prefix monitor ci
MONITOR_BIND_HOST=100.64.0.7 MONITOR_PUBLIC_HOSTS=100.64.0.1,100.64.0.7 BRAIN_MRI_LLM_MODEL=qwen3:4b scripts/start_research_workspace.sh
```

The Node installer is Linux x86-64 only, downloads the pinned official archive,
and verifies its SHA-256 before extracting it under `.tools/` in the repository.

From any authenticated device on the same tailnet, open:

<http://100.64.0.1:4173>

The server rejects non-Tailscale clients and invalid Host/Origin headers. Omit
`MONITOR_BIND_HOST` to retain loopback-only access. The launcher verifies the
checkpoint before starting; the gateway also verifies it before every inference.

To enable the optional local metadata renderer, install an Ollama model that
fits the worker and provide its exact tag. The resolved model digest is recorded
with each accepted explanation:

```bash
ollama pull qwen3:4b
BRAIN_MRI_LLM_MODEL=qwen3:4b scripts/start_research_workspace.sh
```

Without that variable, the product still returns the deterministic validated
explanation and clearly marks the LLM rendering unavailable. LLM prose is shown
only when its exact evidence, limitations, safety fields, and local model digest
pass validation; otherwise the deterministic explanation remains authoritative.

## Study data lifecycle

- Uploads accept exactly one `.nii.gz` file for each named modality, up to
  512 MiB each and 2 GiB total.
- Files stream to a private server-generated directory; browser filenames never
  become inference paths or language-model content.
- Geometry validation requires matching 3D shape and affine, finite voxels,
  axes no larger than 512, and no more than 64 million voxels per volume.
- One GPU inference may run at a time. The process has a 30-minute timeout.
- Original uploaded volumes are deleted after validation failure or inference completion.
  Sanitized viewing copies and result artifacts expire after 24 hours and may be
  cleared immediately. Browser-specific history stores access tokens, not scan bytes.
- Without an optional expert reference mask the app cannot report accuracy.
  With a matching reference it reports single-case overlap and boundary metrics.
  Neither flow produces diagnosis, prognosis, or treatment conclusions.

## Operations network behavior

- Hosts are collected concurrently every 10 seconds.
- SSH uses compression and a persistent multiplexed connection under a
  mode-700, user-specific `/tmp/brain-mri-monitor-<uid>/` directory, reducing
  repeated handshakes over unstable Wi-Fi while staying below macOS socket
  path limits.
- Each attempt has a hard timeout. Failures back off up to 60 seconds.
- Last-good values stay visible with their confirmation age and error.
- Browser requests use ETags, so unchanged local snapshots return no body.

Override a fixed target only from the server environment:

```bash
MONITOR_NVIDIA_TARGET=software@100.64.0.1 npm start
```

Targets must remain `user@host` values. There are no browser-provided host,
path, or command parameters.

## Finalized local resources

By default, `/api/evidence`, `/api/explanation`, and `/api/proposals` read the
strict, checked-in resources under `monitor/resources/`. They report the
completed confirmation decision: the nesting candidate is rejected, baseline
seed `20260821` is retained for internal research, and no additional training
is proposed. Duplicate JSON keys, non-finite values, invalid hashes, unsafe
promotion states, and executing proposals are rejected by the gateway.

## Offline UI fixture

```bash
MONITOR_FIXTURE=1 npm start
```

This serves representative non-identifying worker telemetry without contacting
either worker while retaining the finalized local evidence resources.

Set `MONITOR_FIXTURE_SCENARIO` to exercise contract-valid failure states:

```bash
MONITOR_FIXTURE=1 MONITOR_FIXTURE_SCENARIO=stale npm start
MONITOR_FIXTURE=1 MONITOR_FIXTURE_SCENARIO=fresh npm start
MONITOR_FIXTURE=1 MONITOR_FIXTURE_SCENARIO=unavailable npm start
MONITOR_FIXTURE=1 MONITOR_FIXTURE_SCENARIO=rejected npm start
MONITOR_FIXTURE=1 MONITOR_FIXTURE_SCENARIO=unsafe npm start
MONITOR_FIXTURE=1 MONITOR_FIXTURE_SCENARIO=already-complete npm start
```

Scenarios are allowlisted fixture filenames; browser input never becomes a
host, path, command, or remote action. A scenario that does not exist is
returned as a safe rejected contract state.

Run `npm test` for gateway safety and contract tests, component-state tests,
TypeScript validation, and the production build. The detailed views are
lazy-loaded, and the dependency-free SVG telemetry chart keeps the compressed
first-party JavaScript comfortably below the 150 KiB version-1 target.
