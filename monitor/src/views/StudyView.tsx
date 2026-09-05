import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { studyFetch, remember, history, forget, download } from "../study-access";
const MriViewer = lazy(() => import("./MriViewer"));
import type { StudyCapabilities, StudyJob } from "../types";

type Modality = "t1" | "t1ce" | "t2" | "flair";
type Files = Record<Modality, File | null>;

const modalityDetails: ReadonlyArray<{ key: Modality; label: string; detail: string }> = [
  { key: "t1", label: "T1", detail: "T1 scan" },
  { key: "t1ce", label: "T1ce", detail: "T1 scan with contrast" },
  { key: "t2", label: "T2", detail: "T2 scan" },
  { key: "flair", label: "FLAIR", detail: "FLAIR scan" }
];

const emptyFiles = (): Files => ({ t1: null, t1ce: null, t2: null, flair: null });

function shortHash(value: string | null | undefined) {
  return value ? `${value.slice(0, 12)}…${value.slice(-8)}` : "Unavailable";
}

function formatBytes(value: number) {
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GiB`;
  return `${(value / 1024 ** 2).toFixed(1)} MiB`;
}

async function jsonResponse<T>(response: Response): Promise<T> {
  const value = await response.json().catch(() => ({})) as Record<string, unknown>;
  if (!response.ok) throw new Error(String(value.detail || value.error || `Local gateway returned ${response.status}`));
  return value as T;
}

function StatusMark({ state, label = state }: { state: "ready" | "waiting" | "complete" | "failed"; label?: string }) {
  return <span className={`study-status study-status--${state}`}><span aria-hidden="true" />{label}</span>;
}

export default function StudyView() {
  const [capabilities, setCapabilities] = useState<StudyCapabilities | null>(null);
  const [files, setFiles] = useState<Files>(emptyFiles);
  const [reference, setReference] = useState<File | null>(null);
  const [job, setJob] = useState<StudyJob | null>(null);
  const [busy, setBusy] = useState<"demo" | "upload" | "run" | "clear" | null>(null);
  const [usingDemo, setUsingDemo] = useState(false);
  const [checkingCapabilities, setCheckingCapabilities] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recent, setRecent] = useState(history);
  const [health, setHealth] = useState<{ checkedAt: string; diskFreeBytes: number; gpu: { freeBytes: number; totalBytes: number } | null; ollama: boolean; checkpoint: string } | null>(null);
  const [showHealth, setShowHealth] = useState(false);
  useEffect(() => {
    if (!showHealth) return;
    let disposed = false;
    const check = () => { void fetch("/api/studies/health").then(jsonResponse<typeof health>).then(value => { if (!disposed) setHealth(value); }).catch(() => { if (!disposed) setHealth(null); }); };
    check(); const timer = window.setInterval(check, 30000);
    return () => { disposed = true; clearInterval(timer); };
  }, [showHealth]);
  useEffect(() => {
    if (job) remember(job);
    setRecent(history());
  }, [job]);
  useEffect(() => {
    const restore = history()[0];
    if (restore) void studyFetch(`/api/studies/${restore.jobId}`).then(jsonResponse<StudyJob>).then(setJob).catch(() => { forget(restore.jobId); setRecent(history()); });
    const sync = () => setRecent(history());
    window.addEventListener("storage", sync);
    return () => window.removeEventListener("storage", sync);
  }, []);

  const checkCapabilities = () => {
    const controller = new AbortController();
    setError(null);
    setCheckingCapabilities(true);
    const request = fetch("/api/capabilities", { signal: controller.signal })
      .then(jsonResponse<StudyCapabilities>)
      .then(setCapabilities)
      .catch(reason => {
        if (!(reason instanceof DOMException && reason.name === "AbortError")) setError(String(reason));
      })
      .finally(() => setCheckingCapabilities(false));
    return { controller, request };
  };

  useEffect(() => {
    const { controller } = checkCapabilities();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!job || job.state !== "running") return;
    let cancelled = false;
    const poll = window.setInterval(() => {
      void studyFetch(`/api/studies/${job.jobId}`)
        .then(jsonResponse<StudyJob>)
        .then(value => { if (!cancelled) setJob(value); })
        .catch(reason => { if (!cancelled) setError(String(reason)); });
    }, 2_000);
    return () => { cancelled = true; clearInterval(poll); };
  }, [job]);

  const allSelected = useMemo(() => modalityDetails.every(({ key }) => files[key]), [files]);
  const totalBytes = useMemo(() => modalityDetails.reduce((total, { key }) => total + (files[key]?.size || 0), reference?.size || 0), [files, reference]);
  const modelReady = capabilities?.inference.status === "ready";

  const selectFile = (modality: Modality, file: File | null) => {
    setFiles(current => ({ ...current, [modality]: file }));
    setJob(null);
    setUsingDemo(false);
    setError(null);
  };

  const loadDemo = async (evaluation = false) => {
    setBusy("demo");
    setError(null);
    try {
      const response = await fetch(evaluation ? "/api/studies/demo-evaluation" : "/api/studies/demo", { method: "POST" });
      setJob(await jsonResponse<StudyJob>(response));
      setFiles(emptyFiles());
      setReference(null);
      setUsingDemo(true);
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(null); }
  };

  const validateStudy = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!allSelected) return setError("Choose one NIfTI file for every scan type.");
    setBusy("upload");
    setError(null);
    const body = new FormData();
    for (const { key } of modalityDetails) body.append(key, files[key]!);
    if (reference) body.append("reference", reference);
    try {
      const response = await fetch("/api/studies", { method: "POST", body });
      setJob(await jsonResponse<StudyJob>(response));
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(null); }
  };

  const runInference = async () => {
    if (!job) return;
    setBusy("run");
    setError(null);
    try {
      const response = await studyFetch(`/api/studies/${job.jobId}/inference`, { method: "POST" });
      setJob(await jsonResponse<StudyJob>(response));
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(null); }
  };

  const clearStudy = async () => {
    setBusy("clear");
    setError(null);
    try {
      if (job) {
        const response = await studyFetch(`/api/studies/${job.jobId}`, { method: "DELETE" });
        if (!response.ok) await jsonResponse(response);
        forget(job.jobId);
      }
      setJob(null);
      setFiles(emptyFiles());
      setReference(null);
      setUsingDemo(false);
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(null); }
  };

  const clearHistory = async () => {
    setBusy("clear"); setError(null);
    const failures = [];
    for (const row of history()) {
      const response = await studyFetch(`/api/studies/${row.jobId}`, { method: "DELETE" });
      if (response.ok || response.status === 404) { forget(row.jobId); if (job?.jobId === row.jobId) setJob(null); }
      else failures.push(row.jobId.slice(0, 8));
    }
    setRecent(history()); setBusy(null);
    if (failures.length) setError(`Could not clear ${failures.join(", ")}. Running studies must finish first.`);
  };

  const llm = job?.explanation?.llm;
  const shownExplanation = llm?.status === "validated" && llm.artifact ? llm.artifact : job?.explanation?.deterministic;
  const evaluation = job?.result?.evaluation;
  const emptyOutline = job?.result?.segmentation.nonzero_voxels === 0;
  const smallOutline = Boolean(job?.result && job.result.segmentation.nonzero_voxels > 0 && job.result.segmentation.nonzero_voxels < 1000);
  const outputNeedsReview = emptyOutline || smallOutline;
  const benchmark = capabilities?.externalBenchmark;
  const benchmarkDice = benchmark?.metrics?.whole_lesion_dice;
  const benchmarkHd95 = benchmark?.metrics?.hd95_mm;

  return <div className="view study-view" id="view-study">
    <details onToggle={event => setShowHealth(event.currentTarget.open)}><summary>Application health</summary>{health ? <p>Checked {new Date(health.checkedAt).toLocaleTimeString()} · Server disk free: {formatBytes(health.diskFreeBytes)} · GPU free: {health.gpu ? formatBytes(health.gpu.freeBytes) : "unavailable"} · Model: {health.checkpoint} · Local explainer: {health.ollama ? "reachable" : "unavailable"} · Gateway reachable</p> : <p>Checking application health…</p>}</details>
    {recent.length > 0 && <button disabled={busy !== null} onClick={() => { void clearHistory().catch(reason => { setError(String(reason)); setBusy(null); }); }}>Clear this browser’s studies</button>}
    {recent.length > 0 && <details className="study-history"><summary>Recent studies in this browser ({recent.length})</summary>{recent.map(row => <div key={row.jobId}><button onClick={() => { void studyFetch(`/api/studies/${row.jobId}`).then(jsonResponse<StudyJob>).then(setJob).catch(reason => setError(String(reason))); }}>Study {row.jobId.slice(0, 8)} · {row.state}</button><span>{row.modelId || "Frozen model"} · Created {new Date(row.createdAt).toLocaleString()} · Expires {new Date(row.expiresAt).toLocaleString()}</span></div>)}</details>}
    <div className="study-intro">
      <div>
        <h1>Try the MRI outline tool</h1>
        <p>Choose four scan files to create an outline. Add an expert outline to measure accuracy on one case.</p>
      </div>
      <div className="model-readiness" aria-live="polite">
        <StatusMark state={!capabilities && error ? "failed" : !capabilities ? "waiting" : modelReady ? "ready" : "failed"} label={!capabilities ? "checking" : modelReady ? "ready" : "not ready"} />
        <strong>{modelReady ? "Ready to run" : error ? "Connection check failed" : "Checking the model"}</strong>
        <span>{capabilities ? "Runs privately on the research server" : "One moment"}</span>
      </div>
    </div>

    <ol className="study-steps" aria-label="Study workflow progress">
      <li aria-current={!job ? "step" : undefined}><span>1</span><strong>Choose</strong></li>
      <li aria-current={job?.state === "validated" ? "step" : undefined}><span>2</span><strong>Check</strong></li>
      <li aria-current={job?.state === "running" ? "step" : undefined}><span>3</span><strong>Process</strong></li>
      <li aria-current={job?.state === "succeeded" ? "step" : undefined}><span>4</span><strong>Download</strong></li>
    </ol>

    {benchmark ? <section className="external-benchmark" aria-labelledby="external-benchmark-title">
      <div>
        <span className="eyebrow">Separate public test set</span>
        <h2 id="external-benchmark-title">{benchmark.status === "complete" ? `Tested on ${benchmark.case_count} unseen cases` : `Testing ${benchmark.completed_cases}/${benchmark.total_cases} cases`}</h2>
        <p>{benchmark.status === "complete" ? "The fixed model was tested once on data kept out of training and tuning." : "The fixed model is running through the full outside dataset now."}</p>
        {benchmark.status === "complete" ? <a className="external-report-link" href="/api/external-benchmark/report" download>Download validation report</a> : null}
      </div>
      {benchmark.status === "complete" && benchmarkDice && benchmarkHd95 ? <dl>
        <div><dt>Average Dice</dt><dd>{benchmarkDice.mean.toFixed(3)}</dd><small>95% range for the mean: {benchmarkDice.mean_ci95[0].toFixed(3)}–{benchmarkDice.mean_ci95[1].toFixed(3)}</small></div>
        <div><dt>Typical boundary error</dt><dd>{benchmarkHd95.median.toFixed(1)} mm</dd><small>HD95 median</small></div>
        <div><dt>Empty results</dt><dd>{benchmark.failures?.empty_prediction_count ?? 0}</dd><small>out of {benchmark.case_count} cases</small></div>
      </dl> : <progress max={benchmark.total_cases || 1} value={benchmark.completed_cases || 0} aria-label="Outside dataset test progress" />}
    </section> : null}

    {error ? <div className="study-alert" role="alert"><strong>We couldn’t continue.</strong><span>{error}</span>{!capabilities ? <button className="inline-recheck" type="button" disabled={checkingCapabilities} onClick={() => { void checkCapabilities().request; }}>{checkingCapabilities ? "Checking…" : "Try connection again"}</button> : null}</div> : null}
    {!modelReady && capabilities ? <div className="study-alert" role="status"><strong>The model is not ready.</strong><span>Please try again in a moment.</span><button className="inline-recheck" type="button" disabled={checkingCapabilities} onClick={() => { void checkCapabilities().request; }}>{checkingCapabilities ? "Checking…" : "Check again"}</button></div> : null}

    {job?.state === "succeeded" && <button onClick={() => { setJob(null); setFiles(emptyFiles()); setReference(null); setUsingDemo(false); }}>New study</button>}
    {job?.state !== "succeeded" && <div className="study-workspace">
      <form className="modality-form" onSubmit={validateStudy}>
        <div className="section-heading"><div><h2>{usingDemo ? job?.validation?.reference_mask ? "Accuracy sample" : "Sample scans" : "Choose four scan files"}</h2><p>{usingDemo ? job?.evaluationSampleScope === "external_public" ? "A labeled case from a separate public dataset is ready." : job?.validation?.reference_mask ? "A labeled development-validation case is ready." : "This checks the workflow, not accuracy." : "Scans and outlines stay on the research server for up to 24 hours. Clear them sooner from history."}</p></div><span className="file-total">{usingDemo ? "Ready" : allSelected ? `${formatBytes(totalBytes)} selected` : `${Object.values(files).filter(Boolean).length}/4 scans`}</span></div>
        {usingDemo ? <div className="demo-ready"><strong>{job?.validation?.reference_mask ? "Labeled sample loaded" : "Built-in sample loaded"}</strong><span>{job?.evaluationSampleScope === "external_public" ? "This case comes from outside the training dataset. Run it to see one-case accuracy." : job?.validation?.reference_mask ? "The model did not train on this case. Run it to see one-case accuracy." : "You can create the outline now, or clear it and use your own files."}</span></div> : <div className="modality-list">
          {modalityDetails.map(({ key, label, detail }) => <label className="modality-row" key={key}>
            <span className="modality-code">{label}</span>
            <span className="modality-description"><strong>{files[key]?.name || detail}</strong>{files[key] ? <small>{formatBytes(files[key]!.size)} · ready</small> : null}</span>
            <span className={`file-action ${files[key] ? "file-action--selected" : ""}`}>{files[key] ? "Replace" : "Choose file"}</span>
            <input type="file" accept=".nii,.nii.gz,application/gzip,application/x-gzip,application/x-nifti" onChange={event => selectFile(key, event.target.files?.[0] || null)} disabled={busy !== null || job?.state === "running"} />
          </label>)}
          <div className="reference-upload">
            <div><strong>Test accuracy <span>(optional)</span></strong><small>Add the expert outline for this same case.</small></div>
            <label className="reference-row">
              <span className="modality-description"><strong>{reference?.name || "Reference outline"}</strong>{reference ? <small>{formatBytes(reference.size)} · ready</small> : null}</span>
              <span className={`file-action ${reference ? "file-action--selected" : ""}`}>{reference ? "Replace" : "Choose file"}</span>
              <input type="file" accept=".nii,.nii.gz,application/gzip,application/x-gzip,application/x-nifti" onChange={event => { setReference(event.target.files?.[0] || null); setJob(null); setError(null); }} disabled={busy !== null || job?.state === "running"} />
            </label>
            <small>The app compares the model with this outline. The LLM never sees scan or mask pixels.</small>
          </div>
        </div>}
        <div className="study-actions">
          {!usingDemo && !allSelected && capabilities?.evaluationDemoAvailable ? <button className="primary-action" type="button" disabled={!modelReady || busy !== null} onClick={() => loadDemo(true)}>{busy === "demo" ? "Loading sample…" : "Run accuracy sample"}</button> : null}
          {!usingDemo && !allSelected && capabilities?.demoAvailable ? <button className="secondary-action" type="button" disabled={!modelReady || busy !== null} onClick={() => loadDemo(false)}>Try pipeline sample</button> : null}
          {!usingDemo && Object.values(files).some(Boolean) ? <button className="primary-action" type="submit" disabled={!allSelected || !modelReady || busy !== null || job?.state === "running"}>{busy === "upload" ? "Checking files…" : "Check my files"}</button> : null}
          {usingDemo && job?.state !== "running" ? <button className="text-action" type="button" onClick={clearStudy} disabled={busy !== null}>Choose my own files</button> : <span>{usingDemo ? "Public, de-identified research sample" : "Or choose your own files above"}</span>}
        </div>
      </form>

      <aside className="validation-ledger" aria-labelledby="validation-title">
        <div className="section-heading"><div><h2 id="validation-title">File check</h2></div>{job?.validation ? <StatusMark state="complete" label="passed" /> : <StatusMark state="waiting" />}</div>
        {job?.validation ? <dl>
          <div><dt>Files</dt><dd>T1 · T1ce · T2 · FLAIR</dd></div>
          <div><dt>Scan size</dt><dd>{job.validation.shape.join(" × ")} voxels</dd></div>
          <div><dt>Voxel size</dt><dd>{job.validation.spacing_mm.map(value => value.toFixed(2)).join(" × ")} mm</dd></div>
          {job.validation.reference_mask ? <div><dt>Reference outline</dt><dd>Ready · {job.validation.reference_mask.nonzero_voxels.toLocaleString()} marked voxels</dd></div> : null}
          <div><dt>Check ID</dt><dd title={job.validation.geometry_sha256}>{shortHash(job.validation.geometry_sha256)}</dd></div>
        </dl> : <div className="ledger-empty"><strong>Nothing to check yet</strong><span>Use the sample or choose four files.</span></div>}
        {job?.state === "validated" ? <button className="primary-action primary-action--full" type="button" disabled={!modelReady || busy !== null} onClick={runInference}>{busy === "run" ? "Starting…" : job.validation?.reference_mask ? "Run accuracy test" : "Create outline"}</button> : null}
        {job?.state === "running" ? <div className="inference-running" aria-live="polite"><span className="activity-line" /><strong>{job.progress?.stage || "Starting inference"}</strong><span>{Math.max(0, Math.floor((Date.now() - Date.parse(job.progress?.startedAt || job.updatedAt)) / 1000))} seconds in this stage. You can reopen this study from history.</span></div> : null}
        {job?.state === "failed" ? <div className="ledger-failure"><strong>Processing failed</strong><span>{job.error || "The model did not produce an outline."}</span></div> : null}
      </aside>
    </div>}

    {job?.state === "succeeded" && job.result && shownExplanation ? <section className={`study-result ${outputNeedsReview ? "study-result--empty" : ""}`} aria-labelledby="result-title">
      <div className="result-heading"><div><h2 id="result-title">{emptyOutline ? "No outline produced" : smallOutline ? "Very small outline—review carefully" : evaluation ? "Accuracy test ready" : "Research outline ready"}</h2><p role="note">{outputNeedsReview ? "This does not mean the scan is clear—expert review is required." : "For research only—not a medical result."}</p></div><StatusMark state={outputNeedsReview ? "failed" : "complete"} label={outputNeedsReview ? "review" : "complete"} /></div>
      {job.viewing ? <Suspense fallback={<p>Loading viewer…</p>}><MriViewer id={job.jobId} viewing={job.viewing} /></Suspense> : <p>Viewing data is unavailable for this older result. Submit the scans again to inspect an overlay.</p>}
      <div className="result-grid">
        <div className="result-summary">
          {evaluation ? <div className="evaluation-block">
            <p>Dice measures overlap with the expert outline: 1 means exact overlap. HD95 measures boundary distance in millimeters; lower is better. These describe this case only.</p>
            <div className="evaluation-label"><strong>This case only</strong><span>Compared with the expert outline you uploaded</span></div>
            <dl className="evaluation-metrics">
              <div><dt>Dice</dt><dd>{evaluation.whole_lesion_dice.toFixed(3)}</dd></div>
              <div><dt>IoU</dt><dd>{evaluation.whole_lesion_iou.toFixed(3)}</dd></div>
              <div><dt>Precision</dt><dd>{evaluation.precision.toFixed(3)}</dd></div>
              <div><dt>Recall</dt><dd>{evaluation.recall.toFixed(3)}</dd></div>
              <div><dt>HD95</dt><dd>{evaluation.hd95_mm === null ? "N/A" : `${evaluation.hd95_mm.toFixed(1)} mm`}</dd></div>
            </dl>
          </div> : <><strong>{job.result.segmentation.nonzero_voxels.toLocaleString()}</strong><span>voxels included in the outline</span></>}
          <p>{shownExplanation.summary}</p>
          <p className="result-limitations">{shownExplanation.limitations}</p>
          {llm?.status !== "validated" ? <div className="llm-fallback"><strong>Showing the checked facts.</strong><span>The optional plain-language rewrite was not available.</span></div> : <div className="llm-receipt"><strong>Plain-language explanation checked</strong><span>{llm.model_name} · <span title={llm.model_digest || undefined}>{shortHash(llm.model_digest)}</span></span></div>}
        </div>
        <details><summary>Technical details</summary><dl className="receipt-ledger">
          <div><dt>Model</dt><dd>{job.result.provenance.model_id}</dd></div>
          <div><dt>Model file ID</dt><dd title={job.result.provenance.checkpoint_sha256}>{shortHash(job.result.provenance.checkpoint_sha256)}</dd></div>
          <div><dt>Outline file ID</dt><dd title={job.result.segmentation.output_sha256}>{shortHash(job.result.segmentation.output_sha256)}</dd></div>
          <div><dt>Same scan size</dt><dd>Yes · {job.result.segmentation.output_shape.join(" × ")}</dd></div>
          <div><dt>Deleted after</dt><dd>{new Date(job.expiresAt).toLocaleString()}</dd></div>
        </dl></details>
      </div>
      <div className="artifact-actions">
        <button className="primary-action" onClick={() => { void download(`/api/studies/${job.jobId}/package`, "research-result.zip").catch(reason => setError(String(reason))); }}>Download result package</button>
        {(["segmentation", "receipt", "explanation"] as const).map(kind => <button key={kind} className="secondary-action" onClick={() => { void download(`/api/studies/${job.jobId}/artifacts/${kind}`, kind === "segmentation" ? "research_segmentation.nii.gz" : `${kind}.json`).catch(reason => setError(String(reason))); }}>{kind === "segmentation" ? "Download outline" : kind === "receipt" ? "Download technical receipt" : "Download explanation"}</button>)}
        <button className="text-action" type="button" onClick={clearStudy} disabled={busy !== null}>{busy === "clear" ? "Clearing…" : "Clear local result"}</button>
      </div>
    </section> : null}
  </div>;
}
