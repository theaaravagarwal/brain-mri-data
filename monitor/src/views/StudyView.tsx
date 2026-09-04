import { useEffect, useMemo, useState } from "react";
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
  const [job, setJob] = useState<StudyJob | null>(null);
  const [busy, setBusy] = useState<"demo" | "upload" | "run" | "clear" | null>(null);
  const [usingDemo, setUsingDemo] = useState(false);
  const [checkingCapabilities, setCheckingCapabilities] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
      void fetch(`/api/studies/${job.jobId}`)
        .then(jsonResponse<StudyJob>)
        .then(value => { if (!cancelled) setJob(value); })
        .catch(reason => { if (!cancelled) setError(String(reason)); });
    }, 2_000);
    return () => { cancelled = true; clearInterval(poll); };
  }, [job]);

  const allSelected = useMemo(() => modalityDetails.every(({ key }) => files[key]), [files]);
  const totalBytes = useMemo(() => modalityDetails.reduce((total, { key }) => total + (files[key]?.size || 0), 0), [files]);
  const modelReady = capabilities?.inference.status === "ready";

  const selectFile = (modality: Modality, file: File | null) => {
    setFiles(current => ({ ...current, [modality]: file }));
    setJob(null);
    setUsingDemo(false);
    setError(null);
  };

  const loadDemo = async () => {
    setBusy("demo");
    setError(null);
    try {
      const response = await fetch("/api/studies/demo", { method: "POST" });
      setJob(await jsonResponse<StudyJob>(response));
      setFiles(emptyFiles());
      setUsingDemo(true);
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(null); }
  };

  const validateStudy = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!allSelected) return setError("Select one .nii.gz volume for every modality.");
    setBusy("upload");
    setError(null);
    const body = new FormData();
    for (const { key } of modalityDetails) body.append(key, files[key]!);
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
      const response = await fetch(`/api/studies/${job.jobId}/inference`, { method: "POST" });
      setJob(await jsonResponse<StudyJob>(response));
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(null); }
  };

  const clearStudy = async () => {
    setBusy("clear");
    setError(null);
    try {
      if (job) {
        const response = await fetch(`/api/studies/${job.jobId}`, { method: "DELETE" });
        if (!response.ok) await jsonResponse(response);
      }
      setJob(null);
      setFiles(emptyFiles());
      setUsingDemo(false);
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(null); }
  };

  const llm = job?.explanation?.llm;
  const shownExplanation = llm?.status === "validated" && llm.artifact ? llm.artifact : job?.explanation?.deterministic;

  return <div className="view study-view" id="view-study">
    <div className="study-intro">
      <div>
        <h1>Try the MRI outline tool</h1>
        <p>Use our sample or choose four scan files. The app checks them, runs the model, and gives you an outline to download.</p>
      </div>
      <div className="model-readiness" aria-live="polite">
        <StatusMark state={!capabilities && error ? "failed" : !capabilities ? "waiting" : modelReady ? "ready" : "failed"} label={!capabilities ? "checking" : modelReady ? "ready" : "not ready"} />
        <strong>{modelReady ? "Ready to run" : error ? "Connection check failed" : "Checking the model"}</strong>
        <span>{capabilities ? "Runs privately on this computer" : "One moment"}</span>
      </div>
    </div>

    <ol className="study-steps" aria-label="Study workflow progress">
      <li aria-current={!job ? "step" : undefined}><span>1</span><strong>Choose</strong></li>
      <li aria-current={job?.state === "validated" ? "step" : undefined}><span>2</span><strong>Check</strong></li>
      <li aria-current={job?.state === "running" ? "step" : undefined}><span>3</span><strong>Process</strong></li>
      <li aria-current={job?.state === "succeeded" ? "step" : undefined}><span>4</span><strong>Download</strong></li>
    </ol>

    {error ? <div className="study-alert" role="alert"><strong>We couldn’t continue.</strong><span>{error}</span>{!capabilities ? <button className="inline-recheck" type="button" disabled={checkingCapabilities} onClick={() => { void checkCapabilities().request; }}>{checkingCapabilities ? "Checking…" : "Try connection again"}</button> : null}</div> : null}
    {!modelReady && capabilities ? <div className="study-alert" role="status"><strong>The model is not ready.</strong><span>Please try again in a moment.</span><button className="inline-recheck" type="button" disabled={checkingCapabilities} onClick={() => { void checkCapabilities().request; }}>{checkingCapabilities ? "Checking…" : "Check again"}</button></div> : null}

    <div className="study-workspace">
      <form className="modality-form" onSubmit={validateStudy}>
        <div className="section-heading"><div><h2>{usingDemo ? "Sample scans" : "Choose four scan files"}</h2><p>{usingDemo ? "The sample passed the file check." : "Files stay private and are deleted after processing."}</p></div><span className="file-total">{usingDemo ? "Ready" : allSelected ? `${formatBytes(totalBytes)} selected` : `${Object.values(files).filter(Boolean).length}/4 selected`}</span></div>
        {usingDemo ? <div className="demo-ready"><strong>Built-in sample loaded</strong><span>You can create the outline now, or clear it and use your own files.</span></div> : <div className="modality-list">
          {modalityDetails.map(({ key, label, detail }) => <label className="modality-row" key={key}>
            <span className="modality-code">{label}</span>
            <span className="modality-description"><strong>{files[key]?.name || detail}</strong>{files[key] ? <small>{formatBytes(files[key]!.size)} · ready</small> : null}</span>
            <span className={`file-action ${files[key] ? "file-action--selected" : ""}`}>{files[key] ? "Replace" : "Choose file"}</span>
            <input type="file" accept=".nii.gz,application/gzip,application/x-gzip" onChange={event => selectFile(key, event.target.files?.[0] || null)} disabled={busy !== null || job?.state === "running"} />
          </label>)}
        </div>}
        <div className="study-actions">
          {!usingDemo && !allSelected && capabilities?.demoAvailable ? <button className="primary-action" type="button" disabled={!modelReady || busy !== null} onClick={loadDemo}>{busy === "demo" ? "Loading sample…" : "Try with sample scans"}</button> : null}
          {!usingDemo && Object.values(files).some(Boolean) ? <button className="primary-action" type="submit" disabled={!allSelected || !modelReady || busy !== null || job?.state === "running"}>{busy === "upload" ? "Checking files…" : "Check my files"}</button> : null}
          <span>{usingDemo ? "Public, de-identified research sample" : "Or choose your own files above"}</span>
        </div>
      </form>

      <aside className="validation-ledger" aria-labelledby="validation-title">
        <div className="section-heading"><div><h2 id="validation-title">File check</h2></div>{job?.validation ? <StatusMark state="complete" label="passed" /> : <StatusMark state="waiting" />}</div>
        {job?.validation ? <dl>
          <div><dt>Files</dt><dd>T1 · T1ce · T2 · FLAIR</dd></div>
          <div><dt>Scan size</dt><dd>{job.validation.shape.join(" × ")} voxels</dd></div>
          <div><dt>Voxel size</dt><dd>{job.validation.spacing_mm.map(value => value.toFixed(2)).join(" × ")} mm</dd></div>
          <div><dt>Check ID</dt><dd title={job.validation.geometry_sha256}>{shortHash(job.validation.geometry_sha256)}</dd></div>
        </dl> : <div className="ledger-empty"><strong>Nothing to check yet</strong><span>Use the sample or choose four files.</span></div>}
        {job?.state === "validated" ? <button className="primary-action primary-action--full" type="button" disabled={!modelReady || busy !== null} onClick={runInference}>{busy === "run" ? "Starting…" : "Create outline"}</button> : null}
        {job?.state === "running" ? <div className="inference-running" aria-live="polite"><span className="activity-line" /><strong>Creating the outline</strong><span>This can take a few minutes. You can leave this page open.</span></div> : null}
        {job?.state === "failed" ? <div className="ledger-failure"><strong>Processing failed</strong><span>{job.error || "The model did not produce an outline."}</span></div> : null}
      </aside>
    </div>

    {job?.state === "succeeded" && job.result && shownExplanation ? <section className="study-result" aria-labelledby="result-title">
      <div className="result-heading"><div><h2 id="result-title">Research outline ready</h2><p role="note">For research only—not a medical result.</p></div><StatusMark state="complete" /></div>
      <div className="result-grid">
        <div className="result-summary">
          <strong>{job.result.segmentation.nonzero_voxels.toLocaleString()}</strong>
          <span>voxels included in the outline</span>
          <p>{shownExplanation.summary}</p>
          <p className="result-limitations">{shownExplanation.limitations}</p>
          {llm?.status !== "validated" ? <div className="llm-fallback"><strong>Showing the checked facts.</strong><span>The optional plain-language rewrite was not available.</span></div> : <div className="llm-receipt"><strong>Plain-language explanation checked</strong><span>{llm.model_name} · <span title={llm.model_digest || undefined}>{shortHash(llm.model_digest)}</span></span></div>}
        </div>
        <dl className="receipt-ledger">
          <div><dt>Model</dt><dd>{job.result.provenance.model_id}</dd></div>
          <div><dt>Model file ID</dt><dd title={job.result.provenance.checkpoint_sha256}>{shortHash(job.result.provenance.checkpoint_sha256)}</dd></div>
          <div><dt>Outline file ID</dt><dd title={job.result.segmentation.output_sha256}>{shortHash(job.result.segmentation.output_sha256)}</dd></div>
          <div><dt>Same scan size</dt><dd>Yes · {job.result.segmentation.output_shape.join(" × ")}</dd></div>
          <div><dt>Deleted after</dt><dd>{new Date(job.expiresAt).toLocaleString()}</dd></div>
        </dl>
      </div>
      <div className="artifact-actions">
        <a className="primary-action" href={`/api/studies/${job.jobId}/artifacts/segmentation`} download>Download outline</a>
        <a className="secondary-action" href={`/api/studies/${job.jobId}/artifacts/receipt`} download>Download technical receipt</a>
        <a className="secondary-action" href={`/api/studies/${job.jobId}/artifacts/explanation`} download>Download explanation</a>
        <button className="text-action" type="button" onClick={clearStudy} disabled={busy !== null}>{busy === "clear" ? "Clearing…" : "Clear local result"}</button>
      </div>
    </section> : null}
  </div>;
}
