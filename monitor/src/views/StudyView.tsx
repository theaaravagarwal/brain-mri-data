import { useEffect, useMemo, useState } from "react";
import type { StudyCapabilities, StudyJob } from "../types";

type Modality = "t1" | "t1ce" | "t2" | "flair";
type Files = Record<Modality, File | null>;

const modalityDetails: ReadonlyArray<{ key: Modality; label: string; detail: string }> = [
  { key: "t1", label: "T1", detail: "Native T1-weighted volume" },
  { key: "t1ce", label: "T1ce", detail: "Contrast-enhanced T1 volume" },
  { key: "t2", label: "T2", detail: "T2-weighted volume" },
  { key: "flair", label: "FLAIR", detail: "Fluid-attenuated volume" }
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
  const [busy, setBusy] = useState<"upload" | "run" | "clear" | null>(null);
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
    setError(null);
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
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(null); }
  };

  const llm = job?.explanation?.llm;
  const shownExplanation = llm?.status === "validated" && llm.artifact ? llm.artifact : job?.explanation?.deterministic;

  return <div className="view study-view" id="view-study">
    <div className="study-intro">
      <div>
        <h1>Segment one four-volume MRI study</h1>
        <p>Select T1, T1ce, T2, and FLAIR NIfTI volumes. The fixed CNN validates and runs on the private NVIDIA worker; image data never reaches the language model.</p>
      </div>
      <div className="model-readiness" aria-live="polite">
        <StatusMark state={!capabilities && error ? "failed" : !capabilities ? "waiting" : modelReady ? "ready" : "failed"} label={!capabilities && error ? "gateway unavailable" : !capabilities ? "checking inference" : modelReady ? "inference ready" : "inference unavailable"} />
        <strong>{capabilities?.inference.modelId || (error ? "Capability check failed" : "Checking fixed model")}</strong>
        <span>{capabilities ? `${capabilities.inference.device} · ${capabilities.inference.outputKind.replaceAll("_", " ")}` : "Confirming the local checkpoint"}</span>
      </div>
    </div>

    <ol className="study-steps" aria-label="Study workflow progress">
      <li aria-current={!job ? "step" : undefined}><span>1</span><strong>Select</strong><small>Four NIfTI volumes</small></li>
      <li aria-current={job?.state === "validated" ? "step" : undefined}><span>2</span><strong>Validate</strong><small>Modalities and geometry</small></li>
      <li aria-current={job?.state === "running" ? "step" : undefined}><span>3</span><strong>Run</strong><small>Fixed local CNN</small></li>
      <li aria-current={job?.state === "succeeded" ? "step" : undefined}><span>4</span><strong>Return</strong><small>Mask, receipt, explanation</small></li>
    </ol>

    {error ? <div className="study-alert" role="alert"><strong>The study could not continue.</strong><span>{error}</span>{!capabilities ? <button className="inline-recheck" type="button" disabled={checkingCapabilities} onClick={() => { void checkCapabilities().request; }}>{checkingCapabilities ? "Checking fixed model…" : "Retry capability check"}</button> : null}</div> : null}
    {!modelReady && capabilities ? <div className="study-alert" role="status"><strong>Inference unavailable.</strong><span>The checkpoint is missing or its digest does not match. Files will not upload until the exact fixed model is ready.</span><button className="inline-recheck" type="button" disabled={checkingCapabilities} onClick={() => { void checkCapabilities().request; }}>{checkingCapabilities ? "Checking fixed model…" : "Recheck fixed model"}</button></div> : null}

    <div className="study-workspace">
      <form className="modality-form" onSubmit={validateStudy}>
        <div className="section-heading"><div><h2>Study volumes</h2><p>Files are renamed to server-generated research identifiers and removed after processing.</p></div><span className="file-total">{allSelected ? `${formatBytes(totalBytes)} selected` : `${Object.values(files).filter(Boolean).length}/4 selected`}</span></div>
        <div className="modality-list">
          {modalityDetails.map(({ key, label, detail }) => <label className="modality-row" key={key}>
            <span className="modality-code">{label}</span>
            <span className="modality-description"><strong>{files[key]?.name || detail}</strong><small>{files[key] ? `${formatBytes(files[key]!.size)} · ready to validate` : "Compressed NIfTI (.nii.gz)"}</small></span>
            <span className={`file-action ${files[key] ? "file-action--selected" : ""}`}>{files[key] ? "Replace" : "Choose file"}</span>
            <input type="file" accept=".nii.gz,application/gzip,application/x-gzip" onChange={event => selectFile(key, event.target.files?.[0] || null)} disabled={busy !== null || job?.state === "running"} />
          </label>)}
        </div>
        <div className="study-actions">
          <button className="primary-action" type="submit" disabled={!allSelected || !modelReady || busy !== null || job?.state === "running"}>{busy === "upload" ? "Validating study…" : job?.validation ? "Validate again" : "Validate study"}</button>
          <span>Maximum 512 MiB per volume · uploads remain on this private worker</span>
        </div>
      </form>

      <aside className="validation-ledger" aria-labelledby="validation-title">
        <div className="section-heading"><div><h2 id="validation-title">Validation ledger</h2><p>Only contract facts are shown.</p></div>{job?.validation ? <StatusMark state="complete" /> : <StatusMark state="waiting" />}</div>
        {job?.validation ? <dl>
          <div><dt>Modalities</dt><dd>T1 · T1ce · T2 · FLAIR</dd></div>
          <div><dt>Geometry</dt><dd>{job.validation.shape.join(" × ")} voxels</dd></div>
          <div><dt>Spacing</dt><dd>{job.validation.spacing_mm.map(value => value.toFixed(2)).join(" × ")} mm</dd></div>
          <div><dt>Geometry receipt</dt><dd title={job.validation.geometry_sha256}>{shortHash(job.validation.geometry_sha256)}</dd></div>
        </dl> : <div className="ledger-empty"><strong>No validated study yet.</strong><span>Select all four volumes and run validation. The CNN cannot start before this ledger passes.</span></div>}
        {job?.state === "validated" ? <button className="primary-action primary-action--full" type="button" disabled={!modelReady || busy !== null} onClick={runInference}>{busy === "run" ? "Starting inference…" : "Run fixed CNN"}</button> : null}
        {job?.state === "running" ? <div className="inference-running" aria-live="polite"><span className="activity-line" /><strong>Local inference is running</strong><span>Closing this page does not cancel the job. The gateway stops it after 30 minutes if it does not finish; no unsafe browser cancel control is exposed.</span></div> : null}
        {job?.state === "failed" ? <div className="ledger-failure"><strong>Inference failed</strong><span>{job.error || "The fixed runner did not produce a valid result."}</span></div> : null}
      </aside>
    </div>

    {job?.state === "succeeded" && job.result && shownExplanation ? <section className="study-result" aria-labelledby="result-title">
      <div className="result-heading"><div><h2 id="result-title">Research segmentation complete</h2><p role="note"><strong>Research output only.</strong> Not a diagnosis, treatment recommendation, or clinical result.</p></div><StatusMark state="complete" /></div>
      <div className="result-grid">
        <div className="result-summary">
          <strong>{job.result.segmentation.nonzero_voxels.toLocaleString()}</strong>
          <span>non-zero output voxels · binary labels 0 and 1</span>
          <p>{shownExplanation.summary}</p>
          <p className="result-limitations">{shownExplanation.limitations}</p>
          {llm?.status !== "validated" ? <div className="llm-fallback"><strong>LLM explanation unavailable — validated deterministic metadata shown.</strong><span>{llm?.reason || "No local model was configured."}</span></div> : <div className="llm-receipt"><strong>Validated local LLM rendering</strong><span>{llm.model_name} · <span title={llm.model_digest || undefined}>{shortHash(llm.model_digest)}</span></span></div>}
        </div>
        <dl className="receipt-ledger">
          <div><dt>Model</dt><dd>{job.result.provenance.model_id}</dd></div>
          <div><dt>Checkpoint</dt><dd title={job.result.provenance.checkpoint_sha256}>{shortHash(job.result.provenance.checkpoint_sha256)}</dd></div>
          <div><dt>Output SHA-256</dt><dd title={job.result.segmentation.output_sha256}>{shortHash(job.result.segmentation.output_sha256)}</dd></div>
          <div><dt>Geometry preserved</dt><dd>Yes · {job.result.segmentation.output_shape.join(" × ")}</dd></div>
          <div><dt>Retention</dt><dd>Until {new Date(job.expiresAt).toLocaleString()}</dd></div>
        </dl>
      </div>
      <div className="artifact-actions">
        <a className="primary-action" href={`/api/studies/${job.jobId}/artifacts/segmentation`} download>Download segmentation</a>
        <a className="secondary-action" href={`/api/studies/${job.jobId}/artifacts/receipt`} download>Download exact receipt</a>
        <a className="secondary-action" href={`/api/studies/${job.jobId}/artifacts/explanation`} download>Download explanation</a>
        <button className="text-action" type="button" onClick={clearStudy} disabled={busy !== null}>{busy === "clear" ? "Clearing…" : "Clear local result"}</button>
      </div>
    </section> : null}
  </div>;
}
