import type { ReactNode } from "react";
import "./views.css";

export type ExplanationEvidence = { field: string; value: string | number | null };

export type ExplanationViewModel = {
  deterministicSummary: string;
  deterministicLimitations: string[];
  llmSummary?: string | null;
  llmLimitations?: string[];
  llmStatus: "valid" | "abstained" | "rejected" | "unavailable";
  fallbackReason?: string | null;
  evidence: ExplanationEvidence[];
  provenance: {
    schemaVersion: string;
    exportId: string;
    artifactHash: string;
    modelName?: string | null;
    modelDigest?: string | null;
    generatedAt: string;
  };
  validationStatus: "valid" | "rejected" | "pending";
};

function value(value: string | number | null) {
  return value == null || value === "" ? "—" : String(value);
}

function CopyableValue({ label, children, copyValue }: { label: string; children: ReactNode; copyValue: string }) {
  return <div className="view-provenance__item"><dt>{label}</dt><dd title={copyValue}><span>{children}</span><button type="button" onClick={() => void navigator.clipboard?.writeText(copyValue)} aria-label={`Copy ${label.toLowerCase()}`}>Copy</button></dd></div>;
}

export function ExplanationsView({ explanation }: { explanation: ExplanationViewModel | null }) {
  if (!explanation) return <section className="view-shell" aria-labelledby="explanations-title"><div className="view-heading"><div><h1 id="explanations-title">Explanations</h1></div></div><div className="view-state view-state--empty"><h2>No explanation available</h2><p>Validated research envelope data has not been received.</p></div></section>;
  const llmUnavailable = explanation.llmStatus !== "valid";
  return <section className="view-shell" aria-labelledby="explanations-title">
    <div className="view-heading"><div><h1 id="explanations-title">Explanations</h1><p className="view-lede">Canonical facts are rendered deterministically from a sanitized/direct-identifier-free research envelope.</p></div><span className={`view-status view-status--${explanation.validationStatus}`}>{explanation.validationStatus}</span></div>
    <p className="view-disclaimer" role="note">Research output only; not a diagnosis or treatment recommendation.</p>
    <div className="view-grid view-grid--two">
      <article className="view-panel"><div className="view-panel__heading"><h2>Deterministic summary</h2><span className="view-status view-status--valid">Source of truth</span></div><p>{explanation.deterministicSummary}</p><h3>Limitations</h3><ul>{explanation.deterministicLimitations.map(item => <li key={item}>{item}</li>)}</ul></article>
      <article className="view-panel"><div className="view-panel__heading"><h2>Optional LLM rendering</h2><span className={`view-status view-status--${explanation.llmStatus}`}>{explanation.llmStatus}</span></div>{explanation.llmSummary && !llmUnavailable ? <><p>{explanation.llmSummary}</p>{explanation.llmLimitations?.length ? <><h3>Limitations</h3><ul>{explanation.llmLimitations.map(item => <li key={item}>{item}</li>)}</ul></> : null}</> : <p className="view-state__copy">LLM explanation unavailable — deterministic facts shown</p>}{llmUnavailable && explanation.fallbackReason ? <p className="view-helper">Fallback reason: {explanation.fallbackReason}</p> : null}</article>
    </div>
    <article className="view-panel"><div className="view-panel__heading"><h2>Evidence fields</h2><span className="view-helper">Traceable source values</span></div><div className="view-table-wrap"><table className="view-table"><caption className="sr-only">Evidence fields from the validated research envelope</caption><thead><tr><th scope="col">Field</th><th scope="col">Value</th></tr></thead><tbody>{explanation.evidence.map(item => <tr key={item.field}><th scope="row">{item.field}</th><td>{value(item.value)}</td></tr>)}</tbody></table></div></article>
    <article className="view-panel"><div className="view-panel__heading"><h2>Provenance and validation</h2><span className={`view-status view-status--${explanation.validationStatus}`}>{explanation.validationStatus}</span></div><dl className="view-provenance"><CopyableValue label="Schema version" copyValue={explanation.provenance.schemaVersion}>{explanation.provenance.schemaVersion}</CopyableValue><CopyableValue label="Export ID" copyValue={explanation.provenance.exportId}>{explanation.provenance.exportId}</CopyableValue><CopyableValue label="Artifact hash" copyValue={explanation.provenance.artifactHash}><code>{explanation.provenance.artifactHash}</code></CopyableValue><CopyableValue label="Model name" copyValue={explanation.provenance.modelName || "No model name recorded"}>{explanation.provenance.modelName || "No model name recorded"}</CopyableValue><CopyableValue label="Model digest" copyValue={explanation.provenance.modelDigest || "No model digest recorded"}><code>{explanation.provenance.modelDigest || "No model digest recorded"}</code></CopyableValue><CopyableValue label="Generation time" copyValue={explanation.provenance.generatedAt}>{explanation.provenance.generatedAt}</CopyableValue></dl></article>
  </section>;
}

export default ExplanationsView;
