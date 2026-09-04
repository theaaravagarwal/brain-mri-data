import "./evidence.css";

type Scope = "internal validation" | "independent test" | string;

export type EvidenceMetric = {
  label: string;
  unit?: string;
  baseline: number | null;
  candidate: number | null;
  higherIsBetter?: boolean;
};

export type EvidenceSeed = {
  seed: number | string;
  baseline: number | null;
  candidate: number | null;
};

export type EvidenceEffect = {
  label: string;
  delta: number;
  ci95: [number, number];
  higherIsBetter: boolean;
  exploratory: boolean;
};

export type EvidenceResource = {
  studyId?: string | null;
  protocol?: string | null;
  architecture?: string | null;
  trainer?: string | null;
  dataset?: string | null;
  scope?: Scope | null;
  generatedAt?: string | null;
  freshness?: string | null;
  baselineLabel?: string | null;
  candidateLabel?: string | null;
  metrics?: EvidenceMetric[];
  seeds?: EvidenceSeed[];
  effects?: EvidenceEffect[];
  gate?: { allowed: boolean; missing: string[]; decision?: string | null } | null;
  researchUse?: { allowed: boolean; reason: string } | null;
  selectedModel?: { label: string; seed: number | string; readiness: string } | null;
  provenance?: Record<string, string | null | undefined> | null;
};

export type EvidenceViewProps = { resource?: EvidenceResource | null };

const fmt = (value: number | null | undefined, unit = "") =>
  value == null || !Number.isFinite(value) ? "—" : `${value.toFixed(unit === "mm" ? 1 : 4)}${unit ? ` ${unit}` : ""}`;

const delta = (metric: EvidenceMetric) => {
  if (metric.baseline == null || metric.candidate == null) return null;
  return metric.candidate - metric.baseline;
};

const formatDelta = (metric: EvidenceMetric) => {
  const value = delta(metric);
  if (value == null) return "—";
  const good = metric.higherIsBetter === false ? value < 0 : value > 0;
  return <span className={good ? "evidence-delta evidence-delta--good" : value === 0 ? "evidence-delta" : "evidence-delta evidence-delta--attention"}>{value > 0 ? "+" : ""}{value.toFixed(metric.unit === "mm" ? 1 : 4)}{metric.unit ? ` ${metric.unit}` : ""}</span>;
};

function Freshness({ resource }: EvidenceViewProps) {
  return <p className="evidence-freshness">{resource?.scope || "Evaluation scope unavailable"} · {resource?.freshness || (resource?.generatedAt ? `Generated ${new Date(resource.generatedAt).toLocaleString()}` : "Freshness unavailable")}</p>;
}

function Provenance({ values }: { values?: EvidenceResource["provenance"] }) {
  const entries = Object.entries(values || {}).filter(([, value]) => value);
  return <section className="evidence-ledger" aria-labelledby="evidence-provenance-title">
    <h2 id="evidence-provenance-title">Provenance</h2>
    {entries.length ? <dl className="evidence-provenance">{entries.map(([key, value]) => <div key={key}><dt>{key.replaceAll(/([A-Z])/g, " $1")}</dt><dd><code title={value || undefined}>{value}</code><button type="button" onClick={() => void navigator.clipboard?.writeText(value || "")} aria-label={`Copy ${key}`}>Copy</button></dd></div>)}</dl> : <p className="evidence-muted">No immutable identifiers are available for this comparison.</p>}
  </section>;
}

function SeedView({ seeds }: { seeds: EvidenceSeed[] }) {
  const max = Math.max(1, ...seeds.flatMap(seed => [seed.baseline || 0, seed.candidate || 0]));
  return <section className="evidence-ledger" aria-labelledby="evidence-seeds-title">
    <div className="evidence-section-heading"><div><h2 id="evidence-seeds-title">Seed stability</h2><p>Points are shown per seed; aggregates do not decide promotion alone.</p></div></div>
    {seeds.length ? <>
      <div className="seed-plot" role="img" aria-label="Seed-level baseline and candidate comparison"><span className="seed-axis">Higher value →</span>{seeds.map(seed => <div className="seed-row" key={String(seed.seed)}><strong>{seed.seed}</strong><span className="seed-track"><i className="seed-bar seed-bar--baseline" style={{ width: `${((seed.baseline || 0) / max) * 100}%` }} /><i className="seed-bar seed-bar--candidate" style={{ width: `${((seed.candidate || 0) / max) * 100}%` }} /></span></div>)}</div>
      <div className="evidence-table-scroll"><table className="evidence-table"><caption>Seed-level results</caption><thead><tr><th scope="col">Seed</th><th scope="col">Baseline</th><th scope="col">Candidate</th><th scope="col">Delta</th></tr></thead><tbody>{seeds.map(seed => <tr key={String(seed.seed)}><th scope="row">{seed.seed}</th><td>{fmt(seed.baseline)}</td><td>{fmt(seed.candidate)}</td><td>{seed.baseline == null || seed.candidate == null ? "—" : `${seed.candidate - seed.baseline > 0 ? "+" : ""}${(seed.candidate - seed.baseline).toFixed(4)}`}</td></tr>)}</tbody></table></div>
    </> : <p className="evidence-muted">Seed results are not available yet.</p>}
  </section>;
}

export function EvidenceView({ resource }: EvidenceViewProps) {
  if (!resource) return <section className="evidence-view" aria-labelledby="evidence-title"><header className="evidence-header"><div><h1 id="evidence-title">Model evidence</h1><p>Frozen run comparison and promotion review</p></div></header><div className="evidence-empty" role="status"><strong>Evidence unavailable</strong><span>This gateway has not supplied a validated evidence envelope.</span><small>Nothing has been promoted. Return when a completed comparison is available.</small></div></section>;
  const metrics = resource.metrics || [];
  const seeds = resource.seeds || [];
  const gate = resource.gate;
  return <section className="evidence-view" aria-labelledby="evidence-title">
    <header className="evidence-header"><div><h1 id="evidence-title">Model evidence</h1><p>Default selection and research testing are separate decisions</p><Freshness resource={resource} /></div><span className={`evidence-gate ${resource.researchUse?.allowed ? "evidence-gate--open" : "evidence-gate--blocked"}`}>{resource.researchUse?.allowed ? "Candidate testable" : "Testing held"}</span></header>
    <section className="evidence-ledger" aria-labelledby="evidence-context-title"><h2 id="evidence-context-title">Study context</h2><dl className="evidence-context">{[["Study", resource.studyId], ["Protocol", resource.protocol], ["Architecture / trainer", [resource.architecture, resource.trainer].filter(Boolean).join(" · ")], ["Dataset / split", resource.dataset], ["Scope", resource.scope], ["Retained model", resource.selectedModel ? `${resource.selectedModel.label} · seed ${resource.selectedModel.seed}` : null], ["Readiness", resource.selectedModel?.readiness]].map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value || "—"}</dd></div>)}</dl></section>
    <section className="evidence-ledger" aria-labelledby="evidence-metrics-title"><div className="evidence-section-heading"><div><h2 id="evidence-metrics-title">Baseline versus candidate</h2><p>Delta direction is explicit: higher is better unless marked otherwise.</p></div><div className="evidence-legend"><span><i className="legend-swatch legend-swatch--baseline" />Baseline</span><span><i className="legend-swatch legend-swatch--candidate" />Candidate</span></div></div><div className="evidence-metric-table"><table className="evidence-table"><caption>Frozen comparison metrics</caption><thead><tr><th scope="col">Measure</th><th scope="col">Baseline</th><th scope="col">Candidate</th><th scope="col">Delta</th></tr></thead><tbody>{metrics.length ? metrics.map(metric => <tr key={metric.label}><th scope="row">{metric.label} {metric.higherIsBetter === false && <small>(lower is better)</small>}</th><td>{fmt(metric.baseline, metric.unit)}</td><td>{fmt(metric.candidate, metric.unit)}</td><td>{formatDelta(metric)}</td></tr>) : <tr><td colSpan={4} className="evidence-muted">No final metrics supplied.</td></tr>}</tbody></table></div></section>
    <section className="evidence-ledger" aria-labelledby="evidence-effects-title"><div className="evidence-section-heading"><div><h2 id="evidence-effects-title">Paired effects and uncertainty</h2><p>Candidate minus baseline, paired by the same observation and seed.</p></div></div><div className="evidence-table-scroll"><table className="evidence-table"><caption>Case-cluster bootstrap intervals conditional on three seeds</caption><thead><tr><th scope="col">Measure</th><th scope="col">Effect</th><th scope="col">95% interval</th><th scope="col">Role</th></tr></thead><tbody>{resource.effects?.length ? resource.effects.map(effect => <tr key={effect.label}><th scope="row">{effect.label} {!effect.higherIsBetter && <small>(lower is better)</small>}</th><td>{effect.delta > 0 ? "+" : ""}{effect.delta.toFixed(effect.label.includes("HD95") ? 2 : 4)}</td><td>[{effect.ci95[0] > 0 ? "+" : ""}{effect.ci95[0].toFixed(effect.label.includes("HD95") ? 2 : 4)}, {effect.ci95[1] > 0 ? "+" : ""}{effect.ci95[1].toFixed(effect.label.includes("HD95") ? 2 : 4)}]</td><td>{effect.exploratory ? "Exploratory" : "Primary"}</td></tr>) : <tr><td colSpan={4} className="evidence-muted">No paired effects supplied.</td></tr>}</tbody></table></div></section>
    <SeedView seeds={seeds} />
    <section className="evidence-ledger" aria-labelledby="evidence-research-title"><h2 id="evidence-research-title">Research use</h2><div className="evidence-gate-copy"><strong>{resource.researchUse?.allowed ? "Candidate allowed for controlled comparisons" : "Candidate testing held"}</strong><p>{resource.researchUse?.reason || "No research-use decision was supplied."}</p></div></section>
    <section className="evidence-ledger" aria-labelledby="evidence-gate-title"><h2 id="evidence-gate-title">Default model</h2><div className="evidence-gate-copy"><strong>{gate?.allowed ? "Candidate may replace the default after review" : "Baseline stays the default"}</strong><p>{gate?.decision || "Automatic model promotion remains disabled."}</p></div><ul className="evidence-missing">{gate?.missing?.length ? gate.missing.map(item => <li key={item}>{item}</li>) : <li>No missing evidence reported.</li>}</ul></section>
    <Provenance values={resource.provenance} />
    <p className="evidence-disclaimer">Research output only; not a diagnosis or treatment recommendation.</p>
  </section>;
}

export default EvidenceView;
