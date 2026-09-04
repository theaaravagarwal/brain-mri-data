import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import type {
  DashboardSnapshot,
  EvidenceResource as EvidencePayload,
  ExplanationResource as ExplanationPayload,
  HostEnvelope,
  ProposalResource as ProposalPayload,
  RecentRun,
  ResourceEnvelope,
  RunSnapshot
} from "./types";
import type { EvidenceResource as EvidenceViewModel } from "./views/EvidenceView";
import type { ExplanationViewModel } from "./views/ExplanationsView";
import type { JobProposal, ProposalsViewModel } from "./views/ProposalsView";

const EvidenceView = lazy(() => import("./views/EvidenceView"));
const ExplanationsView = lazy(() => import("./views/ExplanationsView"));
const ProposalsView = lazy(() => import("./views/ProposalsView"));
const StudyView = lazy(() => import("./views/StudyView"));

type MetricKey = "meanDice" | "meanBoxIou" | "meanHd95Mm";

const metrics: Record<MetricKey, { label: string; unit: string; color: string; domain: [number | "auto", number | "auto"] }> = {
  meanDice: { label: "Internal validation Dice", unit: "", color: "var(--active)", domain: [0, 1] },
  meanBoxIou: { label: "Box IoU", unit: "", color: "var(--complete)", domain: [0, 1] },
  meanHd95Mm: { label: "HD95", unit: " mm", color: "var(--warning)", domain: [0, "auto"] }
};

type ViewKey = "study" | "operations" | "evidence" | "explanations" | "proposals";

const views: ReadonlyArray<{ key: ViewKey; label: string }> = [
  { key: "study", label: "New study" },
  { key: "operations", label: "Operations" },
  { key: "evidence", label: "Model evidence" },
  { key: "explanations", label: "Explanations" },
  { key: "proposals", label: "Proposals" }
];

function viewFromHash(): ViewKey {
  const value = window.location.hash.replace(/^#\/?/, "");
  if (value === "overview") return "operations";
  return views.some(view => view.key === value) ? value as ViewKey : "study";
}

function formatNumber(value: number | null | undefined, digits = 1) {
  return value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

function formatAge(iso: string | null, now: number) {
  if (!iso) return "never confirmed";
  const seconds = Math.max(0, Math.floor((now - Date.parse(iso)) / 1000));
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m ago`;
}

function formatTimestamp(iso: string | null) {
  if (!iso) return "No timestamp recorded";
  const value = new Date(iso);
  return Number.isNaN(value.getTime()) ? "Invalid timestamp" : value.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

function shortRun(name: string) {
  return name.replace("glioma-pilot-v4-uniform-control-retry1--cuda--brats--", "uniform · ")
    .replace("glioma-exploratory-rocm-compact-v2--brats--", "ROCm · ");
}

function statusLabel(host: HostEnvelope) {
  if (!host.data) return "Unavailable";
  if (!host.reachable || host.stale) return "Stale";
  if (host.data.activeRun) return "Training";
  if (host.data.queue.state === "attention") return "Attention";
  return "Reporting";
}

function State({ host }: { host: HostEnvelope }) {
  const state = statusLabel(host).toLowerCase();
  return <span className={`state state--${state}`}><span aria-hidden="true" />{statusLabel(host)}</span>;
}

function Measure({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return <div className="measure"><dt>{label}</dt><dd>{value}</dd>{detail && <small>{detail}</small>}</div>;
}

function HostBand({ host, now }: { host: HostEnvelope; now: number }) {
  const data = host.data;
  if (!data) {
    return <article className="host-band host-band--empty">
      <div><h2>Worker unavailable</h2><p>{host.error || "Waiting for the first snapshot."}</p></div>
      <State host={host} />
    </article>;
  }
  const gpuMemory = data.gpu.memoryUsedMib == null ? "—" : `${formatNumber(data.gpu.memoryUsedMib / 1024)} GiB`;
  const progress = data.activeRun;
  const percent = progress?.completed != null && progress.total ? (progress.completed / progress.total) * 100 : progress?.epoch && progress.epochs ? (progress.epoch / progress.epochs) * 100 : null;
  const lowDisk = data.disk.totalGib > 0 && data.disk.freeGib / data.disk.totalGib < 0.15;
  return <article className={`host-band ${host.stale ? "host-band--stale" : ""}`}>
    <div className="host-identity">
      <div className="host-title"><h2>{data.label}</h2><State host={host} /></div>
      <p>{data.role} · {data.hostname}</p>
      <p className="freshness">Confirmed {formatAge(host.lastSuccessAt, now)} · {host.latencyMs ?? "—"} ms collection latency</p>
      {host.staleSince ? <p className="stale-context">Stale since {formatAge(host.staleSince, now)} · last known values retained</p> : null}
      {host.error && <p className="inline-error">{host.error}</p>}
    </div>
    <dl className="host-measures">
      <Measure label="GPU" value={data.gpu.utilizationPercent == null ? (data.gpu.active ? "Active" : "Idle") : `${formatNumber(data.gpu.utilizationPercent, 0)}%`} detail={data.gpu.name} />
      <Measure label="VRAM" value={gpuMemory} detail={data.gpu.memoryTotalMib ? `${formatNumber(data.gpu.memoryTotalMib / 1024, 0)} GiB total` : undefined} />
      <div className={lowDisk ? "measure measure--attention" : "measure"}><dt>Disk free</dt><dd>{formatNumber(data.disk.freeGib, 0)} GiB</dd><small>{lowDisk ? "Low headroom · " : ""}{formatNumber((data.disk.usedGib / data.disk.totalGib) * 100, 0)}% used</small></div>
      <Measure label="RAM" value={data.memory.usedGib == null ? "—" : `${formatNumber(data.memory.usedGib)} GiB`} detail={data.memory.totalGib == null ? undefined : `${formatNumber(data.memory.totalGib, 0)} GiB total`} />
      <Measure label="Temperature" value={data.gpu.temperatureC == null ? "No metric recorded" : `${formatNumber(data.gpu.temperatureC, 0)} °C`} />
      <Measure label="Power" value={data.gpu.powerW == null ? "No metric recorded" : `${formatNumber(data.gpu.powerW, 0)} W`} />
    </dl>
    <div className="host-work">
      <div className="host-work__line"><span>{progress ? shortRun(progress.name) : "No active training run"}</span><strong>{progress?.phase?.replaceAll("_", " ") || data.queue.state}</strong></div>
      <progress className="progress-track" aria-label={`${data.label} progress`} max={100} value={percent == null ? 0 : Math.min(percent, 100)} />
      <div className="host-work__meta">
        {progress ? <span>Epoch {progress.epoch ?? "—"}/{progress.epochs ?? "—"}{progress.completed != null && progress.total ? ` · ${progress.completed}/${progress.total} ${progress.unit}s` : ""}</span> : <span>{data.queue.detail || "Queue empty"}</span>}
        <span>{progress?.liveLoss != null ? `Live loss estimate ${formatNumber(progress.liveLoss, 4)}` : `${data.sessions.length} session${data.sessions.length === 1 ? "" : "s"}`}</span>
      </div>
    </div>
  </article>;
}

function MetricChart({ run, isHistorical }: { run: RunSnapshot | null; isHistorical: boolean }) {
  const [metric, setMetric] = useState<MetricKey>("meanDice");
  const definition = metrics[metric];
  const latest = run?.metrics.at(-1);
  const points = useMemo(() => {
    const values = (run?.metrics || []).flatMap(point => {
      const value = point[metric];
      return value == null || !Number.isFinite(value) ? [] : [{ epoch: point.epoch, value }];
    });
    if (!values.length) return [];
    const minEpoch = Math.min(...values.map(point => point.epoch));
    const maxEpoch = Math.max(...values.map(point => point.epoch));
    const floor = definition.domain[0] === "auto" ? Math.min(...values.map(point => point.value)) : definition.domain[0];
    const ceiling = definition.domain[1] === "auto" ? Math.max(...values.map(point => point.value)) : definition.domain[1];
    const span = Math.max(ceiling - floor, 0.0001);
    const epochSpan = Math.max(maxEpoch - minEpoch, 1);
    return values.map(point => ({
      ...point,
      x: 44 + ((point.epoch - minEpoch) / epochSpan) * 688,
      y: 210 - ((point.value - floor) / span) * 174
    }));
  }, [definition.domain, metric, run]);
  const scope = isHistorical ? "Completed internal validation result" : "Live training estimate · not final validation";
  return <section className="metric-section" aria-labelledby="metric-title">
    <div className="section-heading">
      <div><h2 id="metric-title">Validation history</h2><p>{run ? `${shortRun(run.name)} · ${scope}` : "No completed training run available"}</p></div>
      <div className="metric-tabs" aria-label="Choose metric">
        {(Object.keys(metrics) as MetricKey[]).map(key => <button key={key} type="button" aria-pressed={metric === key} onClick={() => setMetric(key)}>{metrics[key].label.replace("Internal validation ", "")}</button>)}
      </div>
    </div>
    <div className="metric-latest">
      <strong>{formatNumber(latest?.[metric], metric === "meanHd95Mm" ? 1 : 4)}{latest?.[metric] != null ? definition.unit : ""}</strong>
      <span>{latest ? `Epoch ${latest.epoch}` : "No completed validation epoch"}</span>
    </div>
    <div className="chart" aria-label={`${definition.label} by epoch`}>
      {points.length ? <svg viewBox="0 0 760 240" role="img" aria-labelledby="chart-title chart-desc" preserveAspectRatio="none">
        <title id="chart-title">{definition.label} by validation epoch</title>
        <desc id="chart-desc">{points.length} recorded values. Exact values are available in the table below.</desc>
        {[36, 94, 152, 210].map(y => <line key={y} x1="44" x2="732" y1={y} y2={y} className="chart-rule" />)}
        <polyline points={points.map(point => `${point.x},${point.y}`).join(" ")} fill="none" stroke={definition.color} strokeWidth="2.5" vectorEffect="non-scaling-stroke" />
        {points.map(point => <circle key={point.epoch} cx={point.x} cy={point.y} r="3.5" fill={definition.color}><title>Epoch {point.epoch}: {formatNumber(point.value, metric === "meanHd95Mm" ? 1 : 4)}{definition.unit}</title></circle>)}
      </svg> : <div className="empty-chart"><span>No metric series yet.</span><small>The chart appears after a validation epoch is recorded.</small></div>}
    </div>
    <div className="metric-strip">
      <Measure label="Train loss" value={formatNumber(latest?.trainLoss, 4)} />
      <Measure label="Dice" value={formatNumber(latest?.meanDice, 4)} />
      <Measure label="Box IoU" value={formatNumber(latest?.meanBoxIou, 4)} />
      <Measure label="HD95" value={latest?.meanHd95Mm == null ? "—" : `${formatNumber(latest.meanHd95Mm, 1)} mm`} />
    </div>
    {run?.metrics.length ? <details className="metric-data"><summary>Exact epoch values</summary><div className="table-scroll"><table><caption>{scope}. Pseudo metrics are trajectory telemetry, not final model selection evidence.</caption><thead><tr><th scope="col">Epoch</th><th scope="col">Train loss</th><th scope="col">Dice</th><th scope="col">Box IoU</th><th scope="col">HD95</th></tr></thead><tbody>{run.metrics.map(point => <tr key={point.epoch}><th scope="row">{point.epoch}</th><td>{formatNumber(point.trainLoss, 4)}</td><td>{formatNumber(point.meanDice, 4)}</td><td>{formatNumber(point.meanBoxIou, 4)}</td><td>{point.meanHd95Mm == null ? "No metric recorded" : `${formatNumber(point.meanHd95Mm, 1)} mm`}</td></tr>)}</tbody></table></div></details> : null}
  </section>;
}

function QueueSection({ snapshot }: { snapshot: DashboardSnapshot }) {
  const entries = (["nvidia", "amd"] as const).map(id => snapshot.hosts[id]).filter(host => host.data);
  return <section className="queue-section" aria-labelledby="queue-title">
    <div className="section-heading"><div><h2 id="queue-title">Durable training queues</h2><p>Read-only supervised job state</p></div></div>
    <div className="queue-list">
      {entries.map(host => <div className="queue-row" key={host.data!.id}>
        <div><strong>{host.data!.label}</strong><span>{host.data!.queue.detail || "No queued work"}</span></div>
        <span className={`queue-state queue-state--${host.data!.queue.state}`}>{host.data!.queue.state}</span>
        <div className="session-list">
          <span>{host.data!.queue.serviceState} service · {host.data!.queue.completedCount}/{host.data!.queue.totalCount} complete{host.data!.queue.failedCount ? ` · ${host.data!.queue.failedCount} failed` : ""}</span>
          {host.data!.queue.currentRun ? <code>{host.data!.queue.currentRun}</code> : null}
          {host.data!.queue.queuedRuns.map(run => <code key={run}>{run}</code>)}
        </div>
      </div>)}
    </div>
  </section>;
}

function RunRow({ host, run }: { host: string; run: RecentRun }) {
  return <tr><td><span className="host-code">{host}</span></td><th scope="row" title={run.name}>{shortRun(run.name)}</th><td>{run.seed ?? "—"}</td><td>{run.bestEpoch ?? "—"}</td><td>{formatNumber(run.bestDice, 4)}</td><td>{formatNumber(run.bestBoxIou, 4)}</td><td>{run.bestHd95Mm == null ? "No metric recorded" : `${formatNumber(run.bestHd95Mm, 1)} mm`}</td><td><span className={`run-status run-status--${run.status}`}>{run.status}</span></td></tr>;
}

function RecentRuns({ snapshot }: { snapshot: DashboardSnapshot }) {
  const rows = (["nvidia", "amd"] as const).flatMap(id => (snapshot.hosts[id].data?.recentRuns || []).map(run => ({ host: id, run }))).sort((a, b) => Date.parse(b.run.modifiedAt) - Date.parse(a.run.modifiedAt)).slice(0, 10);
  return <section className="runs-section" aria-labelledby="runs-title">
    <div className="section-heading"><div><h2 id="runs-title">Recent runs</h2><p>Completed internal validation results · best recorded epoch per run</p></div></div>
    <div className="table-scroll"><table><caption>Internal BraTS development and validation evidence; not independent test evidence.</caption><thead><tr><th scope="col">Host</th><th scope="col">Run</th><th scope="col">Seed</th><th scope="col">Best epoch</th><th scope="col">Dice</th><th scope="col">Box IoU</th><th scope="col">HD95</th><th scope="col">Status</th></tr></thead><tbody>{rows.length ? rows.map(({ host, run }) => <RunRow key={`${host}-${run.name}`} host={host} run={run} />) : <tr><td colSpan={8} className="empty-cell">No recorded runs available.</td></tr>}</tbody></table></div>
  </section>;
}

function Loading() {
  return <div className="view-loading" aria-busy="true" aria-live="polite"><strong>Loading research workspace…</strong><span>Checking the local validated fixture.</span></div>;
}

function OverviewView({ snapshot, now }: { snapshot: DashboardSnapshot; now: number }) {
  const activeRun = snapshot.hosts.nvidia.data?.activeRun || snapshot.hosts.amd.data?.activeRun || null;
  const displayedRun = activeRun || snapshot.hosts.nvidia.data?.latestRun || snapshot.hosts.amd.data?.latestRun || null;
  return <div className="view" id="view-overview">
    <div className="view-heading"><div><h1>Worker and training overview</h1><p>Current compute state with last-known values retained through connection failures.</p></div><div className="scope-stamp"><strong>Operations telemetry</strong><span>Updated {formatAge(snapshot.generatedAt, now)}</span></div></div>
    <section className="hosts" aria-label="Compute workers">
      <HostBand host={snapshot.hosts.nvidia} now={now} />
      <HostBand host={snapshot.hosts.amd} now={now} />
    </section>
    <div className="work-grid">
      <MetricChart run={displayedRun} isHistorical={!activeRun && Boolean(displayedRun)} />
      <QueueSection snapshot={snapshot} />
    </div>
    <RecentRuns snapshot={snapshot} />
  </div>;
}

type ResourceState<T> = { data: T | null; loading: boolean; error: string | null };

function useResource<T>(path: string, enabled: boolean): ResourceState<T> {
  const [state, setState] = useState<ResourceState<T>>({ data: null, loading: false, error: null });
  useEffect(() => {
    if (!enabled || state.data) return;
    const controller = new AbortController();
    setState(current => ({ ...current, loading: true, error: null }));
    void fetch(path, { signal: controller.signal }).then(async response => {
      if (!response.ok) throw new Error(`Local gateway returned ${response.status}`);
      return await response.json() as T;
    }).then(data => setState({ data, loading: false, error: null })).catch(reason => {
      if (reason instanceof DOMException && reason.name === "AbortError") return;
      setState({ data: null, loading: false, error: reason instanceof Error ? reason.message : String(reason) });
    });
    return () => controller.abort();
  }, [enabled, path, state.data]);
  return state;
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? value as Record<string, unknown> : {};
}

function number(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function evidenceModel(envelope: ResourceEnvelope<EvidencePayload> | null): EvidenceViewModel | null {
  if (!envelope?.data || envelope.source.status === "rejected") return null;
  const baseline = record(envelope.data.baseline);
  const candidate = record(envelope.data.candidates[0]);
  const baselineMetrics = record(baseline.metrics);
  const candidateMetrics = record(candidate.metrics);
  const baselineSeeds = Array.isArray(baseline.seedMetrics) ? baseline.seedMetrics.map(record) : [];
  const candidateSeeds = Array.isArray(candidate.seedMetrics) ? candidate.seedMetrics.map(record) : [];
  const candidateBySeed = new Map(candidateSeeds.map(item => [number(item.seed), number(item.wholeTumorDice)]));
  const selectedModel = record(envelope.data.selectedModel);
  const promotion = record(envelope.data.promotion);
  const comparison = record(envelope.data.comparison);
  const effectLabels: Record<string, string> = { whole_tumor_dice: "Whole-tumor Dice", tumor_core_dice: "Tumor-core Dice", enhancing_tumor_dice: "Enhancing-tumor Dice", whole_tumor_box_iou: "Whole-tumor box IoU", whole_tumor_hd95_mm: "Whole-tumor HD95 (mm)" };
  const effects = Array.isArray(comparison.effects) ? comparison.effects.map(record).flatMap(item => {
    const interval = Array.isArray(item.ci95) ? item.ci95.map(number) : [];
    const metricId = String(item.metricId || "");
    const effect = number(item.delta);
    if (effect == null || interval.length !== 2 || interval[0] == null || interval[1] == null) return [];
    return [{ label: effectLabels[metricId] || metricId, delta: effect, ci95: [interval[0], interval[1]] as [number, number], higherIsBetter: item.direction !== "lower_is_better", exploratory: item.exploratory === true }];
  }) : [];
  return {
    studyId: envelope.data.studyId,
    protocol: envelope.data.protocol,
    architecture: String(selectedModel.architecture || "nnU-Net residual encoder U-Net"),
    trainer: String(selectedModel.trainer || baseline.trainer || "Not supplied"),
    dataset: String(selectedModel.dataset || "BraTS internal development split · fold 0"),
    scope: envelope.data.evaluationScope.replaceAll("_", " "),
    generatedAt: envelope.generatedAt,
    freshness: envelope.source.status,
    baselineLabel: String(baseline.variantId || "Baseline"),
    candidateLabel: String(candidate.variantId || "Candidate"),
    metrics: [
      { label: "Whole-tumor Dice", baseline: number(baselineMetrics.wholeTumorDice), candidate: number(candidateMetrics.wholeTumorDice), higherIsBetter: true },
      { label: "Tumor-core Dice", baseline: number(baselineMetrics.tumorCoreDice), candidate: number(candidateMetrics.tumorCoreDice), higherIsBetter: true },
      { label: "Enhancing-tumor Dice", baseline: number(baselineMetrics.enhancingTumorDice), candidate: number(candidateMetrics.enhancingTumorDice), higherIsBetter: true },
      { label: "Lowest-quartile whole-tumor Dice", baseline: number(baselineMetrics.smallestQuartileWholeTumorDice), candidate: number(candidateMetrics.smallestQuartileWholeTumorDice), higherIsBetter: true },
      { label: "Mean derived box IoU", baseline: number(baselineMetrics.meanDerivedBoxIou), candidate: number(candidateMetrics.meanDerivedBoxIou), higherIsBetter: true },
      { label: "Mean HD95", unit: "mm", baseline: number(baselineMetrics.meanHd95Mm), candidate: number(candidateMetrics.meanHd95Mm), higherIsBetter: false }
    ],
    effects,
    seeds: baselineSeeds.map(item => ({ seed: number(item.seed) ?? "—", baseline: number(item.wholeTumorDice), candidate: candidateBySeed.get(number(item.seed)) ?? null })),
    gate: { allowed: envelope.data.promotion.status === "selected", missing: envelope.data.promotion.missing, decision: String(promotion.rationale || "Automatic promotion remains disabled.") },
    selectedModel: selectedModel.variantId ? { label: String(selectedModel.variantId), seed: number(selectedModel.seed) ?? "—", readiness: String(selectedModel.readiness || "internal research only").replaceAll("_", " ") } : null,
    provenance: { evidenceContentHash: envelope.artifactDigest, checkpointHash: typeof selectedModel.checkpointSha256 === "string" ? selectedModel.checkpointSha256 : null, sourceRevision: typeof selectedModel.sourceRevision === "string" ? selectedModel.sourceRevision : null, analysisPlanHash: typeof selectedModel.analysisPlanSha256 === "string" ? selectedModel.analysisPlanSha256 : null, schemaVersion: envelope.schemaVersion, generatedAt: envelope.generatedAt }
  };
}

function explanationModel(envelope: ResourceEnvelope<ExplanationPayload> | null): ExplanationViewModel | null {
  if (!envelope?.data || envelope.source.status === "rejected") return null;
  const deterministic = record(envelope.data.deterministic);
  const llmArtifact = record(envelope.data.llm.artifact);
  const llmStatus = envelope.data.llm.status === "validated" ? "valid" : envelope.data.llm.status;
  const limitations = deterministic.limitations;
  const evidence = Array.isArray(deterministic.evidence) ? deterministic.evidence.map(item => record(item)).map(item => ({ field: String(item.field || "unknown"), value: item.value == null || ["string", "number"].includes(typeof item.value) ? item.value as string | number | null : JSON.stringify(item.value) })) : [];
  return {
    deterministicSummary: String(deterministic.summary || "No deterministic summary supplied."),
    deterministicLimitations: Array.isArray(limitations) ? limitations.map(String) : [String(limitations || "No limitations supplied.")],
    llmSummary: typeof llmArtifact.summary === "string" ? llmArtifact.summary : null,
    llmLimitations: Array.isArray(llmArtifact.limitations) ? llmArtifact.limitations.map(String) : [],
    llmStatus,
    fallbackReason: envelope.data.llm.reason,
    evidence,
    provenance: {
      schemaVersion: envelope.schemaVersion,
      exportId: String(llmArtifact.exportId || "No export ID supplied"),
      artifactHash: envelope.artifactDigest || "No artifact hash supplied",
      modelName: typeof llmArtifact.modelName === "string" ? llmArtifact.modelName : null,
      modelDigest: typeof llmArtifact.modelDigest === "string" ? llmArtifact.modelDigest : null,
      generatedAt: envelope.generatedAt
    },
    validationStatus: envelope.data.llm.status === "rejected" ? "rejected" : "valid"
  };
}

function proposalModel(envelope: ResourceEnvelope<ProposalPayload> | null): ProposalsViewModel | null {
  if (!envelope?.data || envelope.source.status === "rejected") return null;
  const jobs = envelope.data.jobs.map(item => record(item)).map(job => ({ runId: String(job.runId || "No run ID"), profile: String(job.profile || "No profile"), state: String(job.state || "unavailable"), reasonCode: String(job.reasonCode || "not_preapproved"), proposalAllowed: job.proposalAllowed === true }));
  const proposals = envelope.data.proposals.map(item => record(item)).map((proposal): JobProposal => {
    const code = String(proposal.reasonCode || "unavailable");
    const state: JobProposal["state"] = code === "exact_preapproved_match" ? "exact-match" : code === "unsafe_request" ? "unsafe" : code === "already_complete" ? "already-complete" : code === "unavailable" ? "unavailable" : "abstained";
    return { runId: typeof proposal.runId === "string" ? proposal.runId : null, profile: typeof proposal.profile === "string" ? proposal.profile : null, state, reason: String(proposal.reason || "No proposal reason supplied."), executed: false };
  });
  return { jobs, proposals };
}

export function App() {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(() => {
    try {
      const cached = localStorage.getItem("brain-mri-monitor.snapshot.v1");
      if (!cached) return null;
      const value = JSON.parse(cached) as DashboardSnapshot;
      return value.schemaVersion === 1 && value.hosts?.nvidia && value.hosts?.amd ? value : null;
    } catch {
      return null;
    }
  });
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(Date.now());
  const [activeView, setActiveView] = useState<ViewKey>(() => viewFromHash());
  const etag = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let poll: number | undefined;
    const load = async () => {
      try {
        const response = await fetch("/api/status", { headers: etag.current ? { "If-None-Match": etag.current } : {} });
        if (response.status === 304) return;
        if (!response.ok) throw new Error(`Local gateway returned ${response.status}`);
        const value = await response.json() as DashboardSnapshot;
        if (value.schemaVersion !== 1 || !value.hosts?.nvidia || !value.hosts?.amd) throw new Error("Unsupported dashboard contract version");
        if (!cancelled) {
          etag.current = response.headers.get("etag");
          setSnapshot(value);
          setError(null);
          try { localStorage.setItem("brain-mri-monitor.snapshot.v1", JSON.stringify(value)); } catch { /* Cache is optional. */ }
        }
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
      } finally {
        if (!cancelled) poll = window.setTimeout(load, Math.max(10_000, snapshot?.pollIntervalMs || 10_000));
      }
    };
    void load();
    const clock = window.setInterval(() => setNow(Date.now()), 5_000);
    return () => { cancelled = true; if (poll) clearTimeout(poll); clearInterval(clock); };
  }, [snapshot?.pollIntervalMs]);

  useEffect(() => {
    const update = () => setActiveView(viewFromHash());
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);

  const evidence = useResource<ResourceEnvelope<EvidencePayload>>("/api/evidence", activeView === "evidence");
  const explanation = useResource<ResourceEnvelope<ExplanationPayload>>("/api/explanation", activeView === "explanations");
  const proposals = useResource<ResourceEnvelope<ProposalPayload>>("/api/proposals", activeView === "proposals");
  const reporting = snapshot ? ([snapshot.hosts.nvidia, snapshot.hosts.amd]).filter(host => host.reachable && !host.stale).length : 0;

  return <div className="shell">
    <a className="skip-link" href="#main-content">Skip to current view</a>
    <header className="masthead">
      <div className="product-identity"><strong>Brain MRI workspace</strong></div>
      <div className="network-summary"><strong>{snapshot ? `${reporting}/2 workers online` : "Checking workers"}</strong></div>
    </header>
    <div className="prototype-notice" role="note"><strong>Research use only</strong><span>Not for diagnosis or treatment.</span></div>
    <nav className="view-nav" aria-label="Research workspace">
      <div className="view-nav__links">{views.map(view => <a key={view.key} href={`#${view.key}`} aria-current={activeView === view.key ? "page" : undefined}>{view.label}</a>)}</div>
      <label className="view-nav__select"><span>Current view</span><select value={activeView} onChange={event => { window.location.hash = event.target.value; }}>{views.map(view => <option key={view.key} value={view.key}>{view.label}</option>)}</select></label>
    </nav>
    {error ? <div className="gateway-banner" role="status"><strong>Local gateway update failed.</strong><span>{snapshot ? "Last-known values remain visible." : "No cached snapshot is available."} {error}</span></div> : null}
    <main id="main-content" tabIndex={-1}>
      <Suspense fallback={<Loading />}>
        {activeView === "study" ? <StudyView /> : null}
        {activeView === "operations" ? snapshot ? <OverviewView snapshot={snapshot} now={now} /> : <Loading /> : null}
        {activeView === "evidence" ? evidence.loading ? <Loading /> : evidence.error ? <div className="view-error" role="alert"><strong>Model evidence unavailable</strong><span>{evidence.error}</span></div> : <EvidenceView resource={evidenceModel(evidence.data)} /> : null}
        {activeView === "explanations" ? explanation.loading ? <Loading /> : explanation.error ? <div className="view-error" role="alert"><strong>Explanation unavailable</strong><span>{explanation.error}</span></div> : <ExplanationsView explanation={explanationModel(explanation.data)} /> : null}
        {activeView === "proposals" ? proposals.loading ? <Loading /> : proposals.error ? <div className="view-error" role="alert"><strong>Proposal data unavailable</strong><span>{proposals.error}</span></div> : <ProposalsView data={proposalModel(proposals.data)} /> : null}
      </Suspense>
    </main>
    <footer><span>Local gateway · fixed research model · no scan access for the LLM</span><span>{snapshot ? `Operations collection every ${snapshot.pollIntervalMs / 1000}s` : "Operations collection unavailable"} · generated {snapshot ? formatTimestamp(snapshot.generatedAt) : "not yet"}</span></footer>
  </div>;
}
