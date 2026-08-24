import { useEffect, useMemo, useRef, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import type { DashboardSnapshot, HostEnvelope, MetricPoint, RecentRun, RunSnapshot } from "./types";

type MetricKey = "meanDice" | "meanBoxIou" | "meanHd95Mm";

const metrics: Record<MetricKey, { label: string; unit: string; color: string; domain: [number | "auto", number | "auto"] }> = {
  meanDice: { label: "Dice", unit: "", color: "var(--active)", domain: [0, 1] },
  meanBoxIou: { label: "Box IoU", unit: "", color: "var(--complete)", domain: [0, 1] },
  meanHd95Mm: { label: "HD95", unit: " mm", color: "var(--warning)", domain: [0, "auto"] }
};

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
  return <article className={`host-band ${host.stale ? "host-band--stale" : ""}`}>
    <div className="host-identity">
      <div className="host-title"><h2>{data.label}</h2><State host={host} /></div>
      <p>{data.role} · {data.hostname}</p>
      <p className="freshness">Confirmed {formatAge(host.lastSuccessAt, now)} · {host.latencyMs ?? "—"} ms</p>
      {host.error && <p className="inline-error">{host.error}</p>}
    </div>
    <dl className="host-measures">
      <Measure label="GPU" value={data.gpu.utilizationPercent == null ? (data.gpu.active ? "Active" : "Idle") : `${formatNumber(data.gpu.utilizationPercent, 0)}%`} detail={data.gpu.name} />
      <Measure label="VRAM" value={gpuMemory} detail={data.gpu.memoryTotalMib ? `${formatNumber(data.gpu.memoryTotalMib / 1024, 0)} GiB total` : undefined} />
      <Measure label="Disk free" value={`${formatNumber(data.disk.freeGib, 0)} GiB`} detail={`${formatNumber((data.disk.usedGib / data.disk.totalGib) * 100, 0)}% used`} />
      <Measure label="RAM" value={data.memory.usedGib == null ? "—" : `${formatNumber(data.memory.usedGib)} GiB`} detail={data.memory.totalGib == null ? undefined : `${formatNumber(data.memory.totalGib, 0)} GiB total`} />
    </dl>
    <div className="host-work">
      <div className="host-work__line"><span>{progress ? shortRun(progress.name) : "No active training run"}</span><strong>{progress?.phase?.replaceAll("_", " ") || data.queue.state}</strong></div>
      <div className="progress-track" role="progressbar" aria-label={`${data.label} progress`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent == null ? undefined : Math.round(percent)}>
        <span style={{ transform: `scaleX(${percent == null ? 0 : Math.min(percent, 100) / 100})` }} />
      </div>
      <div className="host-work__meta">
        {progress ? <span>Epoch {progress.epoch ?? "—"}/{progress.epochs ?? "—"}{progress.completed != null && progress.total ? ` · ${progress.completed}/${progress.total} ${progress.unit}s` : ""}</span> : <span>{data.queue.detail || "Queue empty"}</span>}
        <span>{progress?.liveLoss != null ? `Live loss ${formatNumber(progress.liveLoss, 4)}` : `${data.sessions.length} session${data.sessions.length === 1 ? "" : "s"}`}</span>
      </div>
    </div>
  </article>;
}

function MetricChart({ run, isHistorical }: { run: RunSnapshot | null; isHistorical: boolean }) {
  const [metric, setMetric] = useState<MetricKey>("meanDice");
  const definition = metrics[metric];
  const latest = run?.metrics.at(-1);
  return <section className="metric-section" aria-labelledby="metric-title">
    <div className="section-heading">
      <div><h2 id="metric-title">Validation history</h2><p>{run ? `${shortRun(run.name)}${isHistorical ? " · completed" : " · live"}` : "No completed training run available"}</p></div>
      <div className="metric-tabs" role="tablist" aria-label="Metric">
        {(Object.keys(metrics) as MetricKey[]).map(key => <button key={key} type="button" role="tab" aria-selected={metric === key} onClick={() => setMetric(key)}>{metrics[key].label}</button>)}
      </div>
    </div>
    <div className="metric-latest">
      <strong>{formatNumber(latest?.[metric], metric === "meanHd95Mm" ? 1 : 4)}{latest?.[metric] != null ? definition.unit : ""}</strong>
      <span>{latest ? `Epoch ${latest.epoch}` : "No completed validation epoch"}</span>
    </div>
    <div className="chart" aria-label={`${definition.label} by epoch`}>
      {run?.metrics.length ? <ResponsiveContainer width="100%" height="100%">
        <LineChart data={run.metrics} margin={{ top: 8, right: 10, bottom: 0, left: -12 }}>
          <CartesianGrid vertical={false} stroke="var(--rule)" />
          <XAxis dataKey="epoch" tickLine={false} axisLine={false} tick={{ fill: "var(--muted)", fontSize: 12 }} />
          <YAxis domain={definition.domain} tickLine={false} axisLine={false} tick={{ fill: "var(--muted)", fontSize: 12 }} width={52} />
          <Tooltip animationDuration={0} cursor={{ stroke: "var(--rule-strong)" }} contentStyle={{ background: "var(--surface)", border: "1px solid var(--rule-strong)", borderRadius: 8, color: "var(--ink)" }} formatter={(value) => [`${formatNumber(Number(value), metric === "meanHd95Mm" ? 1 : 4)}${definition.unit}`, definition.label]} labelFormatter={label => `Epoch ${label}`} />
          <Line type="monotone" dataKey={metric} stroke={definition.color} strokeWidth={2} dot={{ r: 2, fill: definition.color, strokeWidth: 0 }} activeDot={{ r: 4 }} isAnimationActive={false} connectNulls={false} />
        </LineChart>
      </ResponsiveContainer> : <div className="empty-chart"><span>No metric series yet.</span><small>The chart appears after a validation epoch is recorded.</small></div>}
    </div>
    <div className="metric-strip">
      <Measure label="Train loss" value={formatNumber(latest?.trainLoss, 4)} />
      <Measure label="Dice" value={formatNumber(latest?.meanDice, 4)} />
      <Measure label="Box IoU" value={formatNumber(latest?.meanBoxIou, 4)} />
      <Measure label="HD95" value={latest?.meanHd95Mm == null ? "—" : `${formatNumber(latest.meanHd95Mm, 1)} mm`} />
    </div>
  </section>;
}

function QueueSection({ snapshot }: { snapshot: DashboardSnapshot }) {
  const entries = (["nvidia", "amd"] as const).map(id => snapshot.hosts[id]).filter(host => host.data);
  return <section className="queue-section" aria-labelledby="queue-title">
    <div className="section-heading"><div><h2 id="queue-title">Queues and sessions</h2><p>Read-only runtime state</p></div></div>
    <div className="queue-list">
      {entries.map(host => <div className="queue-row" key={host.data!.id}>
        <div><strong>{host.data!.label}</strong><span>{host.data!.queue.detail || "No queued work"}</span></div>
        <span className={`queue-state queue-state--${host.data!.queue.state}`}>{host.data!.queue.state}</span>
        <div className="session-list">{host.data!.sessions.length ? host.data!.sessions.map(session => <code key={session}>{session}</code>) : <span>No tmux sessions</span>}</div>
      </div>)}
    </div>
  </section>;
}

function RunRow({ host, run }: { host: string; run: RecentRun }) {
  return <tr><td><span className="host-code">{host}</span></td><td title={run.name}>{shortRun(run.name)}</td><td>{run.seed ?? "—"}</td><td>{run.bestEpoch ?? "—"}</td><td>{formatNumber(run.bestDice, 4)}</td><td>{formatNumber(run.bestBoxIou, 4)}</td><td>{run.bestHd95Mm == null ? "—" : `${formatNumber(run.bestHd95Mm, 1)} mm`}</td><td><span className={`run-status run-status--${run.status}`}>{run.status}</span></td></tr>;
}

function RecentRuns({ snapshot }: { snapshot: DashboardSnapshot }) {
  const rows = (["nvidia", "amd"] as const).flatMap(id => (snapshot.hosts[id].data?.recentRuns || []).map(run => ({ host: id, run }))).sort((a, b) => Date.parse(b.run.modifiedAt) - Date.parse(a.run.modifiedAt)).slice(0, 10);
  return <section className="runs-section" aria-labelledby="runs-title">
    <div className="section-heading"><div><h2 id="runs-title">Recent runs</h2><p>Best recorded validation epoch per run</p></div></div>
    <div className="table-scroll"><table><thead><tr><th>Host</th><th>Run</th><th>Seed</th><th>Best epoch</th><th>Dice</th><th>Box IoU</th><th>HD95</th><th>Status</th></tr></thead><tbody>{rows.length ? rows.map(({ host, run }) => <RunRow key={`${host}-${run.name}`} host={host} run={run} />) : <tr><td colSpan={8} className="empty-cell">No recorded runs available.</td></tr>}</tbody></table></div>
  </section>;
}

function Loading() {
  return <main className="shell" aria-busy="true"><header className="masthead"><div><h1>Training monitor</h1><p>Connecting to workers…</p></div></header><div className="loading-lines"><span /><span /><span /></div></main>;
}

export function App() {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(Date.now());
  const etag = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await fetch("/api/status", { headers: etag.current ? { "If-None-Match": etag.current } : {} });
        if (response.status === 304) return;
        if (!response.ok) throw new Error(`Local gateway returned ${response.status}`);
        const value = await response.json() as DashboardSnapshot;
        if (!cancelled) {
          etag.current = response.headers.get("etag");
          setSnapshot(value);
          setError(null);
        }
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
      }
    };
    void load();
    const poll = window.setInterval(load, 5_000);
    const clock = window.setInterval(() => setNow(Date.now()), 1_000);
    return () => { cancelled = true; clearInterval(poll); clearInterval(clock); };
  }, []);

  const activeRun = useMemo(() => snapshot?.hosts.nvidia.data?.activeRun || snapshot?.hosts.amd.data?.activeRun || null, [snapshot]);
  const displayedRun = useMemo(() => activeRun || snapshot?.hosts.nvidia.data?.latestRun || snapshot?.hosts.amd.data?.latestRun || null, [activeRun, snapshot]);
  if (!snapshot) return <Loading />;
  const reporting = ([snapshot.hosts.nvidia, snapshot.hosts.amd]).filter(host => host.reachable && !host.stale).length;

  return <main className="shell">
    <header className="masthead">
      <div><h1>Training monitor</h1><p>Two private workers · read-only</p></div>
      <div className="network-summary"><strong>{reporting}/2 reporting</strong><span>Updated {formatAge(snapshot.generatedAt, now)}</span>{error && <span className="gateway-error">Gateway: {error}</span>}</div>
    </header>

    <section className="hosts" aria-label="Compute workers">
      <HostBand host={snapshot.hosts.nvidia} now={now} />
      <HostBand host={snapshot.hosts.amd} now={now} />
    </section>

    <div className="work-grid">
      <MetricChart run={displayedRun} isHistorical={!activeRun && Boolean(displayedRun)} />
      <QueueSection snapshot={snapshot} />
    </div>

    <RecentRuns snapshot={snapshot} />
    <footer><span>Local gateway · fixed Tailscale SSH targets</span><span>Host polling every {snapshot.pollIntervalMs / 1000}s · stale values remain visible</span></footer>
  </main>;
}
