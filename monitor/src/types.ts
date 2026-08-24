export type MetricPoint = {
  epoch: number;
  trainLoss: number | null;
  meanDice: number | null;
  meanBoxIou: number | null;
  meanHd95Mm: number | null;
};

export type RunSnapshot = {
  name: string;
  status: "running" | "complete" | "idle" | "failed" | "unknown";
  arm: string | null;
  seed: number | null;
  profile: string | null;
  epoch: number | null;
  epochs: number | null;
  phase: string | null;
  completed: number | null;
  total: number | null;
  unit: "batch" | "case" | null;
  liveLoss: number | null;
  updatedAt: string | null;
  metrics: MetricPoint[];
};

export type RecentRun = {
  name: string;
  seed: number | null;
  status: string;
  bestEpoch: number | null;
  bestDice: number | null;
  bestBoxIou: number | null;
  bestHd95Mm: number | null;
  modifiedAt: string;
};

export type HostData = {
  id: "nvidia" | "amd";
  label: string;
  role: string;
  hostname: string;
  gpu: {
    name: string;
    utilizationPercent: number | null;
    memoryUsedMib: number | null;
    memoryTotalMib: number | null;
    temperatureC: number | null;
    powerW: number | null;
    active: boolean;
  };
  memory: { usedGib: number | null; totalGib: number | null };
  disk: { usedGib: number; totalGib: number; freeGib: number };
  activeRun: RunSnapshot | null;
  latestRun: RunSnapshot | null;
  recentRuns: RecentRun[];
  sessions: string[];
  queue: { state: string; detail: string | null };
};

export type HostEnvelope = {
  reachable: boolean;
  stale: boolean;
  lastSuccessAt: string | null;
  staleSince: string | null;
  latencyMs: number | null;
  nextAttemptAt: string;
  error: string | null;
  data: HostData | null;
};

export type DashboardSnapshot = {
  schemaVersion: 1;
  generatedAt: string;
  pollIntervalMs: number;
  hosts: { nvidia: HostEnvelope; amd: HostEnvelope };
};
