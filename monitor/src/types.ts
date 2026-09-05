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
  queue: {
    schemaVersion?: "research-training-queue/v1";
    state: "waiting" | "running" | "complete" | "attention";
    detail: string | null;
    serviceState: string;
    currentRun: string | null;
    queuedRuns: string[];
    completedCount: number;
    totalCount: number;
    failedCount: number;
    updatedAt: string | null;
    lastError: string | null;
  };
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

export type ResourceEnvelope<T> = {
  schemaVersion: string;
  generatedAt: string;
  source: { status: "fresh" | "stale" | "unavailable" | "rejected"; staleSince: string | null; error: string | null };
  artifactDigest: string | null;
  data: T | null;
};

export type EvidenceResource = {
  studyId: "glioma";
  protocol: "glioma_4seq_v1";
  evaluationScope: string;
  reviewStatus: string;
  automaticPromotion: false;
  promotion: { status: "blocked" | "pending" | "rejected" | "selected"; missing: string[] };
  selectedModel?: Record<string, unknown>;
  baseline: Record<string, unknown>;
  candidates: Record<string, unknown>[];
  comparison?: Record<string, unknown>;
};

export type ExplanationResource = {
  deterministic: Record<string, unknown>;
  llm: { status: "validated" | "abstained" | "rejected" | "unavailable"; artifact: Record<string, unknown> | null; reason: string | null };
};

export type ProposalResource = {
  jobs: Record<string, unknown>[];
  proposals: Record<string, unknown>[];
};

export type StudyCapabilities = {
  schemaVersion: "research-study-capabilities/v1";
  generatedAt: string;
  inference: {
    status: "ready" | "unavailable" | "digest_mismatch";
    modelId: string;
    modelScope: "internal_research_only";
    checkpointSha256: string;
    observedCheckpointSha256: string | null;
    outputKind: string;
    device: string;
  };
  explanation: {
    deterministic: "available";
    llm: "configured" | "not_configured";
    model: string | null;
  };
  demoAvailable: boolean;
  evaluationDemoAvailable: boolean;
  externalBenchmark?: null | {
    schema_version: "fixed-segresnet-external-status/v1" | "fixed-segresnet-external-summary/v1";
    benchmark_id: string;
    status: "running" | "complete";
    completed_cases?: number;
    total_cases?: number;
    case_count?: number;
    expected_case_count?: number;
    metrics?: Record<string, {
      n: number;
      mean: number;
      median: number;
      mean_ci95: [number, number];
    }>;
    failures?: { empty_prediction_count: number; hd95_unavailable_count: number; case_error_count: number };
    latency_seconds?: { median: number; p95: number };
    provenance?: { model_id: string; checkpoint_sha256: string; generated_at: string };
  };
  limits: { files: 5; perFileBytes: number; totalBytes: number; retentionHours: number };
};

export type StudyValidation = {
  schema_version: "research-study-validation/v1";
  status: "pass";
  modality_count: 4;
  modalities: ["t1", "t1ce", "t2", "flair"];
  geometry_match: true;
  shape: [number, number, number];
  spacing_mm: [number, number, number];
  geometry_sha256: string;
  modality_sha256: Record<"t1" | "t1ce" | "t2" | "flair", string>;
  reference_mask?: {
    status: "pass";
    geometry_match: true;
    sha256: string;
    labels: Array<0 | 1 | 2 | 3 | 4>;
    nonzero_voxels: number;
  } | null;
};

export type StudyExplanation = {
  schema_version: "research-segmentation-explanation/v1";
  deterministic: {
    disclaimer: string;
    summary: string;
    evidence: Array<{ field: string; value: string | number | boolean | null }>;
    limitations: string;
    abstained: false;
  };
  llm: {
    status: "validated" | "rejected" | "unavailable";
    artifact: null | {
      disclaimer: string;
      summary: string;
      evidence: Array<{ field: string; value: string | number | boolean | null }>;
      limitations: string;
      abstained: false;
    };
    reason: string | null;
    model_name: string | null;
    model_digest: string | null;
  };
};

export type StudyJob = {
  progress?: { stage: string; startedAt: string } | null;
  accessToken?: string;
  viewing?: { volumes: string[]; outlineCenterMm: [number, number, number] | null } | null;
  schemaVersion: "research-study-job/v1";
  jobId: string;
  state: "validated" | "running" | "succeeded" | "failed";
  createdAt: string;
  updatedAt: string;
  expiresAt: string;
  evaluationSampleScope?: "development_validation" | "external_public" | null;
  validation: StudyValidation | null;
  result: null | {
    schema_version: "research-segmentation-result/v1";
    disclaimer: string;
    input_qc: StudyValidation;
    segmentation: {
      status: "complete";
      output_sha256: string;
      output_shape: [number, number, number];
      geometry_preserved: true;
      labels: [0, 1];
      label_count: 2;
      nonzero_voxels: number;
    };
    evaluation?: {
      status: "complete";
      scope: "single_user_supplied_reference";
      whole_lesion_dice: number;
      whole_lesion_iou: number;
      precision: number;
      recall: number;
      hd95_mm: number | null;
      true_positive_voxels: number;
      false_positive_voxels: number;
      false_negative_voxels: number;
    } | null;
    provenance: {
      model_id: string;
      model_scope: "internal_research_only";
      checkpoint_sha256: string;
      training_git_revision: string;
      study_sha256: string;
      profile_sha256: string;
      trainer_sha256: string;
      inference_script_sha256: string;
      device: string;
      torch_version: string;
      monai_version: string;
      nibabel_version: string;
      generated_at: string;
    };
  };
  explanation: StudyExplanation | null;
  error: string | null;
  artifacts: Array<"segmentation" | "receipt" | "explanation">;
};
