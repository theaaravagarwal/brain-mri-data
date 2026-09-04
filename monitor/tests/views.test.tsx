import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import EvidenceView from "../src/views/EvidenceView";
import ExplanationsView from "../src/views/ExplanationsView";
import ProposalsView from "../src/views/ProposalsView";
import StudyView from "../src/views/StudyView";

const capabilities = {
  schemaVersion: "research-study-capabilities/v1",
  generatedAt: "2026-09-01T00:00:00Z",
  inference: { status: "ready", modelId: "glioma-segresnet-20260828", modelScope: "internal_research_only", checkpointSha256: "1".repeat(64), observedCheckpointSha256: "1".repeat(64), outputKind: "binary_whole_lesion_research_segmentation", device: "NVIDIA GeForce RTX 4060" },
  explanation: { deterministic: "available", llm: "not_configured", model: null },
  demoAvailable: true,
  limits: { files: 4, perFileBytes: 536870912, totalBytes: 2147483648, retentionHours: 24 }
};

describe("new study workflow", () => {
  it("offers a retry when the initial capability check fails", async () => {
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new TypeError("gateway offline"))
      .mockResolvedValueOnce(new Response(JSON.stringify(capabilities), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    render(<StudyView />);
    const retry = await screen.findByRole("button", { name: "Try connection again" });
    expect(screen.getByText("Connection check failed")).toBeInTheDocument();
    fireEvent.click(retry);
    expect(await screen.findByText("Ready to run")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Try connection again" })).not.toBeInTheDocument();
  });

  it("requires four named volumes and exposes no scan preview", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(capabilities), { status: 200, headers: { "Content-Type": "application/json" } })));
    const { container } = render(<StudyView />);
    expect(await screen.findByText("Ready to run")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Try with sample scans" })).toBeEnabled();
    expect(container.querySelector("img, canvas, video")).toBeNull();
    const inputs = container.querySelectorAll<HTMLInputElement>('input[type="file"]');
    expect(inputs).toHaveLength(4);
    inputs.forEach((input, index) => fireEvent.change(input, { target: { files: [new File([`volume-${index}`], `volume-${index}.nii.gz`, { type: "application/gzip" })] } }));
    expect(screen.getByRole("button", { name: "Check my files" })).toBeEnabled();
    expect(screen.getByText(/MiB selected/)).toBeInTheDocument();
  });

  it("shows validated metadata and a deterministic fallback result", async () => {
    const validated = {
      schemaVersion: "research-study-job/v1", jobId: "65ecf1c3-ae23-4c40-ae7f-6aecc9453904", state: "validated", createdAt: "2026-09-01T00:00:00Z", updatedAt: "2026-09-01T00:00:00Z", expiresAt: "2026-09-02T00:00:00Z",
      validation: { schema_version: "research-study-validation/v1", status: "pass", modality_count: 4, modalities: ["t1", "t1ce", "t2", "flair"], geometry_match: true, shape: [240, 240, 155], spacing_mm: [1, 1, 1], geometry_sha256: "a".repeat(64), modality_sha256: { t1: "b".repeat(64), t1ce: "c".repeat(64), t2: "d".repeat(64), flair: "e".repeat(64) } },
      result: null, explanation: null, error: null, artifacts: []
    };
    const succeeded = {
      ...validated, state: "succeeded", result: { schema_version: "research-segmentation-result/v1", disclaimer: "Research output only; not a diagnosis or treatment recommendation.", input_qc: validated.validation, segmentation: { status: "complete", output_sha256: "f".repeat(64), output_shape: [240, 240, 155], geometry_preserved: true, labels: [0, 1], label_count: 2, nonzero_voxels: 42117 }, provenance: { model_id: "glioma-segresnet-20260828", model_scope: "internal_research_only", checkpoint_sha256: "1".repeat(64), training_git_revision: "2".repeat(40), study_sha256: "3".repeat(64), profile_sha256: "4".repeat(64), trainer_sha256: "5".repeat(64), inference_script_sha256: "6".repeat(64), device: "NVIDIA GeForce RTX 4060", torch_version: "2.9.1", monai_version: "1.6.0", nibabel_version: "5.4.2", generated_at: "2026-09-01T00:00:00Z" } },
      explanation: { schema_version: "research-segmentation-explanation/v1", deterministic: { disclaimer: "Research output only; not a diagnosis or treatment recommendation.", summary: "The fixed research model completed a geometry-preserving binary segmentation with 42117 non-zero output voxels.", evidence: [], limitations: "No reference mask was supplied, so accuracy cannot be determined.", abstained: false }, llm: { status: "unavailable", artifact: null, reason: "No local model was configured.", model_name: null, model_digest: null } }, artifacts: ["segmentation", "receipt", "explanation"]
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(capabilities), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(validated), { status: 201 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ...validated, state: "running" }), { status: 202 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(succeeded), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const { container } = render(<StudyView />);
    await screen.findByText("Ready to run");
    container.querySelectorAll<HTMLInputElement>('input[type="file"]').forEach((input, index) => fireEvent.change(input, { target: { files: [new File([`v-${index}`], `v-${index}.nii.gz`, { type: "application/gzip" })] } }));
    fireEvent.click(screen.getByRole("button", { name: "Check my files" }));
    expect(await screen.findByText("240 × 240 × 155 voxels")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Create outline" }));
    await waitFor(() => expect(screen.getByText("Research outline ready")).toBeInTheDocument(), { timeout: 3_000 });
    expect(screen.getByText("Showing the checked facts.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Download technical receipt" })).toBeInTheDocument();
  });
});

describe("evidence view safety and completeness", () => {
  it("renders promotion blocked and missing evidence", () => {
    render(<EvidenceView resource={{ studyId: "study-1", scope: "internal validation", seeds: [{ seed: 1, baseline: .7, candidate: .71 }], gate: { allowed: false, missing: ["3 seeds per arm", "paired confidence interval"] } }} />);
    expect(screen.getAllByText("Promotion blocked").length).toBeGreaterThan(0);
    expect(screen.getByText("3 seeds per arm")).toBeInTheDocument();
    expect(screen.getByText("paired confidence interval")).toBeInTheDocument();
    expect(screen.getByRole("table", { name: /seed-level results/i })).toBeInTheDocument();
  });

  it("renders a rejected candidate and retained internal model", () => {
    render(<EvidenceView resource={{ studyId: "glioma", scope: "internal validation", selectedModel: { label: "baseline", seed: 20260821, readiness: "internal research only" }, seeds: [{ seed: 20260821, baseline: .9063, candidate: .9043 }, { seed: 20260822, baseline: .9004, candidate: .9007 }, { seed: 20260823, baseline: .9039, candidate: .9034 }], effects: [{ label: "Whole-tumor Dice", delta: -.0007, ci95: [-.0020, .0005], higherIsBetter: true, exploratory: false }], gate: { allowed: false, missing: ["independent external evaluation"], decision: "Candidate rejected; baseline retained." } }} />);
    expect(screen.getByText("Candidate rejected", { selector: "span" })).toBeInTheDocument();
    expect(screen.getByText(/baseline · seed 20260821/)).toBeInTheDocument();
    expect(screen.getByText("internal research only")).toBeInTheDocument();
    expect(screen.getByRole("table", { name: /case-cluster bootstrap intervals/i })).toBeInTheDocument();
  });

  it("renders a semantic empty state for null data", () => {
    render(<EvidenceView resource={null} />);
    expect(screen.getByRole("heading", { name: "Model evidence" })).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(/Evidence unavailable/);
  });
});

describe("explanations view", () => {
  const explanation = { deterministicSummary: "Dice is reported for internal validation.", deterministicLimitations: ["Not externally tested."], llmSummary: "Should never appear", llmStatus: "rejected" as const, fallbackReason: "Schema validation failed", evidence: [{ field: "meanDice", value: .71 }], provenance: { schemaVersion: "ResearchRunSummaryEnvelopeV1", exportId: "exp-1", artifactHash: "sha256:abc", generatedAt: "2026-08-23T00:00:00Z" }, validationStatus: "valid" as const };

  it("keeps deterministic facts and omits rejected LLM prose", () => {
    render(<ExplanationsView explanation={explanation} />);
    expect(screen.getByText("Research output only; not a diagnosis or treatment recommendation.")).toBeInTheDocument();
    expect(screen.getByText("Dice is reported for internal validation.")).toBeInTheDocument();
    expect(screen.queryByText("Should never appear")).not.toBeInTheDocument();
    expect(screen.getByText("LLM explanation unavailable — deterministic facts shown")).toBeInTheDocument();
    expect(screen.getByRole("table", { name: /evidence fields/i })).toBeInTheDocument();
  });

  it("renders null state", () => {
    render(<ExplanationsView explanation={null} />);
    expect(screen.getByRole("heading", { name: "Explanations" })).toBeInTheDocument();
    expect(screen.getByText("No explanation available")).toBeInTheDocument();
  });
});

describe("proposals view", () => {
  it("shows executed false and has no execution control", () => {
    render(<ProposalsView data={{ jobs: [{ runId: "run-1", profile: "profile-a", state: "frozen", reasonCode: "ready", proposalAllowed: true }], proposals: [{ runId: "run-1", profile: "profile-a", state: "exact-match", reason: "Exact pre-approved match.", executed: false }] }} />);
    expect(screen.getByText("executed: false")).toBeInTheDocument();
    expect(screen.getByRole("table", { name: /frozen jobs/i })).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
    expect(screen.getByText("exact-match")).toBeInTheDocument();
  });

  it("shows abstained proposal state and null state", () => {
    const { rerender } = render(<ProposalsView data={{ jobs: [], proposals: [{ runId: null, profile: null, state: "abstained", reason: "Ambiguous request.", executed: false }] }} />);
    expect(screen.getByText("abstained")).toBeInTheDocument();
    expect(screen.getByText(/no exact match/i)).toBeInTheDocument();
    rerender(<ProposalsView data={null} />);
    expect(screen.getByText("No proposal data")).toBeInTheDocument();
  });
});
