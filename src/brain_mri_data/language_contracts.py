"""Strict, direct-identifier-free contracts for the research language boundary."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

DISCLAIMER = "Research output only; not a diagnosis or treatment recommendation."
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,127}$"
FORBIDDEN_TEXT_PATTERN = re.compile(
    r"(?:[/\\]|\b(?:patient|subject|case|dicom|mrn|dob|name|scan|image|mask|"
    r"diagnos(?:e|is|ed)|treatment|medication|chemotherapy|surgery)\b|"
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d+(?:\.\d+){3,}\b)",
    re.IGNORECASE,
)

SafeIdentifier = Annotated[
    str, StringConstraints(pattern=SAFE_IDENTIFIER_PATTERN, strict=True)
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$", strict=True)]
Uuid4String = Annotated[
    str, StringConstraints(strict=True, min_length=36, max_length=36)
]
UnitMetric = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
NonnegativeMetric = Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]


class StrictModel(BaseModel):
    """Base contract that rejects coercion and unknown fields."""

    model_config = ConfigDict(extra="forbid", strict=True)


class RunProvenance(StrictModel):
    git_revision: Annotated[
        str, StringConstraints(pattern=r"^[0-9a-f]{40}$", strict=True)
    ]
    study_sha256: Sha256
    profile_sha256: Sha256
    checkpoint_sha256: Sha256


class AggregateMetrics(StrictModel):
    overall_mean_dice: UnitMetric
    smallest_quartile_mean_dice: UnitMetric
    mean_derived_box_iou: UnitMetric
    mean_hd95_mm: NonnegativeMetric


class MetricDeltas(StrictModel):
    overall_mean_dice: Annotated[
        float, Field(strict=True, ge=-1.0, le=1.0, allow_inf_nan=False)
    ]
    smallest_quartile_mean_dice: Annotated[
        float, Field(strict=True, ge=-1.0, le=1.0, allow_inf_nan=False)
    ]
    mean_derived_box_iou: Annotated[
        float, Field(strict=True, ge=-1.0, le=1.0, allow_inf_nan=False)
    ]
    mean_hd95_mm: Annotated[float, Field(strict=True, allow_inf_nan=False)]


class ScreenGates(StrictModel):
    smallest_quartile_improves_by_at_least_0_02: bool
    overall_dice_declines_by_no_more_than_0_005: bool


class BaselineResult(StrictModel):
    variant_id: Literal["uniform"]
    seed: int = Field(strict=True, ge=0)
    best_epoch: int = Field(strict=True, ge=1)
    foreground_probability: Literal[0.0]
    metrics: AggregateMetrics
    provenance: RunProvenance


class CandidateResult(StrictModel):
    variant_id: SafeIdentifier
    seed: int = Field(strict=True, ge=0)
    best_epoch: int = Field(strict=True, ge=1)
    foreground_probability: UnitMetric
    metrics: AggregateMetrics
    delta_vs_uniform: MetricDeltas
    screen_gates: ScreenGates
    passes_screen_gate: bool
    provenance: RunProvenance

    @model_validator(mode="after")
    def gates_agree(self) -> CandidateResult:
        expected = all(
            (
                self.screen_gates.smallest_quartile_improves_by_at_least_0_02,
                self.screen_gates.overall_dice_declines_by_no_more_than_0_005,
            )
        )
        if self.passes_screen_gate is not expected:
            raise ValueError("passes_screen_gate does not match deterministic gates")
        return self


class ResearchRunSummaryEnvelopeV1(StrictModel):
    schema_version: Literal["research-run-summary/v1"]
    export_id: Uuid4String
    artifact_kind: Literal["cnn_research_run_summary"]
    study_id: Literal["glioma"]
    protocol: Literal["glioma_4seq_v1"]
    run_group_id: SafeIdentifier
    evaluation_scope: Literal["single_seed_internal_validation_screen"]
    review_status: Literal["human_review_required"]
    automatic_promotion: Literal[False]
    smallest_quartile_case_count: int = Field(strict=True, ge=1)
    source_summary_sha256: Sha256
    baseline: BaselineResult
    candidates: Annotated[list[CandidateResult], Field(min_length=1, max_length=16)]
    eligible_for_human_review: list[SafeIdentifier]

    @field_validator("export_id")
    @classmethod
    def export_id_is_uuid4(cls, value: str) -> str:
        parsed = UUID(value)
        if parsed.version != 4 or str(parsed) != value:
            raise ValueError("export_id must be a canonical UUIDv4")
        return value

    @model_validator(mode="after")
    def eligibility_agrees(self) -> ResearchRunSummaryEnvelopeV1:
        ids = [candidate.variant_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate variant_id values must be unique")
        expected = [
            candidate.variant_id
            for candidate in self.candidates
            if candidate.passes_screen_gate
        ]
        if self.eligible_for_human_review != expected:
            raise ValueError(
                "eligible_for_human_review does not match deterministic gates"
            )
        return self


class JobStatus(StrictModel):
    run_id: SafeIdentifier
    profile: SafeIdentifier
    state: Literal["unavailable", "available", "running", "complete", "failed"]
    proposal_allowed: bool
    reason_code: Literal[
        "not_preapproved",
        "ready_for_human_proposal",
        "already_running",
        "already_complete",
        "failed_needs_review",
    ]

    @model_validator(mode="after")
    def proposal_state_agrees(self) -> JobStatus:
        if self.proposal_allowed is not (
            self.state == "available" and self.reason_code == "ready_for_human_proposal"
        ):
            raise ValueError(
                "proposal_allowed is inconsistent with state and reason_code"
            )
        return self


class JobStatusEnvelopeV1(StrictModel):
    schema_version: Literal["language-job-status/v1"]
    export_id: Uuid4String
    artifact_kind: Literal["research_job_status"]
    study_id: Literal["glioma"]
    matrix_sha256: Sha256
    jobs: Annotated[list[JobStatus], Field(max_length=256)]

    @field_validator("export_id")
    @classmethod
    def export_id_is_uuid4(cls, value: str) -> str:
        parsed = UUID(value)
        if parsed.version != 4 or str(parsed) != value:
            raise ValueError("export_id must be a canonical UUIDv4")
        return value

    @model_validator(mode="after")
    def jobs_are_unique(self) -> JobStatusEnvelopeV1:
        keys = [(job.run_id, job.profile) for job in self.jobs]
        if len(keys) != len(set(keys)):
            raise ValueError("job status entries must be unique")
        return self


class EvidenceItem(StrictModel):
    field: SafeIdentifier
    value: str | int | float | bool | None

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("evidence values must be finite")
        return value


class RunSummaryExplanationV1(StrictModel):
    schema_version: Literal["run-summary-explanation/v1"]
    disclaimer: Literal[DISCLAIMER]
    summary: Annotated[
        str, StringConstraints(strict=True, min_length=1, max_length=1200)
    ]
    evidence: Annotated[list[EvidenceItem], Field(min_length=1, max_length=128)]
    limitations: Annotated[
        str, StringConstraints(strict=True, min_length=1, max_length=1200)
    ]
    abstained: bool
    executed: Literal[False]


class JobProposalV1(StrictModel):
    schema_version: Literal["job-proposal/v1"]
    abstained: bool
    run_id: SafeIdentifier | None
    profile: SafeIdentifier | None
    reason_code: Literal[
        "exact_preapproved_match",
        "ambiguous",
        "unavailable",
        "unsafe_request",
        "no_match",
    ]
    reason: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=400)]
    executed: Literal[False]

    @model_validator(mode="after")
    def abstention_shape(self) -> JobProposalV1:
        if self.abstained and (self.run_id is not None or self.profile is not None):
            raise ValueError("abstaining proposal must not include a job")
        if not self.abstained and (self.run_id is None or self.profile is None):
            raise ValueError("non-abstaining proposal requires run_id and profile")
        if not self.abstained and self.reason_code != "exact_preapproved_match":
            raise ValueError("non-abstaining proposal requires exact_preapproved_match")
        return self


class StudyInputQcV1(StrictModel):
    schema_version: Literal["research-study-validation/v1"]
    status: Literal["pass"]
    modality_count: Literal[4]
    modalities: list[Literal["t1", "t1ce", "t2", "flair"]]
    geometry_match: Literal[True]
    shape: list[int]
    spacing_mm: list[float]
    geometry_sha256: Sha256
    modality_sha256: dict[Literal["t1", "t1ce", "t2", "flair"], Sha256]

    @field_validator("shape")
    @classmethod
    def shape_is_bounded(cls, value: list[int]) -> list[int]:
        if len(value) != 3 or any(size < 1 or size > 512 for size in value) or math.prod(value) > 64_000_000:
            raise ValueError("input shape exceeds the fixed inference bounds")
        return value

    @field_validator("spacing_mm")
    @classmethod
    def spacing_is_positive(cls, value: list[float]) -> list[float]:
        if len(value) != 3 or any(not math.isfinite(item) or item <= 0 for item in value):
            raise ValueError("voxel spacing must be finite and positive")
        return value

    @model_validator(mode="after")
    def all_modality_hashes_present(self) -> StudyInputQcV1:
        if self.modalities != ["t1", "t1ce", "t2", "flair"]:
            raise ValueError("modalities must be in the fixed protocol order")
        if set(self.modality_sha256) != {"t1", "t1ce", "t2", "flair"}:
            raise ValueError("all four modality digests are required")
        return self


class StudySegmentationV1(StrictModel):
    status: Literal["complete"]
    output_sha256: Sha256
    output_shape: list[int]
    geometry_preserved: Literal[True]
    labels: list[Literal[0, 1]]
    label_count: Literal[2]
    nonzero_voxels: int = Field(strict=True, ge=0)

    @model_validator(mode="after")
    def binary_contract_is_exact(self) -> StudySegmentationV1:
        if len(self.output_shape) != 3 or self.labels != [0, 1]:
            raise ValueError("segmentation must preserve a 3D binary output contract")
        return self


class StudyInferenceProvenanceV1(StrictModel):
    model_id: Literal["glioma-segresnet-20260828"]
    model_scope: Literal["internal_research_only"]
    checkpoint_sha256: Literal[
        "121422a861bbe7affaa5e161058e69eea737b2390651c3c03ea20256969e99e5"
    ]
    training_git_revision: Literal["570c65ac4709dac3b05f48314ddd5aef70589a7d"]
    study_sha256: Literal[
        "e53f85b429449585089133b1d9f680c3d80125b58da3042e5510522e2b333f6d"
    ]
    profile_sha256: Literal[
        "9ec821920b6a08e914306d1651101dd52693d02c185f2750410297ec1c43fc7e"
    ]
    trainer_sha256: Literal[
        "bf5dede3b5b1ee5d916cd6f046ca7eda8ea579f0f730db6f9201e2523b0456d9"
    ]
    inference_script_sha256: Sha256
    device: str
    torch_version: str
    monai_version: str
    nibabel_version: str
    generated_at: str

    @field_validator("generated_at")
    @classmethod
    def generated_at_is_iso8601(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("generated_at must be an ISO-8601 timestamp") from error
        if parsed.tzinfo is None:
            raise ValueError("generated_at must include a timezone")
        return value


class ResearchSegmentationResultV1(StrictModel):
    schema_version: Literal["research-segmentation-result/v1"]
    job_id: Uuid4String
    study_id: Literal["glioma"]
    protocol: Literal["glioma_4seq_v1"]
    disclaimer: Literal[DISCLAIMER]
    input_qc: StudyInputQcV1
    segmentation: StudySegmentationV1
    provenance: StudyInferenceProvenanceV1

    @field_validator("job_id")
    @classmethod
    def job_id_is_uuid4(cls, value: str) -> str:
        parsed = UUID(value)
        if parsed.version != 4 or str(parsed) != value:
            raise ValueError("job_id must be a canonical UUIDv4")
        return value

    @model_validator(mode="after")
    def output_geometry_matches_input(self) -> ResearchSegmentationResultV1:
        if self.segmentation.output_shape != self.input_qc.shape:
            raise ValueError("segmentation output shape must match the validated input")
        return self
