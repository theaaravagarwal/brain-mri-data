"""Deterministic guardrails for the optional research language layer."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

SAFE_MODALITIES = ("t1", "t1ce", "t2", "flair")
SAFE_TOOLS = {
    "planner": {"list_runs", "read_run", "propose_preapproved_job"},
    "explainer": {"read_validated_result", "search_evidence"},
}
FORBIDDEN_KEYS = {"image", "images", "dicom", "path", "paths", "patient_id", "patient_name", "raw_image"}
CLINICAL_PATTERN = re.compile(
    r"\b(?:you have|the patient has|diagnos(?:e|is|ed)|recommend(?:ed)? treatment|take medication|chemotherapy|surgery)\b",
    re.IGNORECASE,
)


def load_language_policy(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported language policy")
    roles = data.get("roles", {})
    for role, tools in SAFE_TOOLS.items():
        if set(roles.get(role, {}).get("allowed_tools", [])) != tools:
            raise ValueError(f"Unsafe or incomplete tool policy for {role}")
    return data


def _reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_KEYS & set(value)
        if forbidden:
            raise ValueError("Language input contains forbidden fields: " + ", ".join(sorted(forbidden)))
        for child in value.values():
            _reject_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child)


def validate_result_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept only non-identifying, validated segmentation result metadata."""
    _reject_forbidden_keys(payload)
    required = {"schema_version", "record_id", "study_id", "protocol", "input_qc", "segmentation", "provenance"}
    missing = required - set(payload)
    if missing:
        raise ValueError("Result envelope missing: " + ", ".join(sorted(missing)))
    if payload["schema_version"] != 1 or payload["protocol"] != "glioma_4seq_v1":
        raise ValueError("Unsupported result envelope protocol")
    qc = payload["input_qc"]
    if not isinstance(qc, dict) or qc.get("status") not in {"pass", "fail"}:
        raise ValueError("input_qc must have pass/fail status")
    modalities = tuple(qc.get("modalities", []))
    if modalities and modalities != SAFE_MODALITIES:
        raise ValueError("Modalities must be exactly t1, t1ce, t2, flair in protocol order")
    segmentation = payload["segmentation"]
    if not isinstance(segmentation, dict) or segmentation.get("status") not in {"complete", "abstain"}:
        raise ValueError("segmentation must have complete/abstain status")
    if segmentation["status"] == "complete" and not isinstance(segmentation.get("whole_lesion_dice"), (int, float)):
        raise ValueError("Complete segmentation requires whole_lesion_dice")
    return payload


def validate_explanation(response: dict[str, Any], envelope: dict[str, Any]) -> dict[str, Any]:
    """Validate a structured research explanation before it is displayed."""
    validate_result_envelope(envelope)
    required = {"disclaimer", "summary", "evidence", "limitations", "abstained"}
    missing = required - set(response)
    if missing:
        raise ValueError("Explanation missing: " + ", ".join(sorted(missing)))
    if response["disclaimer"] != "Research output only; not a diagnosis or treatment recommendation.":
        raise ValueError("Explanation has an invalid disclaimer")
    text = " ".join(str(response[key]) for key in ("summary", "limitations"))
    if CLINICAL_PATTERN.search(text):
        raise ValueError("Explanation contains a prohibited clinical claim")
    if not isinstance(response["evidence"], list) or not all(isinstance(item, dict) for item in response["evidence"]):
        raise ValueError("Explanation evidence must be a list of objects")
    allowed = {"input_qc.status", "segmentation.status", "segmentation.whole_lesion_dice", "provenance.source_id"}
    for item in response["evidence"]:
        if set(item) != {"field", "value"} or item["field"] not in allowed:
            raise ValueError("Explanation cites an unsupported field")
        value: Any = envelope
        for component in item["field"].split("."):
            if not isinstance(value, dict) or component not in value:
                raise ValueError("Explanation cites a field absent from the result")
            value = value[component]
        if item["value"] != value:
            raise ValueError("Explanation evidence value does not match the result")
    return response


def build_explainer_prompt(envelope: dict[str, Any]) -> str:
    safe = validate_result_envelope(envelope)
    evidence_fields = ["input_qc.status", "segmentation.status"]
    if safe["segmentation"]["status"] == "complete":
        evidence_fields.append("segmentation.whole_lesion_dice")
    evidence_fields.append("provenance.source_id")
    return (
        "Return JSON only with disclaimer, summary, evidence, limitations, and abstained. "
        "Use only the supplied fields. Never diagnose, recommend treatment, infer an unseen fact, "
        "or request an image. Evidence must contain exactly one {field, value} object for every "
        "required evidence field; value must be the exact scalar value at that dotted path. "
        f"Required evidence fields: {json.dumps(evidence_fields)}. "
        "Set abstained to true exactly when segmentation.status is abstain. "
        "Exact disclaimer: Research output only; not a diagnosis or treatment recommendation.\n"
        + json.dumps(safe, sort_keys=True)
    )


def validate_job_proposal(proposal: dict[str, Any], allowed_jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """Allow a planner to propose, never execute, only an existing matrix job."""
    if set(proposal) != {"run_id", "profile", "reason"}:
        raise ValueError("A job proposal must contain only run_id, profile, and reason")
    matches = [job for job in allowed_jobs if job["run_id"] == proposal["run_id"]]
    if len(matches) != 1 or matches[0]["profile"] != proposal["profile"]:
        raise ValueError("Proposal is not a pre-approved matrix job")
    return proposal
