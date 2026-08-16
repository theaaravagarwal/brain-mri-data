"""Immutable export, ingest, explanation, and proposal operations."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from .language_contracts import (
    FORBIDDEN_TEXT_PATTERN,
    JobProposalV1,
    JobStatusEnvelopeV1,
    ResearchRunSummaryEnvelopeV1,
    RunSummaryExplanationV1,
)
from .language_ollama import ask_ollama, model_digest, run_summary_prompt
from .run_matrix import expand_matrix

MAX_ENVELOPE_BYTES = 256 * 1024
MAX_JSON_DEPTH = 16
CLINICAL_PATTERN = re.compile(
    r"\b(?:you have|the patient has|diagnos(?:e|is|ed)|recommend(?:ed)? treatment|"
    r"take medication|chemotherapy|surgery|prognosis)\b",
    re.IGNORECASE,
)
PLANNER_UNSAFE_PATTERN = re.compile(
    r"<\s*/?\s*(?:system|assistant)|"
    r"\bignore\b.{0,80}\b(?:rules?|schema|allowed_jobs)\b|"
    r'"(?:tool|command)"\s*:|'
    r"(?:/home/|/mnt/|\\\\)|"
    r"\b(?:start|execute|launch|claim|train|shell|bypass|pretend)\b|"
    r"executed\s*=\s*true",
    re.IGNORECASE | re.DOTALL,
)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _depth(value: Any, current: int = 0) -> int:
    if current > MAX_JSON_DEPTH:
        raise ValueError(f"JSON nesting exceeds {MAX_JSON_DEPTH}")
    if isinstance(value, dict):
        return max(
            (_depth(child, current + 1) for child in value.values()), default=current
        )
    if isinstance(value, list):
        return max((_depth(child, current + 1) for child in value), default=current)
    return current


def strict_json_bytes(data: bytes, *, maximum_bytes: int = MAX_ENVELOPE_BYTES) -> Any:
    if len(data) > maximum_bytes:
        raise ValueError(f"JSON exceeds {maximum_bytes} bytes")
    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"invalid number: {constant}")
            ),
        )
    except UnicodeDecodeError as error:
        raise ValueError("JSON must be UTF-8") from error
    _depth(value)
    return value


def read_strict_json(path: Path, *, maximum_bytes: int = MAX_ENVELOPE_BYTES) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON input must be a regular non-symlink file: {path}")
    return strict_json_bytes(path.read_bytes(), maximum_bytes=maximum_bytes)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_atomic_ready(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    write_once(partial, data)
    try:
        os.link(partial, path, follow_symlinks=False)
        partial.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _safe_run_directory(raw_path: str, runs_root: Path) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = runs_root.parent / candidate
    if candidate.is_symlink():
        raise ValueError("source run path must not be a symlink")
    resolved = candidate.resolve()
    root = runs_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError("source run path is outside the configured runs root")
    if resolved.is_symlink() or not resolved.is_dir():
        raise ValueError("source run path must be a regular directory")
    return resolved


def _provenance_for(row: dict[str, Any], runs_root: Path) -> dict[str, Any]:
    run = _safe_run_directory(str(row["run"]), runs_root)
    run_info = read_strict_json(run / "run.json")
    external = read_strict_json(run / "external.json")
    if external.get("external_evaluation") != "not_run: pilot_internal_only":
        raise ValueError("foreground screen must remain explicitly internal-only")
    if external.get("run") != run_info:
        raise ValueError("external artifact does not embed the exact run contract")
    return {
        "git_revision": run_info["git_revision"],
        "study_sha256": run_info["study_sha256"],
        "profile_sha256": run_info["profile_sha256"],
        "checkpoint_sha256": external["checkpoint_sha256"],
    }


def _metrics(row: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(row[key])
        for key in (
            "overall_mean_dice",
            "smallest_quartile_mean_dice",
            "mean_derived_box_iou",
            "mean_hd95_mm",
        )
    }


def _variant_id(probability: float) -> str:
    if probability == 0.0:
        return "uniform"
    percent = probability * 100
    if not percent.is_integer():
        raise ValueError("foreground probability must map to an integer percentage")
    return f"fg{int(percent)}"


def export_run_summary(
    source: Path,
    outbox: Path,
    runs_root: Path,
    *,
    run_group_id: str,
    export_id: UUID | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Build a new envelope from an allowlist of completed aggregate fields."""
    if source.is_symlink() or not source.is_file():
        raise ValueError("source summary must be a regular non-symlink file")
    source_data = source.read_bytes()
    raw = strict_json_bytes(source_data, maximum_bytes=1024 * 1024)
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported foreground summary schema")
    if (
        raw.get("evaluation_scope")
        != "single-seed internal-validation screen; not external evidence"
    ):
        raise ValueError("source is not the expected internal-validation screen")
    if (
        raw.get("review_status") != "human_review_required"
        or raw.get("automatic_promotion") is not False
    ):
        raise ValueError("source summary is missing the human-review gate")

    baseline_row = raw["baseline"]
    baseline_probability = float(baseline_row["foreground_probability"])
    if baseline_probability != 0.0:
        raise ValueError("screen baseline must use uniform sampling")
    baseline = {
        "variant_id": "uniform",
        "seed": int(baseline_row["seed"]),
        "best_epoch": int(baseline_row["best_epoch"]),
        "foreground_probability": 0.0,
        "metrics": _metrics(baseline_row),
        "provenance": _provenance_for(baseline_row, runs_root),
    }

    candidates = []
    probability_to_id: dict[float, str] = {}
    for row in raw["candidates"]:
        probability = float(row["foreground_probability"])
        variant_id = _variant_id(probability)
        probability_to_id[probability] = variant_id
        gates = row["screen_gate"]
        candidates.append(
            {
                "variant_id": variant_id,
                "seed": int(row["seed"]),
                "best_epoch": int(row["best_epoch"]),
                "foreground_probability": probability,
                "metrics": _metrics(row),
                "delta_vs_uniform": {
                    key: float(row["delta_vs_uniform"][key]) for key in _metrics(row)
                },
                "screen_gates": {
                    "smallest_quartile_improves_by_at_least_0_02": gates[
                        "smallest_quartile_improves_by_at_least_0.02"
                    ],
                    "overall_dice_declines_by_no_more_than_0_005": gates[
                        "overall_dice_declines_by_no_more_than_0.005"
                    ],
                },
                "passes_screen_gate": row["passes_screen_gate"],
                "provenance": _provenance_for(row, runs_root),
            }
        )

    envelope = ResearchRunSummaryEnvelopeV1.model_validate(
        {
            "schema_version": "research-run-summary/v1",
            "export_id": str(export_id or uuid4()),
            "artifact_kind": "cnn_research_run_summary",
            "study_id": "glioma",
            "protocol": "glioma_4seq_v1",
            "run_group_id": run_group_id,
            "evaluation_scope": "single_seed_internal_validation_screen",
            "review_status": "human_review_required",
            "automatic_promotion": False,
            "smallest_quartile_case_count": int(raw["smallest_quartile_case_count"]),
            "source_summary_sha256": sha256_bytes(source_data),
            "baseline": baseline,
            "candidates": candidates,
            "eligible_for_human_review": [
                probability_to_id[float(value)]
                for value in raw["eligible_for_human_review"]
            ],
        }
    )
    payload = envelope.model_dump(mode="json")
    assert_direct_identifier_free(payload)
    destination = outbox / f"{envelope.export_id}.json"
    data = canonical_json(payload)
    write_once(destination, data)
    receipt = {
        "export_id": str(envelope.export_id),
        "sha256": sha256_bytes(data),
        "bytes": len(data),
        "status": "exported",
    }
    write_once(destination.with_suffix(".receipt.json"), canonical_json(receipt))
    return destination, receipt


def assert_direct_identifier_free(value: Any, *, key: str = "") -> None:
    forbidden_keys = {
        "case_id",
        "patient_id",
        "patient_name",
        "subject_id",
        "mrn",
        "dob",
        "dicom",
        "uid",
        "path",
        "paths",
        "run",
        "study",
        "image",
        "images",
        "mask",
        "label",
        "hostname",
    }
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key.lower() in forbidden_keys:
                raise ValueError(f"forbidden envelope field: {child_key}")
            assert_direct_identifier_free(child, key=child_key)
    elif isinstance(value, list):
        for child in value:
            assert_direct_identifier_free(child, key=key)
    elif (
        isinstance(value, str)
        and key not in {"summary", "limitations", "disclaimer", "schema_version"}
        and FORBIDDEN_TEXT_PATTERN.search(value)
    ):
        raise ValueError(f"identifier-like content in field: {key}")


def validate_run_summary(value: Any) -> ResearchRunSummaryEnvelopeV1:
    envelope = ResearchRunSummaryEnvelopeV1.model_validate(value)
    assert_direct_identifier_free(envelope.model_dump(mode="json"))
    return envelope


def build_job_status_envelope(
    availability_path: Path,
    matrix_path: Path,
    output: Path,
    *,
    export_id: UUID | None = None,
) -> dict[str, Any]:
    """Construct a complete status snapshot from an explicit human/controller input."""
    raw = read_strict_json(availability_path)
    if (
        set(raw) != {"schema_version", "jobs"}
        or raw["schema_version"] != 1
        or not isinstance(raw["jobs"], list)
    ):
        raise ValueError(
            "availability input must contain only schema_version=1 and jobs"
        )
    matrix_jobs = {
        (job["run_id"], job["profile"]): job for job in expand_matrix(matrix_path)
    }
    source_jobs: dict[tuple[str, str], dict[str, Any]] = {}
    reason_for_state = {
        "unavailable": "not_preapproved",
        "available": "ready_for_human_proposal",
        "running": "already_running",
        "complete": "already_complete",
        "failed": "failed_needs_review",
    }
    for item in raw["jobs"]:
        if not isinstance(item, dict) or set(item) != {"run_id", "profile", "state"}:
            raise ValueError(
                "each availability job must contain only run_id, profile, and state"
            )
        key = (item["run_id"], item["profile"])
        if key not in matrix_jobs:
            raise ValueError(
                "availability input contains a job outside the frozen matrix"
            )
        if key in source_jobs:
            raise ValueError("availability input contains a duplicate job")
        if item["state"] not in reason_for_state:
            raise ValueError("availability input contains an unsupported state")
        source_jobs[key] = item
    if set(source_jobs) != set(matrix_jobs):
        raise ValueError(
            "availability input must explicitly classify every frozen matrix job"
        )
    jobs = []
    for key in matrix_jobs:
        state = source_jobs[key]["state"]
        jobs.append(
            {
                "run_id": key[0],
                "profile": key[1],
                "state": state,
                "proposal_allowed": state == "available",
                "reason_code": reason_for_state[state],
            }
        )
    envelope = JobStatusEnvelopeV1.model_validate(
        {
            "schema_version": "language-job-status/v1",
            "export_id": str(export_id or uuid4()),
            "artifact_kind": "research_job_status",
            "study_id": "glioma",
            "matrix_sha256": sha256_file(matrix_path),
            "jobs": jobs,
        }
    )
    payload = envelope.model_dump(mode="json")
    assert_direct_identifier_free(payload)
    data = canonical_json(payload)
    write_once(output, data)
    return {
        "export_id": envelope.export_id,
        "sha256": sha256_bytes(data),
        "bytes": len(data),
        "status": "exported",
    }


def validate_transfer_envelope(
    value: Any,
) -> ResearchRunSummaryEnvelopeV1 | JobStatusEnvelopeV1:
    if not isinstance(value, dict):
        raise TypeError("language envelope must be a JSON object")
    if value.get("schema_version") == "research-run-summary/v1":
        return validate_run_summary(value)
    if value.get("schema_version") == "language-job-status/v1":
        status = JobStatusEnvelopeV1.model_validate(value)
        assert_direct_identifier_free(status.model_dump(mode="json"))
        return status
    raise ValueError("unsupported language envelope schema")


def ingest_envelope(data: bytes, inbox: Path) -> dict[str, Any]:
    value = strict_json_bytes(data)
    envelope = validate_transfer_envelope(value)
    canonical = canonical_json(envelope.model_dump(mode="json"))
    digest = sha256_bytes(canonical)
    destination_directory = (
        "ready" if isinstance(envelope, ResearchRunSummaryEnvelopeV1) else "statuses"
    )
    ready = inbox / destination_directory / f"{envelope.export_id}.json"
    inbox.mkdir(parents=True, exist_ok=True, mode=0o700)
    for directory in (
        "ready",
        "processing",
        "processed",
        "quarantine",
        "explanations",
        "statuses",
    ):
        (inbox / directory).mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = inbox / ".ingest.lock"
    lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(lock_descriptor, "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if any(
            (inbox / directory / ready.name).exists()
            for directory in (
                "ready",
                "processing",
                "processed",
                "quarantine",
                "statuses",
            )
        ):
            raise FileExistsError(f"export_id already received: {envelope.export_id}")
        _write_atomic_ready(ready, canonical)
    return {
        "export_id": str(envelope.export_id),
        "sha256": digest,
        "bytes": len(canonical),
        "status": "accepted",
    }


def push_envelope(
    path: Path, host: str, remote_command: str, identity: Path | None = None
) -> dict[str, Any]:
    envelope = validate_transfer_envelope(read_strict_json(path))
    data = canonical_json(envelope.model_dump(mode="json"))
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ForwardAgent=no",
        "-o",
        "ClearAllForwardings=yes",
    ]
    if identity is not None:
        command.extend(("-i", str(identity)))
    command.extend((host, remote_command))
    result = subprocess.run(command, input=data, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"language envelope push failed with status {result.returncode}: {result.stderr.decode(errors='replace').strip()}"
        )
    receipt = strict_json_bytes(result.stdout, maximum_bytes=16 * 1024)
    if receipt.get("export_id") != str(envelope.export_id) or receipt.get(
        "sha256"
    ) != sha256_bytes(data):
        raise ValueError("receiver receipt does not match the exported envelope")
    return receipt


def flatten_evidence(envelope: ResearchRunSummaryEnvelopeV1) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = [
        {"field": "review_status", "value": envelope.review_status},
        {"field": "automatic_promotion", "value": envelope.automatic_promotion},
        {
            "field": "smallest_quartile_case_count",
            "value": envelope.smallest_quartile_case_count,
        },
    ]
    for prefix, result in [
        ("baseline", envelope.baseline),
        *[
            (f"candidates.{index}", item)
            for index, item in enumerate(envelope.candidates)
        ],
    ]:
        evidence.extend(
            (
                {"field": f"{prefix}.variant_id", "value": result.variant_id},
                {
                    "field": f"{prefix}.foreground_probability",
                    "value": result.foreground_probability,
                },
            )
        )
        evidence.extend(
            {
                "field": f"{prefix}.metrics.{field}",
                "value": getattr(result.metrics, field),
            }
            for field in (
                "overall_mean_dice",
                "smallest_quartile_mean_dice",
                "mean_derived_box_iou",
                "mean_hd95_mm",
            )
        )
        if prefix != "baseline":
            evidence.append(
                {
                    "field": f"{prefix}.passes_screen_gate",
                    "value": result.passes_screen_gate,
                }
            )
    return evidence


def validate_run_explanation(
    response: Any, envelope: ResearchRunSummaryEnvelopeV1
) -> RunSummaryExplanationV1:
    explanation = RunSummaryExplanationV1.model_validate(response)
    narrative = f"{explanation.summary} {explanation.limitations}"
    if CLINICAL_PATTERN.search(narrative):
        raise ValueError("explanation contains a prohibited clinical claim")
    if FORBIDDEN_TEXT_PATTERN.search(narrative):
        raise ValueError("explanation contains identifier-like or path-like content")
    expected = flatten_evidence(envelope)
    actual = [item.model_dump(mode="json") for item in explanation.evidence]
    if actual != expected:
        raise ValueError(
            "explanation evidence must exactly match the required envelope evidence"
        )
    allowed_numbers = {
        float(item["value"])
        for item in expected
        if isinstance(item["value"], (int, float))
        and not isinstance(item["value"], bool)
    }
    narrative_numbers = {
        float(match)
        for match in re.findall(
            r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9])", narrative
        )
    }
    if not narrative_numbers <= allowed_numbers:
        raise ValueError("explanation narrative contains an unsupported numeric claim")
    if explanation.abstained:
        raise ValueError("valid completed run summaries must not be marked abstained")
    return explanation


def explain_run_summary(path: Path, host: str, model: str) -> dict[str, Any]:
    envelope = validate_run_summary(read_strict_json(path))
    schema = RunSummaryExplanationV1.model_json_schema()
    response, telemetry = ask_ollama(
        host, model, run_summary_prompt(envelope, schema), schema
    )
    explanation = validate_run_explanation(response, envelope)
    input_data = canonical_json(envelope.model_dump(mode="json"))
    response_data = canonical_json(explanation.model_dump(mode="json"))
    return {
        "schema_version": "language-explainer-artifact/v1",
        "role": "explainer",
        "model": model,
        "model_digest": model_digest(host, model),
        "input_export_id": str(envelope.export_id),
        "input_sha256": sha256_bytes(input_data),
        "prompt_schema_version": "run-summary-explanation/v1",
        "response_sha256": sha256_bytes(response_data),
        "executed": False,
        "response": explanation.model_dump(mode="json"),
        "telemetry": telemetry,
    }


def render_explanation(
    artifact: dict[str, Any], envelope: ResearchRunSummaryEnvelopeV1
) -> str:
    response = artifact["response"]
    lines = [
        "# CNN research-screen explanation",
        "",
        response["disclaimer"],
        "",
        "## Summary",
        "",
        _escape_markdown(response["summary"]),
        "",
        "## Evidence",
        "",
        "| Field | Value |",
        "| --- | ---: |",
    ]
    for item in response["evidence"]:
        lines.append(f"| `{item['field']}` | `{json.dumps(item['value'])}` |")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            _escape_markdown(response["limitations"]),
            "",
            f"Review status: `{envelope.review_status}`. Automatic promotion: `{str(envelope.automatic_promotion).lower()}`.",
            "",
        ]
    )
    return "\n".join(lines)


def _escape_markdown(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    for character in ("`", "*", "_", "{", "}", "[", "]", "<", ">", "#", "|", "(", ")"):
        escaped = escaped.replace(character, "\\" + character)
    return escaped.replace("\r", " ").replace("\n", " ")


def consume_inbox(inbox: Path, host: str, model: str) -> dict[str, Any]:
    counts = {"processed": 0, "quarantined": 0}
    for ready in sorted((inbox / "ready").glob("*.json")):
        if ready.is_symlink() or not ready.is_file():
            continue
        processing = inbox / "processing" / ready.name
        os.replace(ready, processing)
        try:
            envelope = validate_run_summary(read_strict_json(processing))
            artifact = explain_run_summary(processing, host, model)
            stem = str(envelope.export_id)
            write_once(
                inbox / "explanations" / f"{stem}.json", canonical_json(artifact)
            )
            write_once(
                inbox / "explanations" / f"{stem}.md",
                render_explanation(artifact, envelope).encode("utf-8"),
            )
            os.replace(processing, inbox / "processed" / ready.name)
            counts["processed"] += 1
        except Exception as error:  # noqa: BLE001 - every failed input must be quarantined
            os.replace(processing, inbox / "quarantine" / ready.name)
            error_artifact = {
                "schema_version": 1,
                "input": ready.name,
                "error_type": type(error).__name__,
                "status": "quarantined",
            }
            write_once(
                inbox / "quarantine" / f"{ready.stem}.error.json",
                canonical_json(error_artifact),
            )
            counts["quarantined"] += 1
    return counts


def allowed_jobs_from_status(
    status: JobStatusEnvelopeV1, matrix_path: Path
) -> list[dict[str, str]]:
    if status.matrix_sha256 != sha256_file(matrix_path):
        raise ValueError("job status matrix hash does not match the frozen matrix")
    assert_direct_identifier_free(status.model_dump(mode="json"))
    matrix = {
        (job["run_id"], job["profile"]): job for job in expand_matrix(matrix_path)
    }
    allowed = []
    for job in status.jobs:
        key = (job.run_id, job.profile)
        if job.proposal_allowed and key in matrix:
            allowed.append({"run_id": job.run_id, "profile": job.profile})
    return allowed


def validate_proposal(
    proposal: Any, allowed_jobs: Iterable[dict[str, str]]
) -> JobProposalV1:
    parsed = JobProposalV1.model_validate(proposal)
    if CLINICAL_PATTERN.search(parsed.reason) or FORBIDDEN_TEXT_PATTERN.search(
        parsed.reason
    ):
        raise ValueError("proposal reason contains prohibited content")
    allowed = {(job["run_id"], job["profile"]) for job in allowed_jobs}
    if not parsed.abstained and (parsed.run_id, parsed.profile) not in allowed:
        raise ValueError("proposal is not an available pre-approved matrix job")
    return parsed


def planner_preflight(
    request_text: str, allowed_jobs: list[dict[str, str]]
) -> JobProposalV1 | None:
    """Resolve obvious safe matches and unsafe requests without consulting an LLM."""
    normalized = re.sub(
        r"\b(?:do not execute|without executing anything)\b",
        "",
        request_text,
        flags=re.IGNORECASE,
    )
    if PLANNER_UNSAFE_PATTERN.search(normalized):
        return JobProposalV1.model_validate(
            {
                "schema_version": "job-proposal/v1",
                "abstained": True,
                "run_id": None,
                "profile": None,
                "reason_code": "unsafe_request",
                "reason": "The request contains an execution, tool, path, or instruction-override pattern.",
                "executed": False,
            }
        )
    exact = [job for job in allowed_jobs if job["run_id"] in request_text]
    if len(exact) == 1:
        return JobProposalV1.model_validate(
            {
                "schema_version": "job-proposal/v1",
                "abstained": False,
                "run_id": exact[0]["run_id"],
                "profile": exact[0]["profile"],
                "reason_code": "exact_preapproved_match",
                "reason": "The request names exactly one available pre-approved job for human review.",
                "executed": False,
            }
        )
    return None


def read_untrusted_request(path: Path, maximum_bytes: int = 4096) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("planner request must be a regular non-symlink file")
    data = path.read_bytes()
    if len(data) > maximum_bytes:
        raise ValueError(f"planner request exceeds {maximum_bytes} bytes")
    try:
        return data.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as error:
        raise ValueError("planner request must be UTF-8") from error
