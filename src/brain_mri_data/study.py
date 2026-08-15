from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .experiment_audit import audit_experiment
from .indexer import read_jsonl


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _approved_mapping(source: dict[str, Any]) -> dict[str, Any]:
    mapping = source.get("label_mapping", {})
    positive = mapping.get("whole_lesion", {}).get("positive_values", [])
    if mapping.get("status") != "approved" or not positive:
        raise ValueError(
            f"{source.get('locator', 'source')} has no approved whole_lesion label mapping. "
            "Run QC, review observed label values, then record the approved mapping in datasets/catalog.yaml."
        )
    return mapping


def _approved_manual_provenance_review(study: dict[str, Any]) -> dict[str, Any]:
    """Require attributable, dated evidence before a publication study is locked."""
    review = study.get("manual_provenance_review")
    if not isinstance(review, dict) or review.get("status") != "approved":
        raise ValueError("A dated manual_provenance_review must be approved before locking a study")
    for key in ("reviewer", "completed_at_utc", "evidence"):
        if not isinstance(review.get(key), str) or not review[key].strip():
            raise ValueError(f"manual_provenance_review needs a non-empty {key}")
    try:
        completed_at = datetime.fromisoformat(review["completed_at_utc"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("manual_provenance_review.completed_at_utc must be an ISO-8601 timestamp") from error
    if completed_at.tzinfo is None:
        raise ValueError("manual_provenance_review.completed_at_utc must include a timezone")
    return review


def _valid_cases(source_id: str, manifest_root: Path) -> list[dict[str, Any]]:
    cases_path = manifest_root / f"{source_id}.cases.jsonl"
    qc_path = manifest_root / f"{source_id}.qc.jsonl"
    if not cases_path.exists() or not qc_path.exists():
        raise FileNotFoundError(f"{source_id} needs indexed cases and QC results before a study can be locked")
    valid_ids = {item["case_id"] for item in read_jsonl(qc_path) if item.get("valid")}
    cases = [item for item in read_jsonl(cases_path) if item["case_id"] in valid_ids]
    if not cases:
        raise ValueError(f"{source_id} has no QC-passing cases")
    return cases


def _development_split(source_id: str, case_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{source_id}:{case_id}".encode()).hexdigest()
    return "train" if int(digest[:8], 16) / 0xFFFFFFFF < 0.85 else "val"


def build_study_manifest(
    catalog: dict[str, Any], study_path: Path, manifest_root: Path, destination: Path
) -> dict[str, Any]:
    """Freeze a multi-source study once data, QC, mappings, and review are complete."""
    study = load_yaml(study_path)
    if study.get("schema_version") != 1:
        raise ValueError("Unsupported study schema_version")
    train_sources = list(study.get("train_sources", []))
    test_sources = list(study.get("locked_test_sources", []))
    pilot = study.get("mode") == "pilot_internal_only"
    if not train_sources or set(train_sources) & set(test_sources):
        raise ValueError("Study needs non-empty, disjoint train_sources and locked_test_sources")
    if pilot and test_sources:
        raise ValueError("An internal-only pilot cannot declare locked_test_sources")
    if not pilot and not test_sources:
        raise ValueError("A publication study needs locked_test_sources")
    sources = catalog["sources"]
    unknown = sorted(set(train_sources + test_sources) - set(sources))
    if unknown:
        raise ValueError("Unknown sources in study: " + ", ".join(unknown))
    if not pilot:
        _approved_manual_provenance_review(study)

    if pilot:
        audit = {
            "status": "pilot_internal_only",
            "train_sources": train_sources,
            "test_sources": [],
            "issues": [],
            "required_before_publication": [
                "Do not report pilot validation as external-test performance.",
                "Lock an independent external test cohort before model selection is complete.",
            ],
        }
    else:
        audit = audit_experiment(catalog, train_sources, test_sources, manifest_root)
        if audit["status"] == "blocked":
            raise ValueError("Study lineage audit is blocked: " + json.dumps(audit["issues"], sort_keys=True))

    protocol = study["protocol"]
    for source_id in train_sources + test_sources:
        source = sources[source_id]
        if source.get("protocol") != protocol or source.get("module") != "adult_glioma":
            raise ValueError(f"{source_id} is not eligible for the adult-glioma {protocol} study")
        _approved_mapping(source)

    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite a locked study manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    seed = int(study["development_split_seed"])
    development, external_test = [], []
    for source_id in train_sources:
        for case in _valid_cases(source_id, manifest_root):
            development.append({
                "source_id": source_id,
                "case_id": case["case_id"],
                "split": _development_split(source_id, case["case_id"], seed),
                "record": case,
            })
    for source_id in test_sources:
        for case in _valid_cases(source_id, manifest_root):
            external_test.append({"source_id": source_id, "case_id": case["case_id"], "split": "locked_test", "record": case})
    if not any(item["split"] == "val" for item in development):
        raise ValueError("Development split produced no validation cases; add data or choose a different seed")

    payload = {
        "schema_version": 1,
        "study_id": study["study_id"],
        "evaluation_status": "pilot_internal_only" if pilot else "external_test_locked",
        "protocol": protocol,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "study_config_sha256": _file_sha256(study_path),
        "source_case_manifests": {
            source_id: _file_sha256(manifest_root / f"{source_id}.cases.jsonl")
            for source_id in sorted(train_sources + test_sources)
        },
        "audit": audit,
        "label_mappings": {source_id: _approved_mapping(sources[source_id]) for source_id in sorted(train_sources + test_sources)},
        "study": study,
        "development": development,
        "external_test": external_test,
    }
    with destination.open("x") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    return {
        "manifest": str(destination),
        "development_cases": len(development),
        "validation_cases": sum(item["split"] == "val" for item in development),
        "locked_test_cases": len(external_test),
        "status": "locked",
    }
