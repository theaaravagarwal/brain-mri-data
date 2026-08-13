from __future__ import annotations

from pathlib import Path
from typing import Any

from .indexer import read_jsonl


def audit_experiment(
    catalog: dict[str, Any], train_ids: list[str], test_ids: list[str], manifest_root: Path
) -> dict[str, Any]:
    sources = catalog["sources"]
    issues: list[dict[str, Any]] = []
    train_modules = {sources[item].get("module") for item in train_ids}
    test_modules = {sources[item].get("module") for item in test_ids}
    if None in train_modules | test_modules:
        issues.append({"severity": "block", "code": "module_undefined", "message": "Every source must declare a module."})
    if len(train_modules) != 1 or len(test_modules) != 1 or train_modules != test_modules:
        issues.append({"severity": "block", "code": "cross_module_experiment", "message": "Training and testing must use one identical module."})

    for train_id in train_ids:
        for test_id in test_ids:
            overlap = sorted(set(sources[train_id].get("potential_cohort_overlap_groups", [])) & set(sources[test_id].get("potential_cohort_overlap_groups", [])))
            if overlap:
                issues.append({
                    "severity": "block", "code": "declared_cohort_overlap_risk", "train_source": train_id,
                    "test_source": test_id, "groups": overlap,
                    "message": "Potential shared institutional/challenge lineage. A documented patient-level overlap audit is required before this comparison.",
                })
            if sources[train_id].get("label_ontology") == "unreviewed_remind" or sources[test_id].get("label_ontology") == "unreviewed_remind":
                issues.append({"severity": "block", "code": "unreviewed_label_ontology", "message": "REMIND is blocked until mask values and intra-/pre-operative conventions are reviewed."})

    manifest_findings = _manifest_overlap(train_ids, test_ids, manifest_root)
    issues.extend(manifest_findings["issues"])
    return {
        "status": "blocked" if any(issue["severity"] == "block" for issue in issues) else "review_required",
        "train_sources": train_ids, "test_sources": test_ids,
        "module": next(iter(train_modules), None), "issues": issues,
        "manifest_audit": manifest_findings["coverage"],
        "required_before_publication": [
            "Freeze exact dataset versions and source licenses.",
            "Record patient-level overlap evidence for every train/test source pair.",
            "Review and version the source-to-canonical-label mapping.",
            "Lock test cases before hyperparameter selection and report each source separately.",
        ],
    }


def _manifest_overlap(train_ids: list[str], test_ids: list[str], root: Path) -> dict[str, Any]:
    issues, coverage = [], {"manifests_present": [], "manifests_missing": []}
    cases: dict[str, set[str]] = {}
    for source_id in sorted(set(train_ids + test_ids)):
        path = root / f"{source_id}.cases.jsonl"
        if not path.exists():
            coverage["manifests_missing"].append(source_id)
            continue
        coverage["manifests_present"].append(source_id)
        cases[source_id] = {str(record["patient_id"]).strip().lower() for record in read_jsonl(path)}
    for train_id in train_ids:
        for test_id in test_ids:
            if train_id in cases and test_id in cases:
                overlap = sorted(cases[train_id] & cases[test_id])
                if overlap:
                    issues.append({"severity": "block", "code": "manifest_patient_id_overlap", "train_source": train_id, "test_source": test_id, "count": len(overlap), "examples": overlap[:10]})
    return {"issues": issues, "coverage": coverage}
