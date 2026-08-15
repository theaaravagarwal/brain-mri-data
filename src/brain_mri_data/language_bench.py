"""Offline scorers for constrained language-layer benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .language_gateway import validate_explanation


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _responses(path: Path) -> dict[str, dict[str, Any]]:
    return {item["id"]: item["response"] for item in read_jsonl(path)}


def score_structured(fixtures_path: Path, responses_path: Path) -> dict[str, Any]:
    responses = _responses(responses_path)
    cases = []
    for fixture in read_jsonl(fixtures_path):
        case = {"id": fixture["id"], "passed": False, "reasons": []}
        response = responses.get(fixture["id"])
        if response is None:
            case["reasons"].append("missing_response")
        else:
            try:
                validate_explanation(response, fixture["record"])
                fields = {item["field"] for item in response["evidence"]}
                missing = set(fixture["required_fields"]) - fields
                if missing:
                    case["reasons"].append("missing_evidence:" + ",".join(sorted(missing)))
                if bool(response["abstained"]) != bool(fixture["must_abstain"]):
                    case["reasons"].append("incorrect_abstention")
            except ValueError as error:
                case["reasons"].append(str(error))
        case["passed"] = not case["reasons"]
        cases.append(case)
    return {"benchmark": "structured", "passed": sum(item["passed"] for item in cases), "total": len(cases), "cases": cases}


def score_evidence(fixtures_path: Path, responses_path: Path) -> dict[str, Any]:
    responses = _responses(responses_path)
    cases = []
    for fixture in read_jsonl(fixtures_path):
        case = {"id": fixture["id"], "passed": False, "reasons": []}
        response = responses.get(fixture["id"])
        if not isinstance(response, dict):
            case["reasons"].append("missing_response")
        else:
            answer = str(response.get("answer", "")).lower()
            citations = set(response.get("citations", []))
            missing = [term for term in fixture["required_terms"] if term not in answer]
            if missing:
                case["reasons"].append("missing_terms:" + ",".join(missing))
            if not citations <= set(fixture["allowed_source_ids"]) or not citations:
                case["reasons"].append("invalid_citation")
        case["passed"] = not case["reasons"]
        cases.append(case)
    return {"benchmark": "evidence", "passed": sum(item["passed"] for item in cases), "total": len(cases), "cases": cases}
