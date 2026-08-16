"""Pre-specified patient-level analysis for frozen external-study artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .study import load_yaml


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_plan(path: Path) -> dict[str, Any]:
    plan = load_yaml(path)
    required = {"schema_version", "study_id", "seeds", "primary_outcome", "seed_aggregation", "analysis_seed", "bootstrap_resamples", "permutation_resamples", "comparisons"}
    if plan.get("schema_version") != 1 or required - set(plan):
        raise ValueError("Invalid analysis plan")
    if plan["primary_outcome"] != "whole_lesion_dice" or plan["seed_aggregation"] != "mean_per_case_across_seeds":
        raise ValueError("Unsupported analysis endpoint or seed aggregation")
    if not isinstance(plan["comparisons"], list) or not plan["comparisons"]:
        raise ValueError("Analysis plan needs at least one comparison")
    if not all(isinstance(seed, int) for seed in plan["seeds"]) or len(set(plan["seeds"])) != len(plan["seeds"]):
        raise ValueError("Analysis plan needs unique integer seeds")
    return plan


def _load_result(path: Path, study_id: str) -> tuple[dict[str, Any], dict[str, float]]:
    payload = json.loads(path.read_text())
    run = payload.get("run")
    clean = payload.get("external_clean")
    if not isinstance(run, dict) or run.get("study_id") != study_id:
        raise ValueError(f"{path} is not a {study_id} study result")
    if run.get("evaluation_status") != "external_test_locked":
        raise ValueError(f"{path} is not a locked external-test result")
    if not isinstance(clean, dict) or not isinstance(clean.get("per_case"), list):
        raise ValueError(f"{path} lacks case-linked external_clean results")
    scores: dict[str, float] = {}
    for item in clean["per_case"]:
        if not isinstance(item, dict) or not isinstance(item.get("case_id"), str) or not isinstance(item.get("whole_lesion_dice"), (int, float)):
            raise ValueError(f"{path} has an invalid per-case score")
        if item["case_id"] in scores:
            raise ValueError(f"{path} repeats case ID {item['case_id']}")
        scores[item["case_id"]] = float(item["whole_lesion_dice"])
    if not scores:
        raise ValueError(f"{path} has no external cases")
    for key in (
        "profile_id", "arm", "seed", "study_sha256", "profile_sha256",
        "trainer_sha256", "pamc_sha256", "evaluation_sha256",
    ):
        if key not in run:
            raise ValueError(f"{path} run metadata lacks {key}")
    return run, scores


def _aggregate_arm(seed_scores: dict[int, dict[str, float]]) -> dict[str, float]:
    case_sets = [set(scores) for scores in seed_scores.values()]
    if len({frozenset(case_ids) for case_ids in case_sets}) != 1:
        raise ValueError("Seed outputs have different external case sets")
    return {case_id: float(np.mean([scores[case_id] for scores in seed_scores.values()])) for case_id in sorted(case_sets[0])}


def _bootstrap(values: np.ndarray, resamples: int, seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _paired_permutation(values: np.ndarray, resamples: int, seed: int) -> float:
    generator = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    signs = generator.choice(np.array([-1.0, 1.0]), size=(resamples, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (resamples + 1))


def analyze_study(plan_path: Path, result_paths: list[Path], destination: Path) -> dict[str, Any]:
    """Write a deterministic analysis only when complete paired seed evidence exists."""
    plan = _required_plan(plan_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite analysis artifact: {destination}")
    grouped: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    provenance_by_profile: dict[str, dict[str, str]] = {}
    input_hashes: dict[str, str] = {}
    for path in result_paths:
        run, scores = _load_result(path, plan["study_id"])
        profile, arm, seed = str(run["profile_id"]), str(run["arm"]), int(run["seed"])
        provenance = {
            key: str(run[key])
            for key in ("study_sha256", "profile_sha256", "trainer_sha256", "pamc_sha256", "evaluation_sha256")
        }
        previous = provenance_by_profile.setdefault(profile, provenance)
        if previous != provenance:
            raise ValueError(f"{profile} results do not share one locked study, profile, and evaluation code")
        target = grouped.setdefault(profile, {}).setdefault(arm, {})
        if seed in target:
            raise ValueError(f"Duplicate output for profile={profile}, arm={arm}, seed={seed}")
        target[seed] = scores
        input_hashes[str(path)] = _sha256(path)

    reports: list[dict[str, Any]] = []
    for profile, arms in sorted(grouped.items()):
        expected_seeds = set(plan["seeds"])
        for arm, outputs in arms.items():
            if set(outputs) != expected_seeds:
                raise ValueError(f"{profile} arm={arm} does not match the frozen analysis seeds")
        aggregate = {arm: _aggregate_arm(scores) for arm, scores in arms.items()}
        for comparison in plan["comparisons"]:
            intervention, comparator = comparison["intervention"], comparison["comparator"]
            if intervention not in aggregate or comparator not in aggregate:
                raise ValueError(f"{profile} is missing an arm required by {comparison['id']}")
            left, right = aggregate[intervention], aggregate[comparator]
            if set(left) != set(right):
                raise ValueError(f"{profile} comparison {comparison['id']} has unmatched cases")
            case_ids = sorted(left)
            differences = np.array([left[case_id] - right[case_id] for case_id in case_ids], dtype=float)
            analysis_seed = int.from_bytes(hashlib.sha256(f"{plan['analysis_seed']}:{profile}:{comparison['id']}".encode()).digest()[:8], "big")
            reports.append({
                "profile_id": profile,
                "comparison": comparison,
                "case_count": len(case_ids),
                "mean_difference": float(differences.mean()),
                "paired_bootstrap_95_ci": _bootstrap(differences, int(plan["bootstrap_resamples"]), analysis_seed),
                "paired_permutation_p_value": _paired_permutation(differences, int(plan["permutation_resamples"]), analysis_seed + 1),
                "per_case_mean_differences": [{"case_id": case_id, "difference": float(left[case_id] - right[case_id])} for case_id in case_ids],
            })
    if not reports:
        raise ValueError("No result artifacts supplied")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "study_id": plan["study_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "analysis_plan_sha256": _sha256(plan_path),
        "result_sha256": input_hashes,
        "reports": reports,
    }
    with destination.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return {"analysis": str(destination), "reports": len(reports)}
