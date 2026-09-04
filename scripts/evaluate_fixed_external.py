#!/usr/bin/env python3
"""Resumable aggregate-only evaluation of the fixed serving model on BraTS-SSA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.run_4060_research_inference import (
    EXPECTED_CHECKPOINT_SHA256,
    MODEL_ID,
    evaluation_metrics,
    gpu_lock,
    sha256,
    validate_and_normalize,
    validate_reference,
)

MODALITY_ENDINGS = ("-t1n.nii.gz", "-t1c.nii.gz", "-t2w.nii.gz", "-t2f.nii.gz")
REFERENCE_ENDING = "-seg.nii.gz"
METRICS = ("whole_lesion_dice", "whole_lesion_iou", "precision", "recall", "hd95_mm")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan", type=Path, default=ROOT / "config/analysis/fixed-segresnet-external-benchmark.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/glioma-pilot--cuda-4060--brats--20260828--e100/best.pt")
    parser.add_argument("--max-cases", type=int, default=None, help="diagnostic subset only; cannot produce a complete summary")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def one_file(directory: Path, ending: str) -> Path:
    matches = sorted(path for path in directory.iterdir() if path.is_file() and path.name.lower().endswith(ending))
    if len(matches) != 1:
        raise ValueError(f"Expected one {ending} file in a case; found {len(matches)}")
    return matches[0]


def discover_cases(dataset: Path) -> list[tuple[str, list[Path], Path]]:
    directories = sorted(path for path in dataset.iterdir() if path.is_dir() and not path.is_symlink())
    cases = []
    for index, directory in enumerate(directories, 1):
        paths = [one_file(directory, ending) for ending in MODALITY_ENDINGS]
        reference = one_file(directory, REFERENCE_ENDING)
        cases.append((f"case_{index:03d}", paths, reference))
    return cases


def percentile_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def bootstrap_mean_ci(values: list[float], seed: int, replicates: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    chunk = 1000
    for start in range(0, replicates, chunk):
        size = min(chunk, replicates - start)
        indices = rng.integers(0, array.size, size=(size, array.size))
        means[start:start + size] = array[indices].mean(axis=1)
    return [float(value) for value in np.percentile(means, (2.5, 97.5))]


def aggregate(rows: list[dict[str, Any]], plan: dict[str, Any], elapsed_seconds: float) -> dict[str, Any]:
    aggregation = plan["aggregation"]
    metrics: dict[str, Any] = {}
    for offset, metric in enumerate(METRICS):
        values = [float(row["metrics"][metric]) for row in rows if row["metrics"][metric] is not None]
        metrics[metric] = percentile_summary(values)
        metrics[metric]["mean_ci95"] = bootstrap_mean_ci(
            values,
            int(aggregation["bootstrap_seed"]) + offset,
            int(aggregation["bootstrap_replicates"]),
        )
    latencies = [float(row["inference_seconds"]) for row in rows]
    return {
        "schema_version": "fixed-segresnet-external-summary/v1",
        "benchmark_id": plan["benchmark_id"],
        "status": "complete",
        "scope": "fixed_model_external_research_evaluation",
        "research_only": True,
        "case_count": len(rows),
        "expected_case_count": int(plan["dataset"]["expected_cases"]),
        "metrics": metrics,
        "failures": {
            "empty_prediction_count": sum(row["predicted_voxels"] == 0 for row in rows),
            "hd95_unavailable_count": sum(row["metrics"]["hd95_mm"] is None for row in rows),
            "case_error_count": 0,
        },
        "latency_seconds": percentile_summary(latencies),
        "total_elapsed_seconds": elapsed_seconds,
        "provenance": {
            "model_id": MODEL_ID,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "plan_sha256": canonical_sha256(plan),
            "dataset_source_id": plan["dataset"]["source_id"],
            "dataset_source_revision": plan["dataset"]["source_revision"],
            "generated_at": utc_now(),
        },
        "limitations": [
            "Research evaluation only; not clinical evidence or medical advice.",
            "Confidence intervals describe case sampling in this 60-case cohort and do not quantify site or population shift.",
            "This external cohort was not used for retraining, threshold tuning, or checkpoint selection.",
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from monai.inferers import sliding_window_inference
    from training.pamc import PamcSegResNet

    plan = json.loads(args.plan.read_text())
    if plan["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256 or plan["model_id"] != MODEL_ID:
        raise ValueError("Benchmark plan does not name the frozen serving model")
    if sha256(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Frozen checkpoint digest mismatch")
    cases = discover_cases(args.dataset)
    expected = int(plan["dataset"]["expected_cases"])
    if len(cases) != expected:
        raise ValueError(f"Expected exactly {expected} cases; found {len(cases)}")
    selected = cases[: args.max_cases] if args.max_cases is not None else cases
    args.output.mkdir(parents=True, exist_ok=True)
    private_path = args.output / "per-case.private.json"
    public_status_path = args.output / "status.public.json"
    rows: list[dict[str, Any]] = []
    if private_path.exists():
        saved = json.loads(private_path.read_text())
        if saved.get("plan_sha256") != canonical_sha256(plan):
            raise ValueError("Existing results use a different benchmark plan")
        rows = saved.get("cases", [])
    completed = {row["case_token"] for row in rows}
    started = time.monotonic()

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    with gpu_lock():
        model = PamcSegResNet(init_filters=32, source_count=1).to(device)
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        model.eval()

    for case_token, paths, reference_path in selected:
        if case_token in completed:
            continue
        case_started = time.monotonic()
        image, reference_image, validation = validate_and_normalize(paths)
        truth, _ = validate_reference(reference_path, reference_image)
        with gpu_lock(), torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            tensor = torch.from_numpy(image).unsqueeze(0).to(device)
            logits = sliding_window_inference(
                tensor,
                tuple(plan["inference"]["patch_size"]),
                int(plan["inference"]["sw_batch_size"]),
                lambda values: model(values)[0],
                overlap=float(plan["inference"]["overlap"]),
            )
            prediction = (torch.sigmoid(logits) >= float(plan["inference"]["threshold"])).to(torch.uint8).squeeze().cpu().numpy()
        metrics = evaluation_metrics(prediction, truth, tuple(validation["spacing_mm"]))
        row = {
            "case_token": case_token,
            "metrics": {metric: metrics[metric] for metric in METRICS},
            "predicted_voxels": int(np.count_nonzero(prediction)),
            "reference_voxels": int(np.count_nonzero(truth)),
            "inference_seconds": time.monotonic() - case_started,
        }
        rows.append(row)
        rows.sort(key=lambda item: item["case_token"])
        atomic_json(private_path, {
            "schema_version": "fixed-segresnet-external-private-results/v1",
            "plan_sha256": canonical_sha256(plan),
            "cases": rows,
        })
        elapsed = time.monotonic() - started
        new_cases = max(1, len([row for row in rows if row["case_token"] in {case[0] for case in selected}]))
        remaining = max(0, len(selected) - len(rows))
        atomic_json(public_status_path, {
            "schema_version": "fixed-segresnet-external-status/v1",
            "benchmark_id": plan["benchmark_id"],
            "status": "running",
            "completed_cases": len(rows),
            "total_cases": len(selected),
            "estimated_remaining_seconds": (elapsed / new_cases) * remaining,
            "updated_at": utc_now(),
        })

    if args.max_cases is not None:
        return json.loads(public_status_path.read_text())
    if len(rows) != expected:
        raise RuntimeError(f"Incomplete result set: {len(rows)}/{expected}")
    total_elapsed = sum(float(row["inference_seconds"]) for row in rows)
    summary = aggregate(rows, plan, total_elapsed)
    atomic_json(args.output / "summary.public.json", summary)
    atomic_json(public_status_path, {
        "schema_version": "fixed-segresnet-external-status/v1",
        "benchmark_id": plan["benchmark_id"],
        "status": "complete",
        "completed_cases": expected,
        "total_cases": expected,
        "estimated_remaining_seconds": 0,
        "updated_at": utc_now(),
    })
    return summary


def main() -> None:
    result = run(arguments())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
