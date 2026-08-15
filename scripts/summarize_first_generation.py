#!/usr/bin/env python3
"""Summarize three completed internal-pilot runs without overstating evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("runs", type=Path, nargs=3)
    return parser.parse_args()


def best_validation(run: Path) -> dict:
    metrics = run / "metrics.jsonl"
    if not metrics.is_file() or not (run / "external.json").is_file():
        raise FileNotFoundError(f"Incomplete run: {run}")
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    candidates = [row for row in rows if isinstance(row.get("validation"), dict)]
    if not candidates:
        raise ValueError(f"No validation metrics in {metrics}")
    winner = max(candidates, key=lambda row: float(row["validation"]["mean_dice"]))
    run_info = json.loads((run / "run.json").read_text())
    return {
        "run": str(run),
        "seed": int(run_info["seed"]),
        "best_epoch": int(winner["epoch"]),
        "mean_dice": float(winner["validation"]["mean_dice"]),
        "mean_box_iou": float(winner["validation"]["mean_box_iou"]),
        "median_hd95_mm": winner["validation"].get("median_hd95_mm"),
        "external_evaluation": json.loads((run / "external.json").read_text()).get("external_evaluation"),
    }


def main() -> None:
    args = arguments()
    study = json.loads(args.study.read_text())
    if study.get("evaluation_status") != "pilot_internal_only":
        raise ValueError("This report only supports an internal pilot study")
    records = sorted((best_validation(run) for run in args.runs), key=lambda item: item["seed"])
    dice = [item["mean_dice"] for item in records]
    boxes = [item["mean_box_iou"] for item in records]
    payload = {
        "schema_version": 1,
        "study_id": study["study"]["study_id"],
        "study": str(args.study.resolve()),
        "evaluation_scope": "internal validation only; no independent external test was run",
        "runs": records,
        "summary": {
            "seeds": [item["seed"] for item in records],
            "best_validation_dice_mean": mean(dice),
            "best_validation_dice_population_sd": pstdev(dice),
            "best_validation_box_iou_mean": mean(boxes),
            "best_validation_box_iou_population_sd": pstdev(boxes),
        },
        "next_step_gate": "Review failure cases and stability before changing the locked model or starting a new generation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
