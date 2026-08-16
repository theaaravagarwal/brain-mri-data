#!/usr/bin/env python3
"""Summarize the bounded single-seed foreground-sampling screen for human review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-analysis", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("candidates", type=Path, nargs=3)
    return parser.parse_args()


def best_validation(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != 10 or not (run / "external.json").is_file():
        raise ValueError(f"Screen run is incomplete: {run}")
    winner = max(rows, key=lambda row: float(row["validation"]["mean_dice"]))
    return json.loads((run / "run.json").read_text()), winner


def summarize_run(run: Path, smallest_cases: set[str]) -> dict[str, Any]:
    run_info, winner = best_validation(run)
    validation = winner["validation"]
    per_case = {item["case_id"]: item for item in validation["per_case"]}
    if not smallest_cases <= set(per_case):
        raise ValueError(f"Run is missing smallest-quartile cases: {run}")
    sampling = run_info.get("patch_sampling", {})
    return {
        "run": str(run),
        "seed": int(run_info["seed"]),
        "best_epoch": int(winner["epoch"]),
        "foreground_probability": float(sampling.get("foreground_probability", 0.0)),
        "overall_mean_dice": float(validation["mean_dice"]),
        "smallest_quartile_mean_dice": mean(
            float(per_case[case_id]["whole_lesion_dice"]) for case_id in smallest_cases
        ),
        "mean_derived_box_iou": float(validation["mean_derived_box_iou"]),
        "mean_hd95_mm": float(validation["mean_hd95_mm"]),
    }


def main() -> None:
    args = arguments()
    failure = json.loads(args.failure_analysis.read_text())
    smallest_cases = {
        case["case_id"] for case in failure["all_cases"] if int(case["size_quartile"]) == 1
    }
    baseline = summarize_run(args.baseline, smallest_cases)
    candidates = sorted(
        (summarize_run(run, smallest_cases) for run in args.candidates),
        key=lambda item: item["foreground_probability"],
    )
    for candidate in candidates:
        candidate["delta_vs_uniform"] = {
            "overall_mean_dice": candidate["overall_mean_dice"] - baseline["overall_mean_dice"],
            "smallest_quartile_mean_dice": (
                candidate["smallest_quartile_mean_dice"] - baseline["smallest_quartile_mean_dice"]
            ),
            "mean_derived_box_iou": candidate["mean_derived_box_iou"] - baseline["mean_derived_box_iou"],
            "mean_hd95_mm": candidate["mean_hd95_mm"] - baseline["mean_hd95_mm"],
        }
        candidate["screen_gate"] = {
            "smallest_quartile_improves_by_at_least_0.02": (
                candidate["delta_vs_uniform"]["smallest_quartile_mean_dice"] >= 0.02
            ),
            "overall_dice_declines_by_no_more_than_0.005": (
                candidate["delta_vs_uniform"]["overall_mean_dice"] >= -0.005
            ),
        }
        candidate["passes_screen_gate"] = all(candidate["screen_gate"].values())
    payload = {
        "schema_version": 1,
        "evaluation_scope": "single-seed internal-validation screen; not external evidence",
        "review_status": "human_review_required",
        "automatic_promotion": False,
        "smallest_quartile_case_count": len(smallest_cases),
        "baseline": baseline,
        "candidates": candidates,
        "eligible_for_human_review": [
            candidate["foreground_probability"] for candidate in candidates if candidate["passes_screen_gate"]
        ],
        "next_step": "Human review must decide whether one setting receives the remaining two fixed seeds.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    rows = [
        "# Foreground-sampling screen",
        "",
        "Internal validation only. One seed per candidate; no automatic promotion.",
        "",
        "| Forced foreground | Best epoch | Overall Dice | Small-lesion Q1 Dice | Δ overall | Δ Q1 | Gate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for item in [baseline, *candidates]:
        delta = item.get("delta_vs_uniform", {})
        gate = "baseline" if item is baseline else ("pass" if item["passes_screen_gate"] else "fail")
        rows.append(
            f"| {item['foreground_probability']:.2f} | {item['best_epoch']} | "
            f"{item['overall_mean_dice']:.4f} | {item['smallest_quartile_mean_dice']:.4f} | "
            f"{delta.get('overall_mean_dice', 0.0):+.4f} | "
            f"{delta.get('smallest_quartile_mean_dice', 0.0):+.4f} | {gate} |"
        )
    rows.extend(["", "Stop here for human review before confirmation runs.", ""])
    args.output_markdown.write_text("\n".join(rows))
    print(json.dumps({
        "review_status": payload["review_status"],
        "eligible_for_human_review": payload["eligible_for_human_review"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
