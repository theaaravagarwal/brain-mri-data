#!/usr/bin/env python3
"""Analyze cross-seed internal-validation failures for one completed pilot generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import nibabel as nib
import numpy as np
from scipy.stats import spearmanr


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("runs", type=Path, nargs=3)
    return parser.parse_args()


def best_validation(run: Path) -> tuple[int, int, dict]:
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    winner = max(rows, key=lambda row: float(row["validation"]["mean_dice"]))
    seed = int(json.loads((run / "run.json").read_text())["seed"])
    return seed, int(winner["epoch"]), winner["validation"]


def main() -> None:
    args = arguments()
    study = json.loads(args.study.read_text())
    validation = {
        f"{item['source_id']}:{item['case_id']}": item
        for item in study["development"]
        if item["split"] == "val"
    }
    winners = [best_validation(run) for run in args.runs]
    if len({seed for seed, _, _ in winners}) != 3:
        raise ValueError("Failure analysis requires three distinct completed seeds")
    scores = {
        seed: {item["case_id"]: item for item in report["per_case"]}
        for seed, _, report in winners
    }

    cases = []
    for case_id, item in validation.items():
        segmentation = args.data_root / "raw" / item["source_id"] / item["record"]["segmentation"]
        mask = np.asanyarray(nib.load(segmentation).dataobj) != 0
        coordinates = np.where(mask)
        lesion_voxels = int(mask.sum())
        bounding_box = [int(axis.max() - axis.min() + 1) for axis in coordinates]
        seed_dice = [float(scores[seed][case_id]["whole_lesion_dice"]) for seed, _, _ in winners]
        seed_hd95 = [float(scores[seed][case_id]["hd95_mm"]) for seed, _, _ in winners]
        cases.append({
            "case_id": case_id,
            "lesion_voxels": lesion_voxels,
            "lesion_fraction": lesion_voxels / mask.size,
            "bounding_box": bounding_box,
            "seed_dice": seed_dice,
            "mean_dice": mean(seed_dice),
            "dice_population_sd": pstdev(seed_dice),
            "minimum_dice": min(seed_dice),
            "mean_hd95_mm": mean(seed_hd95),
        })

    volumes = np.asarray([case["lesion_voxels"] for case in cases])
    dice = np.asarray([case["mean_dice"] for case in cases])
    quartiles = np.quantile(volumes, (0.25, 0.5, 0.75))
    for case in cases:
        case["size_quartile"] = int(np.searchsorted(quartiles, case["lesion_voxels"], side="right") + 1)
    correlation = spearmanr(np.log10(volumes), dice)
    by_failure = sorted(cases, key=lambda case: case["mean_dice"])
    by_variance = sorted(cases, key=lambda case: case["dice_population_sd"], reverse=True)
    payload = {
        "schema_version": 1,
        "evaluation_scope": "internal validation only; used to choose the next pilot experiment",
        "study": str(args.study),
        "runs": [str(run) for run in args.runs],
        "best_epochs": {str(seed): epoch for seed, epoch, _ in winners},
        "case_count": len(cases),
        "lesion_size_analysis": {
            "quartile_boundaries_voxels": quartiles.tolist(),
            "mean_dice_by_size_quartile": {
                str(quartile): mean(case["mean_dice"] for case in cases if case["size_quartile"] == quartile)
                for quartile in range(1, 5)
            },
            "spearman_log_volume_vs_mean_dice": {
                "rho": float(correlation.statistic),
                "p_value": float(correlation.pvalue),
            },
        },
        "consistent_failures": {
            "below_0.60_in_all_seeds": sum(max(case["seed_dice"]) < 0.60 for case in cases),
            "below_0.70_in_all_seeds": sum(max(case["seed_dice"]) < 0.70 for case in cases),
        },
        "worst_15_cases": by_failure[:15],
        "highest_variance_10_cases": by_variance[:10],
        "all_cases": sorted(cases, key=lambda case: case["case_id"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "case_count": payload["case_count"],
        **payload["lesion_size_analysis"],
        **payload["consistent_failures"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
