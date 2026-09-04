import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluate_fixed_external import aggregate, bootstrap_mean_ci, percentile_summary


class FixedExternalEvaluationTests(unittest.TestCase):
    def test_percentiles_and_bootstrap_are_deterministic(self) -> None:
        summary = percentile_summary([0.5, 0.75, 1.0])
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["median"], 0.75)
        self.assertEqual(bootstrap_mean_ci([0.5, 0.75, 1.0], 7, 200), bootstrap_mean_ci([0.5, 0.75, 1.0], 7, 200))

    def test_public_summary_has_no_case_identifiers(self) -> None:
        plan = {
            "benchmark_id": "test",
            "dataset": {"expected_cases": 2, "source_id": "external", "source_revision": "abc"},
            "aggregation": {"bootstrap_seed": 9, "bootstrap_replicates": 100},
        }
        rows = [
            {"case_token": "case_001", "metrics": {metric: value for metric, value in zip(
                ("whole_lesion_dice", "whole_lesion_iou", "precision", "recall", "hd95_mm"),
                (0.8, 0.7, 0.9, 0.8, 2.0), strict=True)}, "predicted_voxels": 4, "inference_seconds": 1.0},
            {"case_token": "case_002", "metrics": {metric: value for metric, value in zip(
                ("whole_lesion_dice", "whole_lesion_iou", "precision", "recall", "hd95_mm"),
                (0.9, 0.8, 0.8, 0.9, 3.0), strict=True)}, "predicted_voxels": 0, "inference_seconds": 2.0},
        ]
        result = aggregate(rows, plan, 3.0)
        encoded = json.dumps(result)
        self.assertNotIn("case_001", encoded)
        self.assertEqual(result["case_count"], 2)
        self.assertEqual(result["failures"]["empty_prediction_count"], 1)
        self.assertEqual(result["failures"]["descriptive_dice_bands"], {
            "at_least_0_90": 1, "0_75_to_0_90": 1, "0_50_to_0_75": 0, "below_0_50": 0,
        })
        self.assertAlmostEqual(result["metrics"]["whole_lesion_dice"]["mean"], 0.85)


if __name__ == "__main__":
    unittest.main()
