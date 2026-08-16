from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_foreground_screen import summarize_run


class ForegroundScreenSummaryTests(unittest.TestCase):
    def test_summarizes_smallest_quartile_at_best_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "run.json").write_text(json.dumps({
                "seed": 20260812,
                "patch_sampling": {"foreground_probability": 0.5},
            }))
            per_case = [
                {"case_id": "small", "whole_lesion_dice": 0.7},
                {"case_id": "large", "whole_lesion_dice": 0.9},
            ]
            rows = []
            for epoch in range(1, 11):
                dice = 0.8 if epoch == 7 else 0.6
                rows.append({
                    "epoch": epoch,
                    "validation": {
                        "mean_dice": dice,
                        "mean_derived_box_iou": 0.5,
                        "mean_hd95_mm": 10.0,
                        "per_case": per_case,
                    },
                })
            (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            (run / "external.json").write_text("{}\n")

            summary = summarize_run(run, {"small"})
            self.assertEqual(summary["best_epoch"], 7)
            self.assertEqual(summary["foreground_probability"], 0.5)
            self.assertEqual(summary["smallest_quartile_mean_dice"], 0.7)


if __name__ == "__main__":
    unittest.main()
