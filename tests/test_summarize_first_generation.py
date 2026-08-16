import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FirstGenerationSummaryTests(unittest.TestCase):
    def test_summary_uses_trainer_validation_metric_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = root / "study.json"
            study.write_text(json.dumps({
                "evaluation_status": "pilot_internal_only",
                "study": {"study_id": "glioma"},
            }))
            runs = []
            for offset, seed in enumerate((20260812, 20260813, 20260814)):
                run = root / f"run-{seed}"
                run.mkdir()
                (run / "run.json").write_text(json.dumps({"seed": seed}))
                (run / "external.json").write_text(json.dumps({"external_evaluation": None}))
                validation = {
                    "mean_dice": 0.7 + offset * 0.01,
                    "mean_derived_box_iou": 0.6 + offset * 0.01,
                    "mean_hd95_mm": 8.0 - offset,
                }
                (run / "metrics.jsonl").write_text(json.dumps({"epoch": 1, "validation": validation}) + "\n")
                runs.append(run)

            output = root / "summary.json"
            subprocess.run([
                sys.executable,
                str(ROOT / "scripts" / "summarize_first_generation.py"),
                "--study", str(study),
                "--output", str(output),
                *(str(run) for run in runs),
            ], check=True, capture_output=True, text=True)

            summary = json.loads(output.read_text())
            self.assertEqual(summary["runs"][0]["mean_derived_box_iou"], 0.6)
            self.assertEqual(summary["runs"][0]["mean_hd95_mm"], 8.0)
            self.assertAlmostEqual(summary["summary"]["best_validation_derived_box_iou_mean"], 0.61)


if __name__ == "__main__":
    unittest.main()
