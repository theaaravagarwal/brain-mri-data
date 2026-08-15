from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from brain_mri_data.monitor import render, snapshot


class MonitorTests(unittest.TestCase):
    def test_snapshot_and_render_use_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "run.json").write_text(json.dumps({
                "profile_id": "cuda", "arm": "brats", "seed": 20260812, "epochs": 10,
                "hardware": {"device_name": "NVIDIA GeForce RTX 3060"},
            }))
            (run / "metrics.jsonl").write_text(json.dumps({
                "epoch": 1, "train_loss": 0.5,
                "validation": {"mean_dice": 0.7, "mean_derived_box_iou": 0.6, "mean_hd95_mm": 8.0},
            }) + "\n")
            with patch("brain_mri_data.monitor._gpu", return_value={
                "name": "NVIDIA GeForce RTX 3060", "utilization": "99", "memory_used": "4000",
                "memory_total": "12288", "temperature": "60", "power": "120",
            }), patch("brain_mri_data.monitor._memory", return_value={"used_gib": 10.0, "total_gib": 32.0}), patch(
                "brain_mri_data.monitor._process_lines", return_value=[f"1 training/train_glioma.py --output {run}"]
            ):
                data = snapshot(run)
            self.assertEqual(data["status"], "running")
            self.assertEqual(data["last_metric"]["epoch"], 1)
            output = render(data, color=False)
            self.assertIn("CUDA monitor", output)
            self.assertIn("Validation Dice 0.7000", output)
            self.assertNotIn("\x1b[", output)
