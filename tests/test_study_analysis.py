from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from brain_mri_data.study_analysis import analyze_study


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "config" / "analysis" / "glioma.yaml"


def result(profile: str, arm: str, seed: int, first: float, second: float) -> dict:
    return {
        "schema_version": 1,
        "run": {
            "study_id": "glioma", "evaluation_status": "external_test_locked",
            "profile_id": profile, "arm": arm, "seed": seed,
            "study_sha256": "study", "profile_sha256": "profile",
            "trainer_sha256": "trainer", "pamc_sha256": "pamc",
            "evaluation_sha256": "evaluation",
        },
        "external_clean": {"per_case": [
            {"case_id": "external:a", "whole_lesion_dice": first},
            {"case_id": "external:b", "whole_lesion_dice": second},
        ]},
    }


class StudyAnalysisTests(unittest.TestCase):
    def test_complete_seeded_results_emit_paired_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            values = {"brats": (0.60, 0.50), "pooled": (0.65, 0.55), "pamc": (0.70, 0.60)}
            for arm, (first, second) in values.items():
                for seed in (20260812, 20260813, 20260814):
                    path = root / f"cuda-{arm}-{seed}.json"
                    path.write_text(json.dumps(result("cuda", arm, seed, first, second)))
                    paths.append(path)
            output = root / "analysis.json"
            report = analyze_study(PLAN, paths, output)
            payload = json.loads(output.read_text())
            self.assertEqual(report["reports"], 3)
            primary = next(item for item in payload["reports"] if item["comparison"]["id"] == "pamc_vs_brats_primary")
            self.assertAlmostEqual(primary["mean_difference"], 0.1)
            self.assertEqual(primary["case_count"], 2)
            self.assertEqual(len(primary["per_case_mean_differences"]), 2)

    def test_incomplete_seed_set_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for arm in ("brats", "pooled", "pamc"):
                for seed in (20260812, 20260813):
                    path = root / f"cuda-{arm}-{seed}.json"
                    path.write_text(json.dumps(result("cuda", arm, seed, 0.6, 0.5)))
                    paths.append(path)
            with self.assertRaisesRegex(ValueError, "frozen analysis seeds"):
                analyze_study(PLAN, paths, root / "analysis.json")

    def test_mixed_code_or_study_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for arm in ("brats", "pooled", "pamc"):
                for seed in (20260812, 20260813, 20260814):
                    payload = result("cuda", arm, seed, 0.6, 0.5)
                    if arm == "pamc" and seed == 20260814:
                        payload["run"]["trainer_sha256"] = "different-trainer"
                    path = root / f"cuda-{arm}-{seed}.json"
                    path.write_text(json.dumps(payload))
                    paths.append(path)
            with self.assertRaisesRegex(ValueError, "locked study, profile, and evaluation code"):
                analyze_study(PLAN, paths, root / "analysis.json")
