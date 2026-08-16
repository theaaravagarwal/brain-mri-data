from __future__ import annotations

import tempfile
import unittest
import json
import copy
from pathlib import Path

from brain_mri_data.catalog import load_catalog
from brain_mri_data.study import _approved_manual_provenance_review, build_study_manifest


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "config" / "studies" / "glioma.yaml"
PILOT = ROOT / "config" / "studies" / "glioma-pilot.yaml"


class StudyGuardrailTests(unittest.TestCase):
    def test_pending_manual_review_cannot_lock_a_study(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "manual_provenance_review"):
                build_study_manifest(load_catalog(), STUDY, Path(temporary), Path(temporary) / "locked.json")

    def test_manual_review_needs_attributable_dated_evidence(self) -> None:
        study = {"manual_provenance_review": {
            "status": "approved", "reviewer": "reviewer", "completed_at_utc": "2026-08-15T12:00:00Z", "evidence": "audit-notes.md",
        }}
        self.assertEqual(_approved_manual_provenance_review(study)["reviewer"], "reviewer")
        for field in ("reviewer", "completed_at_utc", "evidence"):
            invalid = copy.deepcopy(study)
            invalid["manual_provenance_review"][field] = ""
            with self.assertRaisesRegex(ValueError, field):
                _approved_manual_provenance_review(invalid)
        invalid = copy.deepcopy(study)
        invalid["manual_provenance_review"]["completed_at_utc"] = "2026-08-15"
        with self.assertRaisesRegex(ValueError, "timezone"):
            _approved_manual_provenance_review(invalid)

    def test_internal_pilot_locks_without_an_external_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [
                {"case_id": f"pilot-{index:03d}", "patient_id": f"pilot-{index:03d}"}
                for index in range(40)
            ]
            (root / "brats2020_kaggle.cases.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            (root / "brats2020_kaggle.qc.jsonl").write_text(
                "".join(json.dumps({"case_id": record["case_id"], "valid": True}) + "\n" for record in records)
            )
            output = root / "pilot.json"
            result = build_study_manifest(load_catalog(), PILOT, root, output)
            payload = json.loads(output.read_text())
            self.assertEqual(result["locked_test_cases"], 0)
            self.assertEqual(payload["evaluation_status"], "pilot_internal_only")
            self.assertEqual(payload["external_test"], [])
            self.assertEqual(payload["study"]["training"]["architecture"], "monai_segresnet")
            self.assertEqual(payload["study"]["training"]["init_filters"], 32)
