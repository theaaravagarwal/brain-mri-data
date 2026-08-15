from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from brain_mri_data.language_bench import score_evidence, score_structured
from brain_mri_data.language_gateway import build_explainer_prompt, validate_explanation, validate_result_envelope


def record(status: str = "complete") -> dict:
    return {
        "schema_version": 1, "record_id": "safe", "study_id": "glioma", "protocol": "glioma_4seq_v1",
        "input_qc": {"status": "pass", "modalities": ["t1", "t1ce", "t2", "flair"]},
        "segmentation": {"status": status, **({"whole_lesion_dice": 0.8} if status == "complete" else {})},
        "provenance": {"source_id": "brats2020_kaggle"},
    }


class LanguageGatewayTests(unittest.TestCase):
    def test_prompt_rejects_raw_image_path(self) -> None:
        unsafe = record(); unsafe["image"] = "/raw/scan.nii.gz"
        with self.assertRaisesRegex(ValueError, "forbidden"):
            build_explainer_prompt(unsafe)

    def test_explanation_rejects_clinical_claim(self) -> None:
        response = {
            "disclaimer": "Research output only; not a diagnosis or treatment recommendation.",
            "summary": "The patient has cancer.", "limitations": "Research only.", "abstained": False,
            "evidence": [{"field": "segmentation.status", "value": "complete"}],
        }
        with self.assertRaisesRegex(ValueError, "prohibited"):
            validate_explanation(response, record())

    def test_benchmarks_score_faithful_responses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = root / "structured.jsonl"
            fixtures.write_text(json.dumps({"id": "a", "record": record(), "required_fields": ["segmentation.status"], "must_abstain": False}) + "\n")
            response = {"id": "a", "response": {
                "disclaimer": "Research output only; not a diagnosis or treatment recommendation.",
                "summary": "Segmentation completed.", "limitations": "Research output only.", "abstained": False,
                "evidence": [{"field": "segmentation.status", "value": "complete"}],
            }}
            responses = root / "responses.jsonl"; responses.write_text(json.dumps(response) + "\n")
            self.assertEqual(score_structured(fixtures, responses)["passed"], 1)
            evidence = root / "evidence.jsonl"; evidence.write_text(json.dumps({"id": "e", "required_terms": ["not"], "allowed_source_ids": ["policy"]}) + "\n")
            evidence_response = root / "evidence-responses.jsonl"; evidence_response.write_text(json.dumps({"id": "e", "response": {"answer": "Not permitted.", "citations": ["policy"]}}) + "\n")
            self.assertEqual(score_evidence(evidence, evidence_response)["passed"], 1)
