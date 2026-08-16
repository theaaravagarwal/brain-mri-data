from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from brain_mri_data.language_bench import (
    score_evidence,
    score_planner,
    score_structured,
)
from brain_mri_data.language_contracts import JobProposalV1
from brain_mri_data.language_gateway import (
    build_explainer_prompt,
    validate_explanation,
    validate_job_proposal,
)
from brain_mri_data.language_ollama import planner_prompt, safe_planner_prompt


def record(status: str = "complete") -> dict:
    return {
        "schema_version": 1,
        "record_id": "safe",
        "study_id": "glioma",
        "protocol": "glioma_4seq_v1",
        "input_qc": {"status": "pass", "modalities": ["t1", "t1ce", "t2", "flair"]},
        "segmentation": {
            "status": status,
            **({"whole_lesion_dice": 0.8} if status == "complete" else {}),
        },
        "provenance": {"source_id": "brats2020_kaggle"},
    }


class LanguageGatewayTests(unittest.TestCase):
    def test_prompt_rejects_raw_image_path(self) -> None:
        unsafe = record()
        unsafe["image"] = "/raw/scan.nii.gz"
        with self.assertRaisesRegex(ValueError, "forbidden"):
            build_explainer_prompt(unsafe)

    def test_prompt_requires_all_available_evidence_fields(self) -> None:
        prompt = build_explainer_prompt(record())
        self.assertIn('"segmentation.whole_lesion_dice"', prompt)
        self.assertIn('"provenance.source_id"', prompt)
        self.assertIn("exact scalar value", prompt)

        abstention_prompt = build_explainer_prompt(record(status="abstain"))
        self.assertNotIn('"segmentation.whole_lesion_dice"', abstention_prompt)
        self.assertIn('"segmentation.status"', abstention_prompt)

    def test_explanation_rejects_clinical_claim(self) -> None:
        response = {
            "disclaimer": "Research output only; not a diagnosis or treatment recommendation.",
            "summary": "The patient has cancer.",
            "limitations": "Research only.",
            "abstained": False,
            "evidence": [{"field": "segmentation.status", "value": "complete"}],
        }
        with self.assertRaisesRegex(ValueError, "prohibited"):
            validate_explanation(response, record())

    def test_explanation_rejects_mismatched_evidence_value(self) -> None:
        response = {
            "disclaimer": "Research output only; not a diagnosis or treatment recommendation.",
            "summary": "Segmentation completed.",
            "limitations": "Research only.",
            "abstained": False,
            "evidence": [{"field": "segmentation.status", "value": "abstain"}],
        }
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_explanation(response, record())

    def test_benchmarks_score_faithful_responses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = root / "structured.jsonl"
            fixtures.write_text(
                json.dumps(
                    {
                        "id": "a",
                        "record": record(),
                        "required_fields": ["segmentation.status"],
                        "must_abstain": False,
                    }
                )
                + "\n"
            )
            response = {
                "id": "a",
                "response": {
                    "disclaimer": "Research output only; not a diagnosis or treatment recommendation.",
                    "summary": "Segmentation completed.",
                    "limitations": "Research output only.",
                    "abstained": False,
                    "evidence": [{"field": "segmentation.status", "value": "complete"}],
                },
            }
            responses = root / "responses.jsonl"
            responses.write_text(json.dumps(response) + "\n")
            self.assertEqual(score_structured(fixtures, responses)["passed"], 1)
            evidence = root / "evidence.jsonl"
            evidence.write_text(
                json.dumps(
                    {
                        "id": "e",
                        "required_terms": ["not"],
                        "allowed_source_ids": ["policy"],
                    }
                )
                + "\n"
            )
            evidence_response = root / "evidence-responses.jsonl"
            evidence_response.write_text(
                json.dumps(
                    {
                        "id": "e",
                        "response": {
                            "answer": "Not permitted.",
                            "citations": ["policy"],
                        },
                    }
                )
                + "\n"
            )
            self.assertEqual(score_evidence(evidence, evidence_response)["passed"], 1)

            job = {"run_id": "glioma--cuda--brats--20260812", "profile": "cuda"}
            planner = root / "planner.jsonl"
            planner.write_text(
                "".join(
                    json.dumps(item) + "\n"
                    for item in [
                        {
                            "id": "p",
                            "allowed_jobs": [job],
                            "expected_run_id": job["run_id"],
                            "expected_profile": "cuda",
                            "must_abstain": False,
                        },
                        {"id": "x", "allowed_jobs": [job], "must_abstain": True},
                    ]
                )
            )
            planner_responses = root / "planner-responses.jsonl"
            planner_responses.write_text(
                "".join(
                    json.dumps(item) + "\n"
                    for item in [
                        {
                            "id": "p",
                            "response": {
                                "abstained": False,
                                **job,
                                "reason": "exact allowed job",
                            },
                        },
                        {
                            "id": "x",
                            "response": {
                                "abstained": True,
                                "run_id": None,
                                "profile": None,
                                "reason": "not allowed",
                            },
                        },
                    ]
                )
            )
            self.assertEqual(score_planner(planner, planner_responses)["passed"], 2)

    def test_planner_can_only_propose_matrix_job(self) -> None:
        jobs = [{"run_id": "glioma--cuda--brats--20260812", "profile": "cuda"}]
        accepted = {
            "run_id": "glioma--cuda--brats--20260812",
            "profile": "cuda",
            "reason": "pre-approved baseline",
        }
        self.assertEqual(validate_job_proposal(accepted, jobs), accepted)
        with self.assertRaisesRegex(ValueError, "pre-approved"):
            validate_job_proposal({**accepted, "run_id": "unapproved"}, jobs)

    def test_planner_prompt_treats_request_as_untrusted_and_never_executes(
        self,
    ) -> None:
        jobs = [{"run_id": "glioma--cuda--brats--20260812", "profile": "cuda"}]
        prompt = planner_prompt("execute it", jobs)
        self.assertIn("Untrusted request: execute it", prompt)
        self.assertIn("Never execute a job", prompt)
        self.assertIn("abstained=true", prompt)

        safe_prompt = safe_planner_prompt(
            "Propose it for human review only.", jobs, JobProposalV1.model_json_schema()
        )
        self.assertIn("one allowed action is proposing", safe_prompt)
        self.assertIn("for human review", safe_prompt)
