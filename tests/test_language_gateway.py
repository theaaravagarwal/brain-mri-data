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
    deterministic_result_explanation,
    result_explainer_prompt,
    validate_explanation,
    validate_job_proposal,
    validate_result_explanation,
)
from brain_mri_data.language_contracts import ResearchSegmentationResultV1
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


def serving_result() -> ResearchSegmentationResultV1:
    return ResearchSegmentationResultV1.model_validate(
        {
            "schema_version": "research-segmentation-result/v1",
            "job_id": "65ecf1c3-ae23-4c40-ae7f-6aecc9453904",
            "study_id": "glioma",
            "protocol": "glioma_4seq_v1",
            "disclaimer": "Research output only; not a diagnosis or treatment recommendation.",
            "input_qc": {
                "schema_version": "research-study-validation/v1",
                "status": "pass",
                "modality_count": 4,
                "modalities": ["t1", "t1ce", "t2", "flair"],
                "geometry_match": True,
                "shape": [240, 240, 155],
                "spacing_mm": [1.0, 1.0, 1.0],
                "geometry_sha256": "a" * 64,
                "modality_sha256": {
                    "t1": "b" * 64,
                    "t1ce": "c" * 64,
                    "t2": "d" * 64,
                    "flair": "e" * 64,
                },
            },
            "segmentation": {
                "status": "complete",
                "output_sha256": "f" * 64,
                "output_shape": [240, 240, 155],
                "geometry_preserved": True,
                "labels": [0, 1],
                "label_count": 2,
                "nonzero_voxels": 42117,
            },
            "provenance": {
                "model_id": "glioma-segresnet-20260828",
                "model_scope": "internal_research_only",
                "checkpoint_sha256": "121422a861bbe7affaa5e161058e69eea737b2390651c3c03ea20256969e99e5",
                "training_git_revision": "570c65ac4709dac3b05f48314ddd5aef70589a7d",
                "study_sha256": "e53f85b429449585089133b1d9f680c3d80125b58da3042e5510522e2b333f6d",
                "profile_sha256": "9ec821920b6a08e914306d1651101dd52693d02c185f2750410297ec1c43fc7e",
                "trainer_sha256": "bf5dede3b5b1ee5d916cd6f046ca7eda8ea579f0f730db6f9201e2523b0456d9",
                "inference_script_sha256": "1" * 64,
                "device": "NVIDIA GeForce RTX 4060",
                "torch_version": "2.9.1+cu128",
                "monai_version": "1.6.0",
                "nibabel_version": "5.4.2",
                "generated_at": "2026-08-31T12:00:00Z",
            },
        }
    )


class LanguageGatewayTests(unittest.TestCase):
    def test_new_study_explanation_uses_metadata_without_accuracy_claims(self) -> None:
        result = serving_result()
        explanation = deterministic_result_explanation(result)
        self.assertIn("42117 non-zero output voxels", explanation["summary"])
        self.assertIn("No reference mask was supplied", explanation["limitations"])
        self.assertNotIn("dice", explanation["summary"].lower())
        prompt = result_explainer_prompt(result)
        self.assertNotIn("modality_sha256", prompt)
        self.assertNotIn("spacing_mm", prompt)
        self.assertIn(json.dumps(explanation["limitations"]), prompt)

    def test_new_study_explanation_requires_exact_evidence(self) -> None:
        result = serving_result()
        explanation = deterministic_result_explanation(result)
        self.assertEqual(validate_result_explanation(explanation, result), explanation)
        altered = json.loads(json.dumps(explanation))
        altered["evidence"][0]["value"] = "fail"
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_result_explanation(altered, result)
        altered = json.loads(json.dumps(explanation))
        altered["limitations"] = "800"
        with self.assertRaisesRegex(ValueError, "limitations"):
            validate_result_explanation(altered, result)
        altered = json.loads(json.dumps(explanation))
        altered["summary"] = "Different metadata summary."
        with self.assertRaisesRegex(ValueError, "summary"):
            validate_result_explanation(altered, result)

    def test_new_study_result_rejects_output_geometry_mismatch(self) -> None:
        value = serving_result().model_dump(mode="json")
        value["segmentation"]["output_shape"] = [240, 240, 154]
        with self.assertRaisesRegex(ValueError, "output shape"):
            ResearchSegmentationResultV1.model_validate(value)

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

        no_reference = record()
        no_reference["segmentation"].pop("whole_lesion_dice")
        no_reference_prompt = build_explainer_prompt(no_reference)
        self.assertNotIn('"segmentation.whole_lesion_dice"', no_reference_prompt)
        self.assertIn('"segmentation.status"', no_reference_prompt)

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
