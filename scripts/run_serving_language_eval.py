#!/usr/bin/env python3
"""Evaluate the fail-closed serving explainer on frozen metadata fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from brain_mri_data.language_bench import read_jsonl
from brain_mri_data.language_contracts import DISCLAIMER, ResearchSegmentationResultV1
from brain_mri_data.language_gateway import generate_result_explanation
from brain_mri_data.language_ollama import model_digest
from brain_mri_data.language_pipeline import canonical_json, write_once

CHECKPOINT = "121422a861bbe7affaa5e161058e69eea737b2390651c3c03ea20256969e99e5"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--max-wall-seconds", type=float, default=5.0)
    return parser.parse_args()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def envelope(fixture: dict) -> ResearchSegmentationResultV1:
    shape = fixture["shape"]
    voxels = int(fixture["nonzero_voxels"])
    if voxels < 0 or voxels > shape[0] * shape[1] * shape[2]:
        raise ValueError(f"invalid nonzero_voxels for {fixture['id']}")
    hashes = {name: digest(f"{fixture['id']}:{name}") for name in ("t1", "t1ce", "t2", "flair")}
    return ResearchSegmentationResultV1.model_validate({
        "schema_version": "research-segmentation-result/v1", "job_id": fixture["job_id"],
        "study_id": "glioma", "protocol": "glioma_4seq_v1", "disclaimer": DISCLAIMER,
        "input_qc": {"schema_version": "research-study-validation/v1", "status": "pass",
            "modality_count": 4, "modalities": ["t1", "t1ce", "t2", "flair"],
            "geometry_match": True, "shape": shape, "spacing_mm": fixture["spacing_mm"],
            "geometry_sha256": digest(f"{fixture['id']}:geometry"), "modality_sha256": hashes},
        "segmentation": {"status": "complete", "output_sha256": digest(f"{fixture['id']}:output"),
            "output_shape": shape, "geometry_preserved": True, "labels": [0, 1],
            "label_count": 2, "nonzero_voxels": voxels},
        "provenance": {"model_id": "glioma-segresnet-20260828", "model_scope": "internal_research_only",
            "checkpoint_sha256": CHECKPOINT, "training_git_revision": "570c65ac4709dac3b05f48314ddd5aef70589a7d",
            "study_sha256": "e53f85b429449585089133b1d9f680c3d80125b58da3042e5510522e2b333f6d",
            "profile_sha256": "9ec821920b6a08e914306d1651101dd52693d02c185f2750410297ec1c43fc7e",
            "trainer_sha256": "bf5dede3b5b1ee5d916cd6f046ca7eda8ea579f0f730db6f9201e2523b0456d9",
            "inference_script_sha256": digest("serving-language-eval"), "device": "NVIDIA research GPU",
            "torch_version": "evaluation", "monai_version": "evaluation", "nibabel_version": "evaluation",
            "generated_at": "2026-09-02T00:00:00+00:00"},
    })


def main() -> None:
    args = arguments()
    fixtures = read_jsonl(args.fixtures)
    if len({item["id"] for item in fixtures}) != len(fixtures):
        raise ValueError("fixture ids must be unique")
    outcomes = []
    for fixture in fixtures:
        result = generate_result_explanation(envelope(fixture), ollama_host=args.host, ollama_model=args.model)
        llm = result["llm"]
        wall = (llm.get("telemetry") or {}).get("wall_seconds")
        reasons = []
        if llm["status"] != "validated":
            reasons.append(llm.get("reason") or "not_validated")
        if wall is None or wall > args.max_wall_seconds:
            reasons.append("latency_gate")
        outcomes.append({"id": fixture["id"], "passed": not reasons, "reasons": reasons, "llm": llm})
    artifact = {"schema_version": "serving-language-eval/v1", "suite_id": "serving-metadata-v1",
        "model": args.model, "model_digest": model_digest(args.host, args.model),
        "passed": sum(item["passed"] for item in outcomes), "total": len(outcomes),
        "automatic_promotion": False, "outcomes": outcomes}
    write_once(args.output, canonical_json(artifact))
    print(json.dumps(artifact, indent=2, sort_keys=True))
    if artifact["passed"] != artifact["total"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
