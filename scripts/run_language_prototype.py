#!/usr/bin/env python3
"""Run the constrained explainer or read-only planner against local Ollama."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from brain_mri_data.language_gateway import build_explainer_prompt, validate_explanation, validate_job_proposal
from brain_mri_data.language_ollama import PLANNER_SCHEMA, STRUCTURED_SCHEMA, ask_ollama, planner_prompt
from brain_mri_data.run_matrix import expand_matrix


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--output", type=Path, required=True)
    sub = parser.add_subparsers(dest="role", required=True)
    explain = sub.add_parser("explain")
    explain.add_argument("--result", type=Path, required=True, help="validated non-identifying CNN result envelope")
    explain.add_argument("--model", default="qwen3:14b")
    plan = sub.add_parser("plan")
    plan.add_argument("--request-file", type=Path, required=True, help="untrusted plain-text planning request")
    plan.add_argument("--matrix", type=Path, default=Path("config/run-matrix/glioma.yaml"))
    plan.add_argument("--model", default="qwen3-coder:30b")
    return parser.parse_args()


def write_once(path: Path, artifact: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as file:
            json.dump(artifact, file, indent=2, sort_keys=True)
            file.write("\n")
    except FileExistsError as error:
        raise SystemExit(f"Refusing to overwrite prototype artifact: {path}") from error


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite prototype artifact: {args.output}")
    if args.role == "explain":
        envelope = json.loads(args.result.read_text())
        response, telemetry = ask_ollama(args.host, args.model, build_explainer_prompt(envelope), STRUCTURED_SCHEMA)
        validate_explanation(response, envelope)
        artifact = {"schema_version": 1, "role": "explainer", "model": args.model, "input_record_id": envelope["record_id"], "executed": False, "response": response, "telemetry": telemetry}
    else:
        allowed_jobs = expand_matrix(args.matrix)
        request_text = args.request_file.read_text().strip()
        response, telemetry = ask_ollama(args.host, args.model, planner_prompt(request_text, allowed_jobs), PLANNER_SCHEMA)
        if response["abstained"]:
            if response["run_id"] is not None or response["profile"] is not None:
                raise ValueError("Abstaining planner must not return a job")
        else:
            validate_job_proposal({key: response[key] for key in ("run_id", "profile", "reason")}, allowed_jobs)
        artifact = {"schema_version": 1, "role": "planner", "model": args.model, "executed": False, "response": response, "telemetry": telemetry}
    write_once(args.output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
