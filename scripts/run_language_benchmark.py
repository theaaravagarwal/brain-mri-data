#!/usr/bin/env python3
"""Run a frozen local Ollama model on a constrained language benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from brain_mri_data.language_bench import read_jsonl, score_evidence, score_planner, score_structured
from brain_mri_data.language_gateway import build_explainer_prompt
from brain_mri_data.language_ollama import EVIDENCE_SCHEMA, PLANNER_SCHEMA, STRUCTURED_SCHEMA, ask_ollama, planner_prompt


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("structured", "evidence", "planner"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=Path("benchmarks/language/evidence-sources.json"))
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def evidence_prompt(fixture: dict, sources: dict) -> str:
    allowed = {source_id: sources[source_id] for source_id in fixture["allowed_source_ids"]}
    return (
        "Answer only from the source cards. Return JSON with answer and citations. "
        "citations must contain only source-card IDs. Do not make medical claims.\n"
        f"Question: {fixture['question']}\nSource cards: {json.dumps(allowed, sort_keys=True)}"
    )


def main() -> None:
    args = arguments()
    if not args.dry_run and args.output.exists():
        raise SystemExit(f"Refusing to overwrite immutable benchmark output: {args.output}")
    fixtures = read_jsonl(args.fixtures)
    sources = json.loads(args.evidence.read_text()) if args.kind == "evidence" else {}
    schemas = {"structured": STRUCTURED_SCHEMA, "evidence": EVIDENCE_SCHEMA, "planner": PLANNER_SCHEMA}
    schema = schemas[args.kind]
    output = []
    for fixture in fixtures:
        if args.kind == "structured":
            prompt = build_explainer_prompt(fixture["record"])
        elif args.kind == "evidence":
            prompt = evidence_prompt(fixture, sources)
        else:
            prompt = planner_prompt(fixture["request"], fixture["allowed_jobs"])
        if args.dry_run:
            print(json.dumps({"id": fixture["id"], "prompt": prompt}, sort_keys=True))
            continue
        response, telemetry = ask_ollama(args.host, args.model, prompt, schema)
        output.append({"id": fixture["id"], "model": args.model, "response": response, "telemetry": telemetry})
    if args.dry_run:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x") as file:
            file.write("".join(json.dumps(item, sort_keys=True) + "\n" for item in output))
    except FileExistsError as error:
        raise SystemExit(f"Refusing to overwrite immutable benchmark output: {args.output}") from error
    scorers = {"structured": score_structured, "evidence": score_evidence, "planner": score_planner}
    score = scorers[args.kind](args.fixtures, args.output)
    score["model"] = args.model
    score["performance"] = {
        "mean_wall_seconds": round(sum(item["telemetry"]["wall_seconds"] for item in output) / len(output), 6),
        "mean_eval_tokens_per_second": round(
            sum(item["telemetry"]["eval_tokens_per_second"] or 0 for item in output) / len(output), 3
        ),
        "total_eval_tokens": sum(item["telemetry"]["eval_count"] for item in output),
    }
    print(json.dumps(score, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
