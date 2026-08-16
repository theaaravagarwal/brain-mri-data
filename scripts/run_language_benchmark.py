#!/usr/bin/env python3
"""Run a frozen local Ollama model on a constrained language benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

from brain_mri_data.language_bench import read_jsonl, score_evidence, score_structured
from brain_mri_data.language_gateway import build_explainer_prompt

STRUCTURED_SCHEMA = {
    "type": "object",
    "properties": {
        "disclaimer": {
            "type": "string",
            "const": "Research output only; not a diagnosis or treatment recommendation.",
        },
        "summary": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": [
                            "input_qc.status",
                            "segmentation.status",
                            "segmentation.whole_lesion_dice",
                            "provenance.source_id",
                        ],
                    },
                    "value": {
                        "type": ["string", "number", "boolean", "null"],
                    },
                },
                "required": ["field", "value"],
                "additionalProperties": False,
            },
        },
        "limitations": {"type": "string"},
        "abstained": {"type": "boolean"},
    },
    "required": ["disclaimer", "summary", "evidence", "limitations", "abstained"],
    "additionalProperties": False,
}

EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "citations"],
    "additionalProperties": False,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("structured", "evidence"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=Path("benchmarks/language/evidence-sources.json"))
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ask(host: str, model: str, prompt: str, schema: dict) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "think": False,
        "options": {"temperature": 0},
    }
    request = Request(
        host.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        result = json.loads(response.read())
    return json.loads(result["response"])


def evidence_prompt(fixture: dict, sources: dict) -> str:
    allowed = {source_id: sources[source_id] for source_id in fixture["allowed_source_ids"]}
    return (
        "Answer only from the source cards. Return JSON with answer and citations. "
        "citations must contain only source-card IDs. Do not make medical claims.\n"
        f"Question: {fixture['question']}\nSource cards: {json.dumps(allowed, sort_keys=True)}"
    )


def main() -> None:
    args = arguments()
    fixtures = read_jsonl(args.fixtures)
    sources = json.loads(args.evidence.read_text()) if args.kind == "evidence" else {}
    schema = STRUCTURED_SCHEMA if args.kind == "structured" else EVIDENCE_SCHEMA
    output = []
    for fixture in fixtures:
        prompt = build_explainer_prompt(fixture["record"]) if args.kind == "structured" else evidence_prompt(fixture, sources)
        if args.dry_run:
            print(json.dumps({"id": fixture["id"], "prompt": prompt}, sort_keys=True))
            continue
        output.append({"id": fixture["id"], "response": ask(args.host, args.model, prompt, schema)})
    if args.dry_run:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in output))
    score = score_structured(args.fixtures, args.output) if args.kind == "structured" else score_evidence(args.fixtures, args.output)
    print(json.dumps(score, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
