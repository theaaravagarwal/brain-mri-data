from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalog import get_source, load_catalog, project_root
from .experiment_audit import audit_experiment
from .fetch import fetch_automatic, fetch_source
from .indexer import discover_source, index_source, read_jsonl, resolve_case_path
from .qc import validate_cases
from .splits import make_split
from .study import build_study_manifest
from .study_analysis import analyze_study
from .run_matrix import claim_run, expand_matrix
from .language_bench import score_evidence, score_structured
from .language_gateway import build_explainer_prompt, load_language_policy, validate_job_proposal, validate_result_envelope


def paths(args: argparse.Namespace) -> tuple[Path, Path]:
    data_root = Path(args.data_root).resolve()
    return data_root / "raw", data_root / "manifests"


def main() -> None:
    parser = argparse.ArgumentParser(description="Four-sequence brain MRI dataset aggregator")
    parser.add_argument("--data-root", default=str(project_root() / "data"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog")
    fetch = sub.add_parser("fetch"); fetch.add_argument("source_id"); fetch.add_argument("--dry-run", action="store_true")
    fetch_auto = sub.add_parser("fetch-auto", help="sequentially fetch Kaggle and Hugging Face sources only")
    fetch_auto.add_argument("--only", nargs="+", help="automatic source IDs to fetch instead of all automatic sources")
    fetch_auto.add_argument("--dry-run", action="store_true")
    index = sub.add_parser("index"); index.add_argument("source_id")
    discover = sub.add_parser("discover"); discover.add_argument("source_id")
    validate = sub.add_parser("validate"); validate.add_argument("source_id")
    split = sub.add_parser("split"); split.add_argument("source_id"); split.add_argument("--seed", type=int, required=True)
    export = sub.add_parser("export-monai"); export.add_argument("source_id")
    external = sub.add_parser("export-external-monai")
    external.add_argument("source_id", help="export accepted cases as a locked external test datalist")
    audit = sub.add_parser("audit-experiment")
    audit.add_argument("--train", nargs="+", required=True)
    audit.add_argument("--test", nargs="+", required=True)
    audit.add_argument("--strict", action="store_true", help="exit nonzero when the design is blocked")
    study = sub.add_parser("build-study")
    study.add_argument("study_config")
    study.add_argument("--output", required=True, help="destination under the manifest root")
    analyze = sub.add_parser("analyze-study", help="run the frozen paired external-study analysis")
    analyze.add_argument("analysis_plan")
    analyze.add_argument("results", nargs="+", help="external.json artifacts from completed frozen runs")
    analyze.add_argument("--output", required=True)
    runs = sub.add_parser("runs")
    run_sub = runs.add_subparsers(dest="runs_command", required=True)
    run_list = run_sub.add_parser("list"); run_list.add_argument("matrix_config")
    run_claim = run_sub.add_parser("claim")
    run_claim.add_argument("matrix_config"); run_claim.add_argument("run_id")
    run_claim.add_argument("--profile", required=True)
    language = sub.add_parser("language")
    language_sub = language.add_subparsers(dest="language_command", required=True)
    language_policy = language_sub.add_parser("policy")
    language_policy.add_argument("--config", default="config/language.yaml")
    language_validate = language_sub.add_parser("validate")
    language_validate.add_argument("result")
    language_prompt = language_sub.add_parser("prompt")
    language_prompt.add_argument("result")
    language_score = language_sub.add_parser("score")
    language_score.add_argument("kind", choices=("structured", "evidence"))
    language_score.add_argument("fixtures")
    language_score.add_argument("responses")
    language_propose = language_sub.add_parser("propose-job")
    language_propose.add_argument("proposal", help="JSON proposal with run_id, profile, and reason")
    language_propose.add_argument("matrix", help="pre-approved run matrix YAML")
    args = parser.parse_args()
    catalog = load_catalog()
    raw_root, manifest_root = paths(args)
    if args.command == "catalog":
        print(json.dumps(catalog, indent=2, sort_keys=True)); return
    if args.command == "fetch-auto":
        try:
            results = fetch_automatic(catalog, raw_root, args.dry_run, args.only)
        except (KeyError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(results, indent=2, sort_keys=True)); return
    if args.command == "audit-experiment":
        unknown = sorted(set(args.train + args.test) - set(catalog["sources"]))
        if unknown:
            parser.error("Unknown source IDs: " + ", ".join(unknown))
        report = audit_experiment(catalog, args.train, args.test, manifest_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.strict and report["status"] == "blocked":
            raise SystemExit(2)
        return
    if args.command == "build-study":
        output = Path(args.output)
        if output.is_absolute() or ".." in output.parts:
            parser.error("--output must be a path relative to the manifest root")
        report = build_study_manifest(catalog, Path(args.study_config), manifest_root, manifest_root / output)
        print(json.dumps(report, indent=2, sort_keys=True)); return
    if args.command == "analyze-study":
        output = Path(args.output)
        if output.is_absolute() or ".." in output.parts:
            parser.error("--output must be a path relative to the data root")
        try:
            report = analyze_study(Path(args.analysis_plan), [Path(path) for path in args.results], data_root / output)
        except (FileExistsError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(report, indent=2, sort_keys=True)); return
    if args.command == "runs":
        matrix_path = Path(args.matrix_config)
        if args.runs_command == "list":
            print(json.dumps(expand_matrix(matrix_path), indent=2, sort_keys=True)); return
        report = claim_run(matrix_path, args.run_id, args.profile, data_root / "experiments")
        print(json.dumps(report, indent=2, sort_keys=True)); return
    if args.command == "language":
        if args.language_command == "policy":
            print(json.dumps(load_language_policy(Path(args.config)), indent=2, sort_keys=True)); return
        payload = json.loads(Path(args.result).read_text()) if args.language_command in {"validate", "prompt"} else None
        if args.language_command == "validate":
            print(json.dumps(validate_result_envelope(payload), indent=2, sort_keys=True)); return
        if args.language_command == "prompt":
            print(build_explainer_prompt(payload)); return
        if args.language_command == "propose-job":
            proposal = json.loads(Path(args.proposal).read_text())
            print(json.dumps(validate_job_proposal(proposal, expand_matrix(Path(args.matrix))), indent=2, sort_keys=True)); return
        scorer = score_structured if args.kind == "structured" else score_evidence
        print(json.dumps(scorer(Path(args.fixtures), Path(args.responses)), indent=2, sort_keys=True)); return
    source = get_source(args.source_id, catalog)
    if args.command == "fetch":
        print(fetch_source(args.source_id, source, raw_root, args.dry_run)); return
    if args.command == "discover":
        print(json.dumps(discover_source(args.source_id, source, raw_root), indent=2, sort_keys=True)); return
    if args.command == "index":
        path, accepted, rejected = index_source(args.source_id, source, raw_root, manifest_root)
        print(f"{path}: accepted={accepted} rejected={rejected}"); return
    cases_path = manifest_root / f"{args.source_id}.cases.jsonl"
    if args.command == "validate":
        results = validate_cases(cases_path, raw_root, source)
        out = manifest_root / f"{args.source_id}.qc.jsonl"
        out.write_text("".join(json.dumps(result, sort_keys=True) + "\n" for result in results))
        print(f"{out}: valid={sum(item['valid'] for item in results)}/{len(results)}"); return
    if args.command == "split":
        out = manifest_root / f"{args.source_id}.splits.jsonl"
        print(json.dumps(make_split(cases_path, out, args.seed), sort_keys=True)); return
    if args.command == "export-external-monai":
        order = source["modalities"]
        data = {
            "testing": [
                {
                    "image": [str(resolve_case_path(case, case["modalities"][key], raw_root)) for key in order],
                    "label": str(resolve_case_path(case, case["segmentation"], raw_root)),
                }
                for case in read_jsonl(manifest_root / f"{args.source_id}.cases.jsonl")
            ]
        }
        out = manifest_root / f"{args.source_id}.external.monai.json"
        out.write_text(json.dumps(data, indent=2) + "\n")
        print(out); return
    if args.command == "export-monai":
        split_path = manifest_root / f"{args.source_id}.splits.jsonl"
        labels = {item["case_id"]: item["split"] for item in read_jsonl(split_path)}
        data = {"training": [], "validation": [], "test": []}
        for case in read_jsonl(cases_path):
            order = get_source(args.source_id, catalog)["modalities"]
            item = {
                "image": [str(resolve_case_path(case, case["modalities"][key], raw_root)) for key in order],
                "label": str(resolve_case_path(case, case["segmentation"], raw_root)),
            }
            target = {"train": "training", "val": "validation", "test": "test"}[labels[case["case_id"]]]
            data[target].append(item)
        out = manifest_root / f"{args.source_id}.monai.json"
        out.write_text(json.dumps(data, indent=2) + "\n")
        print(out)


if __name__ == "__main__":
    main()
