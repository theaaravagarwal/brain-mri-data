from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalog import get_source, load_catalog, project_root
from .experiment_audit import audit_experiment
from .fetch import fetch_source
from .indexer import discover_source, index_source, read_jsonl
from .qc import validate_cases
from .splits import make_split


def paths(args: argparse.Namespace) -> tuple[Path, Path]:
    data_root = Path(args.data_root).resolve()
    return data_root / "raw", data_root / "manifests"


def main() -> None:
    parser = argparse.ArgumentParser(description="Four-sequence brain MRI dataset aggregator")
    parser.add_argument("--data-root", default=str(project_root() / "data"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog")
    fetch = sub.add_parser("fetch"); fetch.add_argument("source_id"); fetch.add_argument("--dry-run", action="store_true")
    index = sub.add_parser("index"); index.add_argument("source_id")
    discover = sub.add_parser("discover"); discover.add_argument("source_id")
    validate = sub.add_parser("validate"); validate.add_argument("source_id")
    split = sub.add_parser("split"); split.add_argument("source_id"); split.add_argument("--seed", type=int, required=True)
    export = sub.add_parser("export-monai"); export.add_argument("source_id")
    audit = sub.add_parser("audit-experiment")
    audit.add_argument("--train", nargs="+", required=True)
    audit.add_argument("--test", nargs="+", required=True)
    audit.add_argument("--strict", action="store_true", help="exit nonzero when the design is blocked")
    args = parser.parse_args()
    catalog = load_catalog()
    raw_root, manifest_root = paths(args)
    if args.command == "catalog":
        print(json.dumps(catalog, indent=2, sort_keys=True)); return
    if args.command == "audit-experiment":
        unknown = sorted(set(args.train + args.test) - set(catalog["sources"]))
        if unknown:
            parser.error("Unknown source IDs: " + ", ".join(unknown))
        report = audit_experiment(catalog, args.train, args.test, manifest_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.strict and report["status"] == "blocked":
            raise SystemExit(2)
        return
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
        results = validate_cases(cases_path)
        out = manifest_root / f"{args.source_id}.qc.jsonl"
        out.write_text("".join(json.dumps(result, sort_keys=True) + "\n" for result in results))
        print(f"{out}: valid={sum(item['valid'] for item in results)}/{len(results)}"); return
    if args.command == "split":
        out = manifest_root / f"{args.source_id}.splits.jsonl"
        print(json.dumps(make_split(cases_path, out, args.seed), sort_keys=True)); return
    if args.command == "export-monai":
        split_path = manifest_root / f"{args.source_id}.splits.jsonl"
        labels = {item["case_id"]: item["split"] for item in read_jsonl(split_path)}
        data = {"training": [], "validation": [], "test": []}
        for case in read_jsonl(cases_path):
            order = get_source(args.source_id, catalog)["modalities"]
            item = {"image": [case["modalities"][key] for key in order], "label": case["segmentation"]}
            target = {"train": "training", "val": "validation", "test": "test"}[labels[case["case_id"]]]
            data[target].append(item)
        out = manifest_root / f"{args.source_id}.monai.json"
        out.write_text(json.dumps(data, indent=2) + "\n")
        print(out)


if __name__ == "__main__":
    main()
