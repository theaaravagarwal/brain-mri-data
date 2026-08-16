#!/usr/bin/env python3
"""Create an immutable positive-chunk index that reuses an existing training cache."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.chunk_cache import load_cache_records
from training.train_glioma import file_sha256, manifest_items


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("brats", "pooled", "pamc"), default="brats")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite foreground cache index: {args.output}")
    study = json.loads(args.study.read_text())
    patch_size = tuple(int(value) for value in study["study"]["study_patch_size"])
    items = manifest_items(study, args.data_root.resolve() / "raw", args.arm, "train")
    _, source = load_cache_records(args.source_cache, items, patch_size)
    chunk_size = int(source["chunk_size"])
    by_case = {str(item["case_id"]): item for item in items}
    output = copy.deepcopy(source)
    output["sampling_index"] = {
        "strategy": "positive_chunks_v1",
        "source_cache_manifest": str(args.source_cache.resolve()),
        "source_cache_manifest_sha256": file_sha256(args.source_cache),
    }
    indexed_cases = []
    for index, entry in enumerate(output["cases"], start=1):
        case_id = str(entry["case_id"])
        item = by_case[case_id]
        label = np.asanyarray(nib.load(item["label"]).dataobj)
        positive = np.isin(label, np.asarray(item["positive_values"]))
        if not positive.any():
            raise ValueError(f"Training case has no foreground voxels: {case_id}")
        positive_chunks = np.unique(np.argwhere(positive) // chunk_size, axis=0)
        indexed_cases.append({
            **entry,
            "positive_chunks": positive_chunks.astype(int).tolist(),
            "positive_voxels": int(positive.sum()),
        })
        print(f"indexed {index}/{len(output['cases'])} {case_id}", flush=True)
    output["cases"] = indexed_cases
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")
    records, _ = load_cache_records(args.output, items, patch_size)
    if any(not record.get("positive_chunks") for record in records):
        raise ValueError("Foreground cache index failed its completeness gate")
    print(json.dumps({
        "status": "complete",
        "cases": len(records),
        "manifest": str(args.output.resolve()),
        "manifest_sha256": file_sha256(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
