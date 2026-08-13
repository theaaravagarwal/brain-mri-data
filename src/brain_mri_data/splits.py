from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .indexer import read_jsonl


def make_split(cases_path: Path, destination: Path, seed: int) -> dict[str, int]:
    records = read_jsonl(cases_path)
    ordered = sorted(records, key=lambda item: hashlib.sha256(f"{seed}:{item['patient_id']}".encode()).hexdigest())
    counts = {"train": 0, "val": 0, "test": 0}
    with destination.open("w") as file:
        for index, record in enumerate(ordered):
            fraction = (index + 1) / max(len(ordered), 1)
            split = "train" if fraction <= .70 else "val" if fraction <= .85 else "test"
            counts[split] += 1
            file.write(json.dumps({"case_id": record["case_id"], "patient_id": record["patient_id"], "split": split}, sort_keys=True) + "\n")
    return counts
