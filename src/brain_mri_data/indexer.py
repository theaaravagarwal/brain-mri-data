from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Check the most-specific aliases first: T1c must never be classified as T1.
MODALITY_PATTERNS = {
    "seg": re.compile(r"(?:^|[_\-.])(seg(?:mentation)?|mask|label|tumou?r[_\-.]?seg)(?:[_\-.]|$)", re.I),
    "t1ce": re.compile(r"(?:^|[_\-.])(t1ce|t1c|t1gd|t1post|t1postcontrast)(?:[_\-.]|$)", re.I),
    "flair": re.compile(r"(?:^|[_\-.])(flair|t2flair|t2f)(?:[_\-.]|$)", re.I),
    "t1": re.compile(r"(?:^|[_\-.])(t1n|t1w|t1pre|t1)(?:[_\-.]|$)", re.I),
    "t2": re.compile(r"(?:^|[_\-.])(t2w|t2)(?:[_\-.]|$)", re.I),
}
NIFTI_SUFFIXES = (".nii", ".nii.gz")
CONTAINER_NAMES = {"raw", "data", "images", "image", "labels", "label", "masks", "mask", "training", "train", "validation", "val", "test"}


def _strip_nifti_suffix(name: str) -> str:
    return name.removesuffix(".nii.gz").removesuffix(".nii")


def _kind(path: Path) -> str | None:
    name = _strip_nifti_suffix(path.name)
    for kind, pattern in MODALITY_PATTERNS.items():
        if pattern.search(name):
            return kind
    return None


def _case_id(path: Path, source_root: Path) -> str:
    """Find a stable case ID across common BraTS/TCIA nested layouts."""
    relative = path.relative_to(source_root)
    parent = relative.parent
    if parent != Path(".") and parent.name.lower() not in CONTAINER_NAMES:
        return parent.name
    stem = _strip_nifti_suffix(path.name)
    stem = re.sub(
        r"(?:[_\-.])(seg(?:mentation)?|mask|label|tumou?r[_\-.]?seg|t1ce|t1c|t1gd|t1post|t1postcontrast|flair|t2flair|t2f|t1n|t1w|t1pre|t1|t2w|t2)$",
        "",
        stem,
        flags=re.I,
    )
    return stem


def _fingerprint(path: Path) -> dict[str, str | int]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def discover_source(source_id: str, source: dict[str, Any], raw_root: Path) -> dict[str, Any]:
    source_root = raw_root / source_id
    if not source_root.exists():
        raise FileNotFoundError(f"Source is not present: {source_root}")
    files = sorted(path for suffix in NIFTI_SUFFIXES for path in source_root.rglob(f"*{suffix}"))
    # `.nii.gz` matches `*.nii` nowhere, but make uniqueness explicit for future suffix additions.
    files = sorted(set(files))
    by_kind = Counter(_kind(path) or "unrecognized" for path in files)
    cases: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for path in files:
        kind = _kind(path)
        if kind:
            cases[_case_id(path, source_root)][kind].append(str(path.relative_to(source_root)))
    required = set(source.get("modalities", [])) | ({"seg"} if source.get("has_segmentation") else set())
    status = Counter()
    for modalities in cases.values():
        present = set(modalities)
        if any(len(paths) > 1 for paths in modalities.values()):
            status["ambiguous"] += 1
        elif required - present:
            status["incomplete"] += 1
        else:
            status["complete"] += 1
    return {
        "source_id": source_id,
        "source_root": str(source_root.resolve()),
        "nifti_files": len(files),
        "files_by_kind": dict(sorted(by_kind.items())),
        "candidate_cases": len(cases),
        "case_status": dict(sorted(status.items())),
        "required": sorted(required),
        "unrecognized_examples": [str(path.relative_to(source_root)) for path in files if _kind(path) is None][:20],
    }


def index_source(source_id: str, source: dict[str, Any], raw_root: Path, manifest_root: Path) -> tuple[Path, int, int]:
    source_root = raw_root / source_id
    summary = discover_source(source_id, source, raw_root)
    grouped: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for suffix in NIFTI_SUFFIXES:
        for path in source_root.rglob(f"*{suffix}"):
            kind = _kind(path)
            if kind:
                grouped[_case_id(path, source_root)][kind].append(path.resolve())

    required = set(source["modalities"]) | ({"seg"} if source.get("has_segmentation") else set())
    accepted, rejected = [], []
    for case_id, candidates in sorted(grouped.items()):
        present = set(candidates)
        collisions = {kind: [str(path) for path in paths] for kind, paths in candidates.items() if len(paths) != 1}
        missing = sorted(required - present)
        if collisions or missing:
            reason = "ambiguous_modalities" if collisions else "missing:" + ",".join(missing)
            rejected.append({"case_id": case_id, "reason": reason, "missing": missing, "collisions": collisions})
            continue
        files = {kind: paths[0] for kind, paths in candidates.items()}
        record = {
            "case_id": case_id,
            "patient_id": case_id,
            "source_id": source_id,
            "protocol": source.get("protocol", "unassigned"),
            "modalities": {key: str(files[key]) for key in source["modalities"]},
            "provenance": {
                "provider": source["provider"], "locator": source["locator"],
                "source_root": str(source_root.resolve()),
                "files": {key: _fingerprint(files[key]) for key in sorted(required)},
            },
        }
        if "seg" in files:
            record["segmentation"] = str(files["seg"])
        accepted.append(record)

    manifest_root.mkdir(parents=True, exist_ok=True)
    cases_path = manifest_root / f"{source_id}.cases.jsonl"
    excluded_path = manifest_root / f"{source_id}.excluded.jsonl"
    report_path = manifest_root / f"{source_id}.discovery.json"
    for path, records in ((cases_path, accepted), (excluded_path, rejected)):
        with path.open("w") as file:
            for record in records:
                file.write(json.dumps(record, sort_keys=True) + "\n")
    summary.update({"accepted_cases": len(accepted), "rejected_cases": len(rejected), "manifest": str(cases_path)})
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return cases_path, len(accepted), len(rejected)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]
