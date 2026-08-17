#!/usr/bin/env python3
"""Build a restart-safe chunk-major cache for a locked training split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, NormalizeIntensityd, SpatialPadd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.chunk_cache import (  # noqa: E402
    CACHE_CHANNELS,
    CACHE_PREPROCESSING_ID,
    CACHE_PREPROCESSING_ID_COMPACT,
    CACHE_SCHEMA_VERSION,
    CACHE_SCHEMA_VERSION_COMPACT,
    IMAGE_CHANNELS,
    load_cache_records,
    to_compact_chunk_major,
    to_chunk_major,
    training_split_sha256,
)
from training.train_glioma import WholeLesiond, file_sha256, manifest_items  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("brats", "pooled", "pamc"), default="brats")
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument(
        "--storage-format",
        choices=("dense-f32-v1", "compact-f16-u8-v2"),
        default="dense-f32-v1",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def cache_key(case_id: str) -> str:
    return hashlib.sha256(case_id.encode()).hexdigest()[:20]


def valid_case(
    paths: dict[str, Path],
    metadata_path: Path,
    expected: dict[str, Any],
    chunk_size: int,
) -> dict[str, Any] | None:
    if not all(path.is_file() for path in paths.values()) or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
        if any(metadata.get(key) != value for key, value in expected.items()):
            return None
        shape = tuple(int(value) for value in metadata["shape"])
        expected_grid = tuple((dimension + chunk_size - 1) // chunk_size for dimension in shape)
        expected_arrays = (
            {"image_cache": (np.dtype(np.float16), IMAGE_CHANNELS), "mask_cache": (np.dtype(np.uint8), 1)}
            if expected["schema_version"] == CACHE_SCHEMA_VERSION_COMPACT
            else {"cache": (np.dtype(np.float32), CACHE_CHANNELS)}
        )
        if set(paths) != set(expected_arrays):
            return None
        for field, path in paths.items():
            dtype, channels = expected_arrays[field]
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 7 or array.dtype != dtype or array.shape[3:] != (
                channels, chunk_size, chunk_size, chunk_size,
            ):
                return None
            if array.shape[:3] != expected_grid:
                return None
            if expected["schema_version"] == CACHE_SCHEMA_VERSION_COMPACT:
                expected_sha256 = str(metadata.get("sha256", {}).get(field, ""))
                if len(expected_sha256) != 64 or file_sha256(path) != expected_sha256:
                    return None
        return metadata
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def main() -> None:
    args = arguments()
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    study = json.loads(args.study.read_text())
    patch_size = tuple(int(value) for value in study["study"]["study_patch_size"])
    raw_root = args.data_root.resolve() / "raw"
    items = manifest_items(study, raw_root, args.arm, "train")
    study_sha256 = file_sha256(args.study)
    compact = args.storage_format == "compact-f16-u8-v2"
    schema_version = CACHE_SCHEMA_VERSION_COMPACT if compact else CACHE_SCHEMA_VERSION
    preprocessing_id = CACHE_PREPROCESSING_ID_COMPACT if compact else CACHE_PREPROCESSING_ID
    args.output.mkdir(parents=True, exist_ok=True)
    final_manifest = args.output / "cache.json"
    if final_manifest.is_file():
        existing = json.loads(final_manifest.read_text())
        if existing.get("schema_version") != schema_version or existing.get("preprocessing_id") != preprocessing_id:
            raise ValueError("Existing cache uses a different storage format")
        existing.setdefault("training_split_sha256", training_split_sha256(items))
        compatible = set(existing.get("compatible_study_sha256", []))
        if existing.get("study_sha256"):
            compatible.add(str(existing["study_sha256"]))
        compatible.add(study_sha256)
        existing["compatible_study_sha256"] = sorted(compatible)
        atomic_json(final_manifest, existing)
        records, payload = load_cache_records(final_manifest, items, patch_size)
        print(json.dumps({"status": "already_complete", "cases": len(records), "manifest": str(final_manifest.resolve())}))
        return
    transforms = Compose([
        LoadImaged(keys=("image", "label")),
        EnsureChannelFirstd(keys=("image", "label")),
        WholeLesiond(keys="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        SpatialPadd(keys=("image", "label"), spatial_size=patch_size),
        EnsureTyped(keys=("image", "label"), dtype=torch.float32),
    ])
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, item in enumerate(items, start=1):
        case_id = str(item["case_id"])
        key = cache_key(case_id)
        paths = (
            {
                "image_cache": args.output / f"{key}.image.f16.npy",
                "mask_cache": args.output / f"{key}.mask.u8.npy",
            }
            if compact
            else {"cache": args.output / f"{key}.npy"}
        )
        metadata_path = args.output / f"{key}.json"
        expected = {
            "schema_version": schema_version,
            "preprocessing_id": preprocessing_id,
            "study_sha256": study_sha256,
            "case_id": case_id,
            **{field: path.name for field, path in paths.items()},
            "chunk_size": args.chunk_size,
        }
        metadata = valid_case(paths, metadata_path, expected, args.chunk_size)
        if metadata is None:
            data = transforms(item)
            image = np.asarray(data["image"], dtype=np.float32)
            label = np.asarray(data["label"], dtype=np.float32)
            if image.shape[0] != 4 or label.shape[0] != 1 or image.shape[1:] != label.shape[1:]:
                raise ValueError(f"Unexpected preprocessed shapes for {case_id}: {image.shape}, {label.shape}")
            if not np.logical_or(label == 0, label == 1).all():
                raise ValueError(f"Cached label is not binary: {case_id}")
            arrays = (
                dict(zip(paths, to_compact_chunk_major(image, label, args.chunk_size), strict=True))
                if compact
                else {"cache": to_chunk_major(np.concatenate((image, label), axis=0), args.chunk_size)}
            )
            for field, array in arrays.items():
                path = paths[field]
                temporary = path.with_suffix(path.suffix + ".tmp")
                with temporary.open("wb") as stream:
                    np.save(stream, array, allow_pickle=False)
                os.replace(temporary, path)
            metadata = {
                **expected,
                "shape": list(image.shape[1:]),
                "bytes": {field: path.stat().st_size for field, path in paths.items()},
                **({"sha256": {field: file_sha256(path) for field, path in paths.items()}} if compact else {}),
            }
            atomic_json(metadata_path, metadata)
        records.append({
            "case_id": case_id,
            **{field: path.name for field, path in paths.items()},
            **(
                {f"{field}_sha256": metadata["sha256"][field] for field in paths}
                if compact
                else {}
            ),
            "shape": metadata["shape"],
        })
        atomic_json(args.output / "progress.json", {
            "schema_version": 1,
            "phase": "building_training_cache",
            "cases_complete": index,
            "cases_total": len(items),
            "elapsed_seconds": round(time.monotonic() - started, 1),
        })
        print(f"cached {index}/{len(items)} {case_id}", flush=True)
    payload = {
        "schema_version": schema_version,
        "preprocessing_id": preprocessing_id,
        "storage_format": args.storage_format,
        "study": str(args.study.resolve()),
        "study_sha256": study_sha256,
        "compatible_study_sha256": [study_sha256],
        "training_split_sha256": training_split_sha256(items),
        "arm": args.arm,
        "chunk_size": args.chunk_size,
        "patch_size": list(patch_size),
        "cases": records,
    }
    atomic_json(final_manifest, payload)
    load_cache_records(final_manifest, items, patch_size)
    atomic_json(args.output / "progress.json", {
        "schema_version": 1,
        "phase": "complete",
        "cases_complete": len(items),
        "cases_total": len(items),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "manifest_sha256": file_sha256(final_manifest),
    })
    print(json.dumps({"status": "complete", "cases": len(items), "manifest": str(final_manifest.resolve())}))


if __name__ == "__main__":
    main()
