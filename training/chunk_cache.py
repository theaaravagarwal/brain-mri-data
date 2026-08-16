"""Chunk-major, provenance-bound cache for random 3D training patches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from monai.transforms import MapTransform, RandomizableTransform


CACHE_SCHEMA_VERSION = 1
CACHE_PREPROCESSING_ID = "whole-lesion-normalize-nonzero-channel-wise-pad-f32-chunk-major-v1"
CACHE_CHANNELS = 5  # Four normalized modalities plus one binary whole-lesion mask.


def training_split_sha256(source_items: Sequence[dict[str, Any]]) -> str:
    """Fingerprint ordered cases, resolved inputs, labels, and source provenance."""
    encoded = json.dumps(list(source_items), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def to_chunk_major(values: np.ndarray, chunk_size: int) -> np.ndarray:
    """Pad a [C, D, H, W] float32 array and expose a chunk-major view."""
    if values.ndim != 4 or values.shape[0] != CACHE_CHANNELS or values.dtype != np.float32:
        raise ValueError("Chunk cache requires a float32 [5, D, H, W] array")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    spatial = values.shape[1:]
    padded = tuple(((dimension + chunk_size - 1) // chunk_size) * chunk_size for dimension in spatial)
    storage = np.zeros((CACHE_CHANNELS, *padded), dtype=np.float32)
    storage[:, :spatial[0], :spatial[1], :spatial[2]] = values
    grid = tuple(dimension // chunk_size for dimension in padded)
    return storage.reshape(
        CACHE_CHANNELS,
        grid[0], chunk_size,
        grid[1], chunk_size,
        grid[2], chunk_size,
    ).transpose(1, 3, 5, 0, 2, 4, 6)


def read_chunk_patch(
    path: Path,
    starts: Sequence[int],
    roi_size: Sequence[int],
    chunk_size: int,
) -> np.ndarray:
    """Read only the chunk cuboid intersecting a requested spatial patch."""
    starts = tuple(int(value) for value in starts)
    roi_size = tuple(int(value) for value in roi_size)
    if len(starts) != 3 or len(roi_size) != 3:
        raise ValueError("starts and roi_size must have three spatial dimensions")
    cached = np.load(path, mmap_mode="r", allow_pickle=False)
    expected_tail = (CACHE_CHANNELS, chunk_size, chunk_size, chunk_size)
    if cached.ndim != 7 or cached.dtype != np.float32 or cached.shape[3:] != expected_tail:
        raise ValueError(f"Invalid chunk cache array: {path}")
    low = tuple(start // chunk_size for start in starts)
    high = tuple(
        (start + roi - 1) // chunk_size + 1
        for start, roi in zip(starts, roi_size, strict=True)
    )
    if any(start < 0 for start in starts) or any(high[index] > cached.shape[index] for index in range(3)):
        raise ValueError("Requested patch is outside the cached volume")
    chunks = np.array(
        cached[low[0]:high[0], low[1]:high[1], low[2]:high[2]],
        copy=True,
    )
    volume = chunks.transpose(3, 0, 4, 1, 5, 2, 6).reshape(
        CACHE_CHANNELS,
        (high[0] - low[0]) * chunk_size,
        (high[1] - low[1]) * chunk_size,
        (high[2] - low[2]) * chunk_size,
    )
    offsets = tuple(starts[index] - low[index] * chunk_size for index in range(3))
    return np.array(
        volume[
            :,
            offsets[0]:offsets[0] + roi_size[0],
            offsets[1]:offsets[1] + roi_size[1],
            offsets[2]:offsets[2] + roi_size[2],
        ],
        copy=True,
    )


def load_cache_records(
    manifest_path: Path,
    source_items: Sequence[dict[str, Any]],
    patch_size: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate a completed cache manifest and align it to locked train items."""
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported training cache schema")
    if payload.get("preprocessing_id") != CACHE_PREPROCESSING_ID:
        raise ValueError("Training cache preprocessing does not match the trainer")
    chunk_size = int(payload.get("chunk_size", 0))
    entries = payload.get("cases")
    if chunk_size < 1 or not isinstance(entries, list):
        raise ValueError("Training cache manifest is incomplete")
    by_case = {entry.get("case_id"): entry for entry in entries if isinstance(entry, dict)}
    source_ids = [str(item["case_id"]) for item in source_items]
    if payload.get("training_split_sha256") != training_split_sha256(source_items):
        raise ValueError("Training cache fingerprint does not match the locked training split")
    if len(by_case) != len(entries) or set(by_case) != set(source_ids):
        raise ValueError("Training cache cases do not match the locked training split")
    root = manifest_path.parent.resolve()
    roi = tuple(int(value) for value in patch_size)
    records: list[dict[str, Any]] = []
    for case_id in source_ids:
        entry = by_case[case_id]
        relative = Path(str(entry.get("cache", "")))
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("Training cache path escapes its declared root") from error
        shape = tuple(int(value) for value in entry.get("shape", ()))
        if len(shape) != 3 or any(dimension < roi[index] for index, dimension in enumerate(shape)):
            raise ValueError(f"Training cache shape cannot supply the locked patch: {case_id}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing training cache case: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 7 or array.dtype != np.float32 or array.shape[3:] != (
            CACHE_CHANNELS, chunk_size, chunk_size, chunk_size,
        ):
            raise ValueError(f"Invalid training cache case: {path}")
        expected_grid = tuple((dimension + chunk_size - 1) // chunk_size for dimension in shape)
        if array.shape[:3] != expected_grid:
            raise ValueError(f"Training cache grid does not match its declared shape: {path}")
        records.append({"cache": str(path), "shape": shape, "case_id": case_id})
    return records, payload


class LoadChunkPatchd(RandomizableTransform, MapTransform):
    """Uniformly sample a locked-size patch directly from chunk-major storage."""

    def __init__(self, roi_size: Sequence[int], chunk_size: int) -> None:
        RandomizableTransform.__init__(self, prob=1.0)
        MapTransform.__init__(self, keys=("cache",))
        self.roi_size = tuple(int(value) for value in roi_size)
        self.chunk_size = int(chunk_size)
        self.starts = (0, 0, 0)

    def randomize(self, shape: Sequence[int]) -> None:
        super().randomize(None)
        self.starts = tuple(
            int(self.R.randint(0, int(dimension) - roi + 1))
            for dimension, roi in zip(shape, self.roi_size, strict=True)
        )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)
        shape = tuple(int(value) for value in result["shape"])
        if any(dimension < self.roi_size[index] for index, dimension in enumerate(shape)):
            raise ValueError("Cached volume cannot supply the locked patch size")
        self.randomize(shape)
        combined = read_chunk_patch(
            Path(result["cache"]), self.starts, self.roi_size, self.chunk_size,
        )
        result["image"] = combined[:4]
        result["label"] = combined[4:]
        return result
