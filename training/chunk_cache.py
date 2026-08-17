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
CACHE_SCHEMA_VERSION_COMPACT = 2
CACHE_PREPROCESSING_ID = "whole-lesion-normalize-nonzero-channel-wise-pad-f32-chunk-major-v1"
CACHE_PREPROCESSING_ID_COMPACT = "whole-lesion-normalize-nonzero-channel-wise-pad-f16-u8-chunk-major-v2"
CACHE_CHANNELS = 5  # Four normalized modalities plus one binary whole-lesion mask.
IMAGE_CHANNELS = 4


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _to_chunk_major_typed(values: np.ndarray, chunk_size: int) -> np.ndarray:
    """Pad a typed [C, D, H, W] array and expose a chunk-major view."""
    if values.ndim != 4 or values.shape[0] < 1:
        raise ValueError("Typed chunk cache requires a non-empty [C, D, H, W] array")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    spatial = values.shape[1:]
    padded = tuple(((dimension + chunk_size - 1) // chunk_size) * chunk_size for dimension in spatial)
    storage = np.zeros((values.shape[0], *padded), dtype=values.dtype)
    storage[:, :spatial[0], :spatial[1], :spatial[2]] = values
    grid = tuple(dimension // chunk_size for dimension in padded)
    return storage.reshape(
        values.shape[0],
        grid[0], chunk_size,
        grid[1], chunk_size,
        grid[2], chunk_size,
    ).transpose(1, 3, 5, 0, 2, 4, 6)


def to_compact_chunk_major(
    image: np.ndarray,
    label: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode normalized images as float16 and a binary mask as uint8."""
    if image.ndim != 4 or image.shape[0] != IMAGE_CHANNELS or image.dtype != np.float32:
        raise ValueError("Compact cache requires float32 [4, D, H, W] images")
    if label.ndim != 4 or label.shape[0] != 1 or label.shape[1:] != image.shape[1:]:
        raise ValueError("Compact cache requires a matching [1, D, H, W] mask")
    if not np.logical_or(label == 0, label == 1).all():
        raise ValueError("Compact cache mask must be binary")
    return (
        _to_chunk_major_typed(image.astype(np.float16), chunk_size),
        _to_chunk_major_typed(label.astype(np.uint8), chunk_size),
    )


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


def _read_typed_chunk_patch(
    path: Path,
    starts: Sequence[int],
    roi_size: Sequence[int],
    chunk_size: int,
    channels: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    starts = tuple(int(value) for value in starts)
    roi_size = tuple(int(value) for value in roi_size)
    if len(starts) != 3 or len(roi_size) != 3:
        raise ValueError("starts and roi_size must have three spatial dimensions")
    cached = np.load(path, mmap_mode="r", allow_pickle=False)
    expected_tail = (channels, chunk_size, chunk_size, chunk_size)
    if cached.ndim != 7 or cached.dtype != dtype or cached.shape[3:] != expected_tail:
        raise ValueError(f"Invalid compact chunk cache array: {path}")
    low = tuple(start // chunk_size for start in starts)
    high = tuple((start + roi - 1) // chunk_size + 1 for start, roi in zip(starts, roi_size, strict=True))
    if any(start < 0 for start in starts) or any(high[index] > cached.shape[index] for index in range(3)):
        raise ValueError("Requested patch is outside the cached volume")
    chunks = np.array(cached[low[0]:high[0], low[1]:high[1], low[2]:high[2]], copy=True)
    volume = chunks.transpose(3, 0, 4, 1, 5, 2, 6).reshape(
        channels,
        (high[0] - low[0]) * chunk_size,
        (high[1] - low[1]) * chunk_size,
        (high[2] - low[2]) * chunk_size,
    )
    offsets = tuple(starts[index] - low[index] * chunk_size for index in range(3))
    return np.array(volume[
        :,
        offsets[0]:offsets[0] + roi_size[0],
        offsets[1]:offsets[1] + roi_size[1],
        offsets[2]:offsets[2] + roi_size[2],
    ], copy=True)


def read_compact_chunk_patch(
    image_path: Path,
    mask_path: Path,
    starts: Sequence[int],
    roi_size: Sequence[int],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a compact patch and restore tensors to float32 for training."""
    image = _read_typed_chunk_patch(
        image_path, starts, roi_size, chunk_size, IMAGE_CHANNELS, np.dtype(np.float16),
    ).astype(np.float32)
    label = _read_typed_chunk_patch(
        mask_path, starts, roi_size, chunk_size, 1, np.dtype(np.uint8),
    ).astype(np.float32)
    return image, label


def load_cache_records(
    manifest_path: Path,
    source_items: Sequence[dict[str, Any]],
    patch_size: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate a completed cache manifest and align it to locked train items."""
    payload = json.loads(manifest_path.read_text())
    schema_version = payload.get("schema_version")
    if schema_version not in (CACHE_SCHEMA_VERSION, CACHE_SCHEMA_VERSION_COMPACT):
        raise ValueError("Unsupported training cache schema")
    expected_preprocessing = (
        CACHE_PREPROCESSING_ID_COMPACT
        if schema_version == CACHE_SCHEMA_VERSION_COMPACT
        else CACHE_PREPROCESSING_ID
    )
    if payload.get("preprocessing_id") != expected_preprocessing:
        raise ValueError("Training cache preprocessing does not match the trainer")
    if schema_version == CACHE_SCHEMA_VERSION_COMPACT and payload.get("storage_format") != "compact-f16-u8-v2":
        raise ValueError("Compact training cache has an invalid storage format")
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
        shape = tuple(int(value) for value in entry.get("shape", ()))
        if len(shape) != 3 or any(dimension < roi[index] for index, dimension in enumerate(shape)):
            raise ValueError(f"Training cache shape cannot supply the locked patch: {case_id}")
        expected_grid = tuple((dimension + chunk_size - 1) // chunk_size for dimension in shape)
        path_fields = (
            (("image_cache", np.dtype(np.float16), IMAGE_CHANNELS), ("mask_cache", np.dtype(np.uint8), 1))
            if schema_version == CACHE_SCHEMA_VERSION_COMPACT
            else (("cache", np.dtype(np.float32), CACHE_CHANNELS),)
        )
        resolved_paths: dict[str, str] = {}
        for field, dtype, channels in path_fields:
            relative = Path(str(entry.get(field, "")))
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError("Training cache path escapes its declared root") from error
            if not path.is_file():
                raise FileNotFoundError(f"Missing training cache case: {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 7 or array.dtype != dtype or array.shape[3:] != (
                channels, chunk_size, chunk_size, chunk_size,
            ):
                raise ValueError(f"Invalid training cache case: {path}")
            if array.shape[:3] != expected_grid:
                raise ValueError(f"Training cache grid does not match its declared shape: {path}")
            if schema_version == CACHE_SCHEMA_VERSION_COMPACT:
                expected_sha256 = str(entry.get(f"{field}_sha256", ""))
                if len(expected_sha256) != 64 or _file_sha256(path) != expected_sha256:
                    raise ValueError(f"Training cache payload hash does not match: {path}")
            resolved_paths[field] = str(path)
        positive_chunks = entry.get("positive_chunks")
        if positive_chunks is not None:
            if not isinstance(positive_chunks, list) or not positive_chunks:
                raise ValueError(f"Foreground cache index is empty: {case_id}")
            normalized_chunks = []
            for chunk in positive_chunks:
                if not isinstance(chunk, list) or len(chunk) != 3:
                    raise ValueError(f"Invalid foreground chunk index: {case_id}")
                normalized = tuple(int(value) for value in chunk)
                if any(value < 0 or value >= expected_grid[index] for index, value in enumerate(normalized)):
                    raise ValueError(f"Foreground chunk index is outside the cache grid: {case_id}")
                normalized_chunks.append(normalized)
            positive_chunks = normalized_chunks
        records.append({
            **resolved_paths,
            "shape": shape,
            "case_id": case_id,
            **({"positive_chunks": positive_chunks} if positive_chunks is not None else {}),
        })
    return records, payload


class LoadChunkPatchd(RandomizableTransform, MapTransform):
    """Sample a locked-size patch directly from chunk-major storage."""

    def __init__(
        self,
        roi_size: Sequence[int],
        chunk_size: int,
        foreground_probability: float = 0.0,
    ) -> None:
        RandomizableTransform.__init__(self, prob=1.0)
        MapTransform.__init__(
            self,
            keys=("cache", "image_cache", "mask_cache"),
            allow_missing_keys=True,
        )
        self.roi_size = tuple(int(value) for value in roi_size)
        self.chunk_size = int(chunk_size)
        self.foreground_probability = float(foreground_probability)
        if not 0.0 <= self.foreground_probability <= 1.0:
            raise ValueError("foreground_probability must be between zero and one")
        self.starts = (0, 0, 0)

    def randomize(
        self,
        shape: Sequence[int],
        positive_chunks: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().randomize(None)
        if self.foreground_probability and not positive_chunks:
            raise ValueError("Foreground sampling requires a non-empty positive-chunk index")
        if positive_chunks and self.R.random_sample() < self.foreground_probability:
            selected = positive_chunks[int(self.R.randint(0, len(positive_chunks)))]
            starts = []
            for dimension, roi, chunk_index in zip(shape, self.roi_size, selected, strict=True):
                chunk_start = int(chunk_index) * self.chunk_size
                chunk_end = min(chunk_start + self.chunk_size, int(dimension))
                low = max(0, chunk_end - roi)
                high = min(chunk_start, int(dimension) - roi)
                if low > high:
                    raise ValueError("Positive chunk cannot fit inside the locked patch")
                starts.append(int(self.R.randint(low, high + 1)))
            self.starts = tuple(starts)
            return
        self.starts = tuple(
            int(self.R.randint(0, int(dimension) - roi + 1))
            for dimension, roi in zip(shape, self.roi_size, strict=True)
        )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)
        shape = tuple(int(value) for value in result["shape"])
        if any(dimension < self.roi_size[index] for index, dimension in enumerate(shape)):
            raise ValueError("Cached volume cannot supply the locked patch size")
        positive_chunks = result.pop("positive_chunks", None)
        self.randomize(shape, positive_chunks)
        if "image_cache" in result or "mask_cache" in result:
            if "image_cache" not in result or "mask_cache" not in result:
                raise ValueError("Compact cache record requires image_cache and mask_cache")
            result["image"], result["label"] = read_compact_chunk_patch(
                Path(result["image_cache"]),
                Path(result["mask_cache"]),
                self.starts,
                self.roi_size,
                self.chunk_size,
            )
        else:
            combined = read_chunk_patch(
                Path(result["cache"]), self.starts, self.roi_size, self.chunk_size,
            )
            result["image"] = combined[:4]
            result["label"] = combined[4:]
        return result
