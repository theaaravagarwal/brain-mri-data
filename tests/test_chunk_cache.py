from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.chunk_cache import (
    CACHE_PREPROCESSING_ID,
    CACHE_PREPROCESSING_ID_COMPACT,
    LoadChunkPatchd,
    load_cache_records,
    read_compact_chunk_patch,
    read_chunk_patch,
    to_compact_chunk_major,
    to_chunk_major,
    training_split_sha256,
)


class ChunkCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        values = np.arange(5 * 7 * 8 * 9, dtype=np.float32).reshape(5, 7, 8, 9)
        values[4] = values[4] % 2
        self.values = values
        self.cache = self.root / "case.npy"
        np.save(self.cache, to_chunk_major(values, 4), allow_pickle=False)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_unaligned_patch_is_bit_exact(self) -> None:
        actual = read_chunk_patch(self.cache, (1, 2, 3), (5, 4, 5), 4)
        np.testing.assert_array_equal(actual, self.values[:, 1:6, 2:6, 3:8])

    def test_random_transform_is_reproducible_and_binary(self) -> None:
        record = {"cache": str(self.cache), "shape": (7, 8, 9), "case_id": "case"}
        first = LoadChunkPatchd((5, 4, 5), 4)
        second = LoadChunkPatchd((5, 4, 5), 4)
        first.set_random_state(seed=17)
        second.set_random_state(seed=17)
        first_result = first(record)
        second_result = second(record)
        np.testing.assert_array_equal(first_result["image"], second_result["image"])
        np.testing.assert_array_equal(first_result["label"], second_result["label"])
        self.assertEqual(first_result["image"].shape, (4, 5, 4, 5))
        self.assertTrue(np.logical_or(first_result["label"] == 0, first_result["label"] == 1).all())

    def test_compact_patch_preserves_mask_and_bounds_image_quantization(self) -> None:
        image = self.values[:4]
        label = self.values[4:]
        image_chunks, mask_chunks = to_compact_chunk_major(image, label, 4)
        image_path = self.root / "case.image.f16.npy"
        mask_path = self.root / "case.mask.u8.npy"
        np.save(image_path, image_chunks, allow_pickle=False)
        np.save(mask_path, mask_chunks, allow_pickle=False)
        actual_image, actual_label = read_compact_chunk_patch(
            image_path, mask_path, (1, 2, 3), (5, 4, 5), 4,
        )
        expected_image = image[:, 1:6, 2:6, 3:8]
        np.testing.assert_array_equal(actual_label, label[:, 1:6, 2:6, 3:8])
        self.assertLessEqual(float(np.abs(actual_image - expected_image).max()), 0.01)
        self.assertLessEqual(float(np.abs(actual_image - expected_image).mean()), 0.001)

    def test_compact_manifest_is_validated_and_loaded(self) -> None:
        image_chunks, mask_chunks = to_compact_chunk_major(self.values[:4], self.values[4:], 4)
        image_path = self.root / "case.image.f16.npy"
        mask_path = self.root / "case.mask.u8.npy"
        np.save(image_path, image_chunks, allow_pickle=False)
        np.save(mask_path, mask_chunks, allow_pickle=False)
        image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
        mask_sha256 = hashlib.sha256(mask_path.read_bytes()).hexdigest()
        source_items = [{"case_id": "case"}]
        manifest = self.root / "compact.json"
        manifest.write_text(json.dumps({
            "schema_version": 2,
            "preprocessing_id": CACHE_PREPROCESSING_ID_COMPACT,
            "storage_format": "compact-f16-u8-v2",
            "chunk_size": 4,
            "training_split_sha256": training_split_sha256(source_items),
            "cases": [{
                "case_id": "case",
                "image_cache": image_path.name,
                "mask_cache": mask_path.name,
                "image_cache_sha256": image_sha256,
                "mask_cache_sha256": mask_sha256,
                "shape": [7, 8, 9],
            }],
        }))
        records, payload = load_cache_records(manifest, source_items, (5, 4, 5))
        self.assertEqual(payload["schema_version"], 2)
        result = LoadChunkPatchd((5, 4, 5), 4)(records[0])
        self.assertEqual(result["image"].dtype, np.float32)
        self.assertEqual(result["label"].dtype, np.float32)
        self.assertTrue(np.logical_or(result["label"] == 0, result["label"] == 1).all())
        mask_path.write_bytes(mask_path.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "hash"):
            load_cache_records(manifest, source_items, (5, 4, 5))

    def test_compact_cache_rejects_nonbinary_masks(self) -> None:
        invalid = self.values[4:].copy()
        invalid[0, 0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            to_compact_chunk_major(self.values[:4], invalid, 4)

    def test_manifest_must_match_locked_cases(self) -> None:
        manifest = self.root / "cache.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "preprocessing_id": CACHE_PREPROCESSING_ID,
            "chunk_size": 4,
            "training_split_sha256": training_split_sha256([{"case_id": "case"}]),
            "cases": [{"case_id": "case", "cache": self.cache.name, "shape": [7, 8, 9]}],
        }))
        records, _ = load_cache_records(manifest, [{"case_id": "case"}], (5, 4, 5))
        self.assertEqual(records[0]["case_id"], "case")
        with self.assertRaisesRegex(ValueError, "fingerprint|do not match"):
            load_cache_records(manifest, [{"case_id": "different"}], (5, 4, 5))

    def test_foreground_sampler_always_contains_an_indexed_positive_chunk(self) -> None:
        values = np.zeros((5, 80, 80, 80), dtype=np.float32)
        values[4, 65, 65, 65] = 1
        cache = self.root / "foreground.npy"
        np.save(cache, to_chunk_major(values, 20), allow_pickle=False)
        record = {
            "cache": str(cache),
            "shape": (80, 80, 80),
            "case_id": "foreground",
            "positive_chunks": [(3, 3, 3)],
        }
        transform = LoadChunkPatchd((40, 40, 40), 20, foreground_probability=1.0)
        transform.set_random_state(seed=23)
        for _ in range(10):
            result = transform(record)
            self.assertGreater(result["label"].sum(), 0)
            self.assertNotIn("positive_chunks", result)

    def test_foreground_sampler_requires_an_index(self) -> None:
        transform = LoadChunkPatchd((5, 4, 5), 4, foreground_probability=0.5)
        with self.assertRaisesRegex(ValueError, "positive-chunk index"):
            transform({"cache": str(self.cache), "shape": (7, 8, 9), "case_id": "case"})


if __name__ == "__main__":
    unittest.main()
