from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.chunk_cache import (
    CACHE_PREPROCESSING_ID,
    LoadChunkPatchd,
    load_cache_records,
    read_chunk_patch,
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
