from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from training.train_glioma import validate_cnn_accelerator, validate_profile_against_study, write_progress


class TrainingLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.study = {
            "study": {
                "study_patch_size": [80, 80, 80],
                "effective_batch_size": 4,
                "training": {"mixed_precision": "fp16"},
            }
        }
        self.profile = {
            "patch_size": [80, 80, 80],
            "effective_batch_size": 4,
            "mixed_precision": "fp16",
        }

    def test_matching_runtime_profile_is_accepted(self) -> None:
        validate_profile_against_study(self.profile, self.study)

    def test_changed_patch_or_effective_batch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "patch_size"):
            validate_profile_against_study({**self.profile, "patch_size": [96, 96, 96]}, self.study)
        with self.assertRaisesRegex(ValueError, "effective_batch_size"):
            validate_profile_against_study({**self.profile, "effective_batch_size": 2}, self.study)

    def test_locked_microbatch_size_is_enforced(self) -> None:
        study = {
            "study": {
                **self.study["study"],
                "training": {"mixed_precision": "fp16", "microbatch_size": 4},
            }
        }
        validate_profile_against_study({**self.profile, "batch_size": 4}, study)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            validate_profile_against_study({**self.profile, "batch_size": 2}, study)

    def test_cnn_study_rejects_rocm_profiles(self) -> None:
        validate_cnn_accelerator({"accelerator": "cuda"}, None)
        with self.assertRaisesRegex(ValueError, "restricted to the CUDA"):
            validate_cnn_accelerator({"accelerator": "amd"}, "7.2")
        with self.assertRaisesRegex(ValueError, "CUDA PyTorch"):
            validate_cnn_accelerator({"accelerator": "cuda"}, "7.2")

    def test_live_progress_is_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "progress.json"
            write_progress(path, phase="training", epoch=1, batches_complete=3)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["phase"], "training")
            self.assertEqual(payload["batches_complete"], 3)
            self.assertEqual(payload["schema_version"], 1)
