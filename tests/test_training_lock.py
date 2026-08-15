from __future__ import annotations

import unittest

from training.train_glioma import validate_profile_against_study


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
