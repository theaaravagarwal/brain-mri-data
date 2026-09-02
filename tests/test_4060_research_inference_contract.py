from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from brain_mri_data.language_contracts import StudyInputQcV1
from scripts.run_4060_research_inference import (
    EXPECTED_CHECKPOINT_SHA256,
    MODALITY_SUFFIXES,
    input_paths,
    validate_study,
)


class ResearchInferenceInputTests(unittest.TestCase):
    def write_study(
        self,
        root: Path,
        *,
        shape: tuple[int, int, int] = (8, 9, 10),
        mismatch_suffix: str | None = None,
        nonfinite_suffix: str | None = None,
    ) -> None:
        for suffix in MODALITY_SUFFIXES:
            current_shape = (8, 9, 11) if suffix == mismatch_suffix else shape
            values = np.zeros(current_shape, dtype=np.float32)
            values[2:5, 2:5, 2:5] = 1
            if suffix == nonfinite_suffix:
                values[0, 0, 0] = np.nan
            nib.save(nib.Nifti1Image(values, np.eye(4)), root / f"private-name_{suffix}.nii.gz")

    def test_valid_study_emits_geometry_and_digest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_study(root)
            result = validate_study(root)
            StudyInputQcV1.model_validate(result)
            self.assertEqual(result["schema_version"], "research-study-validation/v1")
            self.assertEqual(result["modalities"], ["t1", "t1ce", "t2", "flair"])
            self.assertEqual(result["shape"], [8, 9, 10])
            self.assertEqual(set(result["modality_sha256"]), {"t1", "t1ce", "t2", "flair"})
            self.assertNotIn("private-name", str(result))

    def test_geometry_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_study(root, mismatch_suffix="0003")
            with self.assertRaisesRegex(ValueError, "Geometry mismatch"):
                validate_study(root)

    def test_nonfinite_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_study(root, nonfinite_suffix="0002")
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                validate_study(root)

    def test_extra_nifti_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_study(root)
            nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)), root / "extra.nii.gz")
            with self.assertRaisesRegex(ValueError, "exactly the four"):
                input_paths(root)

    def test_checkpoint_digest_is_frozen(self) -> None:
        self.assertEqual(
            EXPECTED_CHECKPOINT_SHA256,
            "121422a861bbe7affaa5e161058e69eea737b2390651c3c03ea20256969e99e5",
        )


if __name__ == "__main__":
    unittest.main()
