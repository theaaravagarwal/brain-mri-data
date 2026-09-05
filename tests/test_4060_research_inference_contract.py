from __future__ import annotations

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import nibabel as nib
import numpy as np

from brain_mri_data.language_contracts import StudyInputQcV1
from scripts.run_4060_research_inference import (
    EXPECTED_CHECKPOINT_SHA256,
    MODALITY_SUFFIXES,
    evaluation_metrics,
    input_paths,
    validate_study,
    save_viewing_data,
)


class ResearchInferenceInputTests(unittest.TestCase):
    @unittest.skipUnless(find_spec("scipy"), "Viewing copies require the CUDA/scipy dependency set")
    def test_viewing_copies_preserve_geometry_and_remove_text(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            affine = np.array([[0, -2, 0, 14], [3, 0, 0, -9], [0, 0, 4, 7], [0, 0, 0, 1]], dtype=float)
            data = np.arange(120, dtype=np.float32).reshape(4, 5, 6)
            paths = []
            for index in range(4):
                image = nib.Nifti1Image(data, affine)
                image.header["descrip"] = b"private source text"
                path = root / f"input{index}.nii.gz"
                nib.save(image, path)
                paths.append(path)
            output = root / "output"
            output.mkdir()
            mask = np.zeros(data.shape, dtype=np.uint8)
            mask[1, 2, 3] = 1
            save_viewing_data(paths, mask, mask, output)
            viewed = nib.load(output / "flair.nii.gz")
            np.testing.assert_array_equal(viewed.get_fdata(), data)
            np.testing.assert_allclose(viewed.affine, affine)
            self.assertEqual(viewed.header["descrip"].item(), b"")
            manifest = json.loads((output / "viewing.json").read_text())
            np.testing.assert_allclose(manifest["outlineCenterMm"], nib.affines.apply_affine(affine, [1, 2, 3]))

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

    def test_valid_reference_is_checked_without_exposing_its_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_study(root)
            values = np.zeros((8, 9, 10), dtype=np.uint8)
            values[2:5, 2:5, 2:5] = 4
            nib.save(nib.Nifti1Image(values, np.eye(4)), root / "research_reference.nii.gz")
            result = validate_study(root)
            StudyInputQcV1.model_validate(result)
            self.assertEqual(result["reference_mask"]["labels"], [0, 4])
            self.assertEqual(result["reference_mask"]["nonzero_voxels"], 27)
            self.assertNotIn("filename", str(result).lower())

    def test_invalid_reference_geometry_labels_and_empty_mask_are_rejected(self) -> None:
        cases = [
            (np.ones((8, 9, 11), dtype=np.uint8), np.eye(4), "geometry"),
            (np.full((8, 9, 10), 7, dtype=np.uint8), np.eye(4), "unsupported labels"),
            (np.zeros((8, 9, 10), dtype=np.uint8), np.eye(4), "non-empty"),
        ]
        for values, affine, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.write_study(root)
                nib.save(nib.Nifti1Image(values, affine), root / "research_reference.nii")
                with self.assertRaisesRegex(ValueError, message):
                    validate_study(root)

    @unittest.skipUnless(find_spec("scipy"), "SciPy is installed with the CUDA inference extra")
    def test_evaluation_metrics_cover_match_offset_and_empty_prediction(self) -> None:
        truth = np.zeros((5, 5, 5), dtype=bool)
        truth[2, 2, 2] = True
        identical = evaluation_metrics(truth, truth, (2.0, 1.0, 1.0))
        self.assertEqual(identical["whole_lesion_dice"], 1.0)
        self.assertEqual(identical["whole_lesion_iou"], 1.0)
        self.assertEqual(identical["hd95_mm"], 0.0)
        shifted = np.zeros_like(truth)
        shifted[3, 2, 2] = True
        offset = evaluation_metrics(shifted, truth, (2.0, 1.0, 1.0))
        self.assertEqual(offset["whole_lesion_dice"], 0.0)
        self.assertEqual(offset["hd95_mm"], 2.0)
        empty = evaluation_metrics(np.zeros_like(truth), truth, (1.0, 1.0, 1.0))
        self.assertEqual(empty["recall"], 0.0)
        self.assertIsNone(empty["hd95_mm"])

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
            with self.assertRaisesRegex(ValueError, "four scan volumes"):
                input_paths(root)

    def test_checkpoint_digest_is_frozen(self) -> None:
        self.assertEqual(
            EXPECTED_CHECKPOINT_SHA256,
            "121422a861bbe7affaa5e161058e69eea737b2390651c3c03ea20256969e99e5",
        )


if __name__ == "__main__":
    unittest.main()
