from __future__ import annotations

import unittest

import torch

from monai.data import MetaTensor

from training.evaluation import box_iou_per_case, deterministic_mask_one_modality, dice_per_case, hd95_mm_per_case


class EvaluationTests(unittest.TestCase):
    def test_corruption_is_stable_by_case_not_batch_order(self) -> None:
        image = torch.ones((2, 4, 2, 2, 2))
        first, modalities = deterministic_mask_one_modality(image, ["a", "b"], 20260812)
        reordered, reordered_modalities = deterministic_mask_one_modality(image.flip(0), ["b", "a"], 20260812)
        self.assertEqual(modalities, list(reversed(reordered_modalities)))
        self.assertTrue(torch.equal(first[0], reordered[1]))

    def test_dice_and_derived_box_iou_are_case_linkable(self) -> None:
        logits = torch.full((1, 1, 3, 3, 3), -20.0)
        labels = torch.zeros((1, 1, 3, 3, 3))
        logits[:, :, 1, 1, 1] = 20.0
        labels[:, :, 1, 1, 1] = 1.0
        self.assertEqual(dice_per_case(logits, labels), [1.0])
        self.assertEqual(box_iou_per_case(logits, labels), [1.0])

    def test_hd95_uses_nifti_spacing_and_does_not_hide_empty_prediction(self) -> None:
        logits = torch.full((1, 1, 3, 3, 3), -20.0)
        labels = MetaTensor(torch.zeros((1, 1, 3, 3, 3)), meta={"pixdim": torch.tensor([[1.0, 1.5, 2.0, 2.5]])})
        logits[:, :, 1, 1, 1] = 20.0
        labels[:, :, 1, 1, 1] = 1.0
        self.assertEqual(hd95_mm_per_case(logits, labels), [0.0])
        self.assertEqual(hd95_mm_per_case(torch.full_like(logits, -20.0), labels), [None])
