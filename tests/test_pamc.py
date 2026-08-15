from __future__ import annotations

import unittest

import torch

from training.pamc import mask_one_modality, modality_consistency_loss, reverse_gradient


class PamcTests(unittest.TestCase):
    def test_gradient_is_reversed(self) -> None:
        values = torch.tensor([2.0], requires_grad=True)
        reverse_gradient(values, 0.25).sum().backward()
        self.assertEqual(values.grad.item(), -0.25)

    def test_modality_mask_selects_one_channel_per_case(self) -> None:
        torch.manual_seed(7)
        images = torch.ones((3, 4, 2, 2, 2))
        masked, selected = mask_one_modality(images, 1.0)
        self.assertTrue(torch.all((selected >= 0) & (selected < 4)))
        for row, channel in enumerate(selected.tolist()):
            self.assertEqual(masked[row, channel].sum().item(), 0.0)
            self.assertEqual(masked[row].sum().item(), 24.0)

    def test_consistency_loss_has_no_teacher_gradient(self) -> None:
        full = torch.zeros((1, 1, 2, 2, 2), requires_grad=True)
        masked = torch.ones((1, 1, 2, 2, 2), requires_grad=True)
        modality_consistency_loss(full, masked).backward()
        self.assertIsNone(full.grad)
        self.assertIsNotNone(masked.grad)
