"""Deterministic, case-linked whole-lesion evaluation helpers."""

from __future__ import annotations

import hashlib
from typing import Iterable

import torch


MODALITY_ORDER = ("t1", "t1ce", "t2", "flair")


def deterministic_mask_one_modality(
    images: torch.Tensor, case_ids: Iterable[str], seed: int
) -> tuple[torch.Tensor, list[str]]:
    """Mask one fixed modality per case, independent of loader order or hardware."""
    case_ids = list(case_ids)
    if images.ndim != 5 or images.shape[1] != len(MODALITY_ORDER):
        raise ValueError("Expected a [batch, 4, depth, height, width] MRI tensor")
    if len(case_ids) != images.shape[0]:
        raise ValueError("Every image in the batch needs a case ID")
    masked = images.clone()
    selected: list[str] = []
    for index, case_id in enumerate(case_ids):
        digest = hashlib.sha256(f"{seed}:{case_id}".encode()).digest()
        modality_index = int.from_bytes(digest[:8], "big") % len(MODALITY_ORDER)
        masked[index, modality_index] = 0
        selected.append(MODALITY_ORDER[modality_index])
    return masked, selected


def dice_per_case(logits: torch.Tensor, labels: torch.Tensor) -> list[float]:
    prediction = torch.sigmoid(logits) > 0.5
    truth = labels > 0.5
    numerator = 2 * (prediction & truth).sum(dim=(1, 2, 3, 4)).float()
    denominator = prediction.sum(dim=(1, 2, 3, 4)).float() + truth.sum(dim=(1, 2, 3, 4)).float()
    return ((numerator + 1e-6) / (denominator + 1e-6)).detach().cpu().tolist()


def _box(mask: torch.Tensor) -> tuple[int, int, int, int, int, int] | None:
    """Return an inclusive voxel box for one binary whole-lesion mask."""
    values = torch.as_tensor(mask).bool().squeeze()
    locations = values.nonzero(as_tuple=False)
    if locations.numel() == 0:
        return None
    low = locations.min(dim=0).values.tolist()
    high = locations.max(dim=0).values.tolist()
    return tuple(int(value) for value in (*low, *high))


def box_iou_per_case(logits: torch.Tensor, labels: torch.Tensor) -> list[float]:
    """IoU of boxes derived from predicted and reference whole-lesion masks."""
    prediction = torch.sigmoid(logits) > 0.5
    truth = labels > 0.5
    values: list[float] = []
    for predicted, reference in zip(prediction, truth, strict=True):
        predicted_box, reference_box = _box(predicted), _box(reference)
        if predicted_box is None or reference_box is None:
            values.append(1.0 if predicted_box == reference_box else 0.0)
            continue
        pred_low, pred_high = predicted_box[:3], predicted_box[3:]
        ref_low, ref_high = reference_box[:3], reference_box[3:]
        intersection = [max(0, min(a, b) - max(c, d) + 1) for a, b, c, d in zip(pred_high, ref_high, pred_low, ref_low, strict=True)]
        pred_volume = [high - low + 1 for low, high in zip(pred_low, pred_high, strict=True)]
        ref_volume = [high - low + 1 for low, high in zip(ref_low, ref_high, strict=True)]
        overlap = intersection[0] * intersection[1] * intersection[2]
        union = pred_volume[0] * pred_volume[1] * pred_volume[2] + ref_volume[0] * ref_volume[1] * ref_volume[2] - overlap
        values.append(float(overlap / union))
    return values
