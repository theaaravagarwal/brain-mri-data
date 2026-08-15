"""Provenance-aware modality-consistency components for four-sequence MRI."""

from __future__ import annotations

import torch
from torch import nn


class _ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, values: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return values.view_as(values)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient.neg().mul(ctx.scale), None


def reverse_gradient(values: torch.Tensor, scale: float) -> torch.Tensor:
    return _ReverseGradient.apply(values, scale)


class PamcSegResNet(nn.Module):
    """SegResNet with a training-only source adversary.

    The segmentation decoder is used in every arm.  The domain head receives a
    gradient-reversed encoder representation only in the PAMC arm, encouraging
    the encoder to retain anatomy while discarding source-specific style.
    """

    def __init__(self, init_filters: int, source_count: int) -> None:
        super().__init__()
        from monai.networks.nets import SegResNet

        self.segmenter = SegResNet(
            spatial_dims=3,
            init_filters=init_filters,
            in_channels=4,
            out_channels=1,
            dropout_prob=0.2,
        )
        self.domain_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(init_filters * 8, init_filters * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(init_filters * 2, source_count),
        )

    def forward(self, images: torch.Tensor, adversary_scale: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, skips = self.segmenter.encode(images)
        logits = self.segmenter.decode(encoded, list(reversed(skips)))
        domain_logits = self.domain_head(reverse_gradient(encoded, adversary_scale))
        return logits, domain_logits


def mask_one_modality(images: torch.Tensor, probability: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero exactly one input sequence for selected cases, preserving tensor shape."""
    if images.ndim != 5 or images.shape[1] != 4:
        raise ValueError("PAMC expects [batch, 4, depth, height, width] MRI tensors")
    selected = torch.full((images.shape[0],), -1, dtype=torch.long, device=images.device)
    active = torch.rand(images.shape[0], device=images.device) < probability
    if not active.any():
        return images, selected
    selected[active] = torch.randint(0, 4, (int(active.sum()),), device=images.device)
    masked = images.clone()
    masked[torch.arange(images.shape[0], device=images.device)[active], selected[active]] = 0
    return masked, selected


def modality_consistency_loss(full_logits: torch.Tensor, masked_logits: torch.Tensor) -> torch.Tensor:
    """Keep the masked-view prediction close to the detached full-protocol view."""
    return torch.nn.functional.mse_loss(torch.sigmoid(masked_logits), torch.sigmoid(full_logits).detach())
