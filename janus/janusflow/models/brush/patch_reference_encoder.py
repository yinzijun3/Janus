"""Patch-level brush reference encoder for JanusFlow-Art."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BrushReferenceOutput:
    """Multi-scale texture tokens extracted from style-reference images."""

    coarse_tokens: torch.Tensor
    mid_tokens: torch.Tensor
    fine_tokens: torch.Tensor


class PatchBrushReferenceEncoder(nn.Module):
    """Encode reference images into coarse, mid, and fine brush tokens.

    The encoder is intentionally lightweight and local: it is meant to expose
    brushstroke/material cues to the decoder path without changing LLM prompt
    semantics or global composition priors.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        hidden_size: int = 512,
        base_channels: int = 64,
        num_coarse_tokens: int = 4,
        num_mid_tokens: int = 8,
        num_fine_tokens: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_coarse_tokens = num_coarse_tokens
        self.num_mid_tokens = num_mid_tokens
        self.num_fine_tokens = num_fine_tokens
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels * 4, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.coarse_pool = nn.AdaptiveAvgPool2d((2, max(1, num_coarse_tokens // 2)))
        self.mid_pool = nn.AdaptiveAvgPool2d((2, max(1, num_mid_tokens // 2)))
        self.fine_pool = nn.AdaptiveAvgPool2d((4, max(1, num_fine_tokens // 4)))
        self.token_norm = nn.LayerNorm(hidden_size)

    def _tokens_from_pool(self, features: torch.Tensor, pool: nn.Module, limit: int) -> torch.Tensor:
        tokens = pool(features).flatten(2).transpose(1, 2)
        if tokens.shape[1] > limit:
            tokens = tokens[:, :limit, :]
        if tokens.shape[1] < limit:
            pad_count = limit - tokens.shape[1]
            pad = tokens.new_zeros(tokens.shape[0], pad_count, tokens.shape[2])
            tokens = torch.cat([tokens, pad], dim=1)
        return self.token_norm(tokens)

    def forward(self, reference_images: torch.Tensor) -> BrushReferenceOutput:
        """Encode reference images.

        Shape:
        - `reference_images`: `[batch, 3, height, width]`
        """

        features = self.stem(reference_images)
        return BrushReferenceOutput(
            coarse_tokens=self._tokens_from_pool(features, self.coarse_pool, self.num_coarse_tokens),
            mid_tokens=self._tokens_from_pool(features, self.mid_pool, self.num_mid_tokens),
            fine_tokens=self._tokens_from_pool(features, self.fine_pool, self.num_fine_tokens),
        )


def mask_brush_reference_output(
    output: BrushReferenceOutput,
    has_reference_style_image: torch.Tensor,
) -> BrushReferenceOutput:
    """Zero brush tokens for rows without a valid reference image."""

    mask = has_reference_style_image.float().view(-1, 1, 1).to(output.coarse_tokens.device)
    mask = mask.to(dtype=output.coarse_tokens.dtype)
    return BrushReferenceOutput(
        coarse_tokens=output.coarse_tokens * mask,
        mid_tokens=output.mid_tokens * mask,
        fine_tokens=output.fine_tokens * mask,
    )


def repeat_brush_reference_output(output: BrushReferenceOutput) -> BrushReferenceOutput:
    """Repeat brush tokens for conditioned and unconditioned CFG rows."""

    return BrushReferenceOutput(
        coarse_tokens=torch.cat([output.coarse_tokens, output.coarse_tokens], dim=0),
        mid_tokens=torch.cat([output.mid_tokens, output.mid_tokens], dim=0),
        fine_tokens=torch.cat([output.fine_tokens, output.fine_tokens], dim=0),
    )
