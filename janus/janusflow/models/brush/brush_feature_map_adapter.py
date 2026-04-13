"""Feature-map brush adapters for JanusFlow-Art."""

from __future__ import annotations

import torch
import torch.nn as nn

from .patch_reference_encoder import BrushReferenceOutput


class BrushFeatureMapAdapter(nn.Module):
    """Local residual adapter that operates on decoder feature maps.

    The module runs after decoder tokens are reshaped into
    `[batch, channels, height, width]` so it can modify local image structure
    more directly than a token-only adapter.
    """

    def __init__(
        self,
        *,
        feature_channels: int = 768,
        style_hidden_size: int = 512,
        hidden_channels: int = 384,
        kernel_size: int = 3,
        residual_init: float = 0.05,
        residual_scale: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.residual_scale = float(residual_scale)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=feature_channels)
        self.in_proj = nn.Conv2d(feature_channels, hidden_channels, kernel_size=1)
        self.depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_channels,
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        self.out_proj = nn.Conv2d(hidden_channels, feature_channels, kernel_size=1)
        self.film = nn.Linear(style_hidden_size, feature_channels * 2)
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_init)))

    def _conditioning_embedding(
        self,
        brush_output: BrushReferenceOutput | None,
        feature_map: torch.Tensor,
    ) -> torch.Tensor:
        if brush_output is None:
            return feature_map.new_zeros(feature_map.shape[0], self.film.in_features)
        return brush_output.coarse_tokens.mean(dim=1)

    def forward(
        self,
        feature_map: torch.Tensor,
        brush_output: BrushReferenceOutput | None = None,
    ) -> torch.Tensor:
        """Return brush-enhanced decoder feature maps.

        Shape:
        - `feature_map`: `[batch, feature_channels, height, width]`
        """

        cond = self._conditioning_embedding(brush_output, feature_map).to(dtype=feature_map.dtype)
        scale, shift = self.film(cond).chunk(2, dim=-1)
        scale = torch.tanh(scale).unsqueeze(-1).unsqueeze(-1)
        shift = torch.tanh(shift).unsqueeze(-1).unsqueeze(-1)

        hidden = self.norm(feature_map)
        hidden = self.in_proj(hidden)
        hidden = self.depthwise(hidden)
        hidden = self.out_proj(self.dropout(self.act(hidden)))
        hidden = hidden * (1.0 + scale) + shift

        gate = self.residual_gate.to(dtype=feature_map.dtype) * self.residual_scale
        return feature_map + hidden * gate
