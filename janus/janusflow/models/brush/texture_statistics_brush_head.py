"""Texture-statistics brush head for constrained painterly residuals."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .texture_basis_renderer import TextureBasisRenderer


def _fixed_blur3x3(tensor: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        device=tensor.device,
        dtype=tensor.dtype,
    )
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3)
    kernel = kernel.repeat(tensor.shape[1], 1, 1, 1)
    return F.conv2d(tensor, kernel, padding=1, groups=tensor.shape[1])


@dataclass
class TextureStatisticsOutput:
    """Auxiliary statistics predicted by the texture-statistics head."""

    orientation_logits: torch.Tensor
    scale_logits: torch.Tensor
    density_map: torch.Tensor
    pigment_map: torch.Tensor
    band_logits: torch.Tensor
    gate_map: torch.Tensor


class TextureStatisticsBrushHead(nn.Module):
    """Predict constrained brush statistics and render high-pass residuals.

    Shapes:
    - input `feature_map`: `[B, C, H, W]`
    - output `feature_map`: `[B, C, H, W]`
    """

    def __init__(
        self,
        *,
        feature_channels: int = 768,
        hidden_channels: int = 512,
        basis_channels: int = 96,
        orientation_bins: int = 8,
        scale_bins: int = 3,
        band_count: int = 3,
        kernel_size: int = 5,
        residual_init: float = 0.18,
        residual_scale: float = 1.5,
        dropout: float = 0.02,
        gate_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_init)))

        self.statistics_stem = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=feature_channels),
            nn.Conv2d(feature_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.orientation_head = nn.Conv2d(hidden_channels, orientation_bins, kernel_size=1)
        self.scale_head = nn.Conv2d(hidden_channels, scale_bins, kernel_size=1)
        self.density_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.pigment_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.band_head = nn.Conv2d(hidden_channels, band_count, kernel_size=1)
        self.gate_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        nn.init.constant_(self.gate_head.bias, float(gate_bias_init))

        self.renderer = TextureBasisRenderer(
            feature_channels=feature_channels,
            basis_channels=basis_channels,
            orientation_bins=orientation_bins,
            scale_bins=scale_bins,
            band_count=band_count,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self._last_statistics: TextureStatisticsOutput | None = None

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Apply the constrained texture residual to a decoder feature map."""

        hidden = self.statistics_stem(feature_map)
        orientation_logits = self.orientation_head(hidden)
        scale_logits = self.scale_head(hidden)
        density_map = self.density_head(hidden)
        pigment_map = self.pigment_head(hidden)
        band_logits = self.band_head(hidden)
        gate_map = torch.sigmoid(self.gate_head(hidden))

        texture_residual = self.renderer(
            feature_map,
            orientation_logits=orientation_logits,
            scale_logits=scale_logits,
            density_map=density_map,
            pigment_map=pigment_map,
            band_logits=band_logits,
        )
        texture_residual_hp = texture_residual - _fixed_blur3x3(texture_residual)
        scalar_gate = self.residual_gate.to(dtype=feature_map.dtype) * self.residual_scale
        self._last_statistics = TextureStatisticsOutput(
            orientation_logits=orientation_logits,
            scale_logits=scale_logits,
            density_map=density_map,
            pigment_map=pigment_map,
            band_logits=band_logits,
            gate_map=gate_map,
        )
        return feature_map + gate_map * texture_residual_hp * scalar_gate

    def latest_statistics(self) -> TextureStatisticsOutput | None:
        """Return cached statistics from the most recent forward pass."""

        return self._last_statistics

    def gate_l1_loss(self) -> torch.Tensor | None:
        if self._last_statistics is None:
            return None
        return self._last_statistics.gate_map.abs().mean()

    def gate_tv_loss(self) -> torch.Tensor | None:
        if self._last_statistics is None:
            return None
        gate = self._last_statistics.gate_map
        tv_h = (gate[:, :, 1:, :] - gate[:, :, :-1, :]).abs().mean()
        tv_w = (gate[:, :, :, 1:] - gate[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w

    def basis_usage_entropy_loss(self) -> torch.Tensor | None:
        """Expose the renderer entropy regularizer for training."""

        return self.renderer.basis_usage_entropy_loss()


__all__ = ["TextureStatisticsBrushHead", "TextureStatisticsOutput"]
