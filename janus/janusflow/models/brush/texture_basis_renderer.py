"""Constrained texture-basis renderer for brushstroke statistics heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_orientation_bank(
    channels: int,
    orientation_bins: int,
    kernel_size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build a fixed bank of oriented derivative filters.

    Returns a tensor shaped `[orientation_bins * channels, 1, k, k]` so it can
    be used with grouped convolution over a `channels`-wide feature map.
    """

    coords = torch.linspace(-1.0, 1.0, kernel_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    bank = []
    for index in range(orientation_bins):
        theta = math.pi * float(index) / float(max(orientation_bins, 1))
        axis = xx * math.cos(theta) + yy * math.sin(theta)
        ortho = -xx * math.sin(theta) + yy * math.cos(theta)
        gaussian = torch.exp(-(axis.square() + ortho.square()) / 0.35)
        derivative = axis * gaussian
        derivative = derivative - derivative.mean()
        derivative = derivative / derivative.abs().sum().clamp_min(1.0e-6)
        bank.append(derivative)
    kernels = torch.stack(bank, dim=0).view(orientation_bins, 1, kernel_size, kernel_size)
    kernels = kernels.repeat_interleave(channels, dim=0)
    return kernels.contiguous()


class TextureBasisRenderer(nn.Module):
    """Render constrained texture residuals from explicit brush statistics.

    Expected shapes:
    - `feature_map`: `[B, C, H, W]`
    - `orientation_logits`: `[B, O, H, W]`
    - `scale_logits`: `[B, S, H, W]`
    - `density_map`: `[B, 1, H, W]`
    - `pigment_map`: `[B, 1, H, W]`
    - `band_logits`: `[B, K, H, W]`
    """

    def __init__(
        self,
        *,
        feature_channels: int = 768,
        basis_channels: int = 96,
        orientation_bins: int = 8,
        scale_bins: int = 3,
        band_count: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.02,
    ) -> None:
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.orientation_bins = int(orientation_bins)
        self.scale_bins = int(scale_bins)
        self.band_count = int(band_count)

        self.pre_norm = nn.GroupNorm(num_groups=32, num_channels=feature_channels)
        self.pre_reduce = nn.Conv2d(feature_channels, basis_channels, kernel_size=1)
        self.pre_mix = nn.Conv2d(
            basis_channels,
            basis_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=basis_channels,
        )
        self.post_mix = nn.Conv2d(basis_channels, feature_channels, kernel_size=1)
        self.band_mix = nn.Conv2d(band_count, basis_channels, kernel_size=1)
        self.output_norm = nn.GroupNorm(num_groups=max(1, min(32, feature_channels // 8)), num_channels=feature_channels)
        self.dropout = nn.Dropout2d(float(dropout))
        self.activation = nn.GELU()
        self.kernel_size = int(kernel_size)

        orientation_bank = _make_orientation_bank(
            basis_channels,
            self.orientation_bins,
            self.kernel_size,
        )
        self.register_buffer("orientation_bank", orientation_bank, persistent=False)
        scale_values = torch.linspace(0.6, 1.4, steps=max(self.scale_bins, 1))
        self.register_buffer("scale_values", scale_values.view(1, self.scale_bins, 1, 1), persistent=False)

        self._last_basis_usage_entropy: torch.Tensor | None = None

    def forward(
        self,
        feature_map: torch.Tensor,
        *,
        orientation_logits: torch.Tensor,
        scale_logits: torch.Tensor,
        density_map: torch.Tensor,
        pigment_map: torch.Tensor,
        band_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Render a constrained residual tensor shaped `[B, C, H, W]`."""

        # `[B, C, H, W]`
        reduced = self.pre_reduce(self.pre_norm(feature_map))
        reduced = self.pre_mix(reduced)
        reduced = self.activation(reduced)
        reduced = self.dropout(reduced)

        batch, channels, height, width = reduced.shape
        oriented = F.conv2d(
            reduced,
            self.orientation_bank.to(device=reduced.device, dtype=reduced.dtype),
            padding=self.kernel_size // 2,
            groups=channels,
        )
        oriented = oriented.view(batch, self.orientation_bins, channels, height, width)
        orientation_probs = torch.softmax(orientation_logits, dim=1).unsqueeze(2)
        oriented = (orientation_probs * oriented).sum(dim=1)

        scale_probs = torch.softmax(scale_logits, dim=1)
        scale_gain = (scale_probs * self.scale_values.to(device=reduced.device, dtype=reduced.dtype)).sum(dim=1, keepdim=True)
        band_gain = self.band_mix(torch.softmax(band_logits, dim=1))
        density_gain = torch.sigmoid(density_map)
        pigment_gain = torch.sigmoid(pigment_map)

        mixed = oriented * scale_gain
        mixed = mixed + band_gain
        mixed = mixed * (0.35 + 0.65 * density_gain)
        mixed = mixed * (0.35 + 0.65 * pigment_gain)
        rendered = self.post_mix(mixed)
        rendered = self.output_norm(rendered)
        rendered = self.activation(rendered)

        orientation_entropy = -(orientation_probs.squeeze(2) * torch.log(orientation_probs.squeeze(2).clamp_min(1.0e-6))).sum(dim=1).mean()
        scale_entropy = -(scale_probs * torch.log(scale_probs.clamp_min(1.0e-6))).sum(dim=1).mean()
        band_probs = torch.softmax(band_logits, dim=1)
        band_entropy = -(band_probs * torch.log(band_probs.clamp_min(1.0e-6))).sum(dim=1).mean()
        self._last_basis_usage_entropy = orientation_entropy + 0.5 * scale_entropy + 0.5 * band_entropy
        return rendered

    def basis_usage_entropy_loss(self) -> torch.Tensor | None:
        """Return the latest basis-usage entropy regularizer."""

        return self._last_basis_usage_entropy


__all__ = ["TextureBasisRenderer"]
