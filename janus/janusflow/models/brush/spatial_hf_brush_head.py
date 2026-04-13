"""Spatially gated high-frequency brush head for JanusFlow-Art."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fixed_blur3x3(tensor: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        device=tensor.device,
        dtype=tensor.dtype,
    )
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3)
    kernel = kernel.repeat(tensor.shape[1], 1, 1, 1)
    return F.conv2d(tensor, kernel, padding=1, groups=tensor.shape[1])


class SpatialHFBrushHead(nn.Module):
    """Apply spatially gated high-frequency residuals on decoder feature maps."""

    def __init__(
        self,
        *,
        feature_channels: int = 768,
        hidden_channels: int = 512,
        kernel_size: int = 5,
        residual_init: float = 0.18,
        residual_scale: float = 1.5,
        dropout: float = 0.02,
        gate_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.residual_scale = float(residual_scale)
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_init)))

        self.gate_branch = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=feature_channels),
            nn.Conv2d(feature_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )
        nn.init.constant_(self.gate_branch[-1].bias, float(gate_bias_init))

        self.texture_norm = nn.GroupNorm(num_groups=32, num_channels=feature_channels)
        self.texture_in = nn.Conv2d(feature_channels, hidden_channels, kernel_size=1)
        self.texture_dw1 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_channels,
        )
        self.texture_dw2 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_channels,
        )
        self.texture_out = nn.Conv2d(hidden_channels, feature_channels, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)

        self._last_gate_map: torch.Tensor | None = None

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        gate_map = torch.sigmoid(self.gate_branch(feature_map))

        hidden = self.texture_norm(feature_map)
        hidden = self.texture_in(hidden)
        hidden = self.texture_dw1(hidden)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        hidden = self.texture_dw2(hidden)
        hidden = self.act(hidden)
        hidden = self.texture_out(hidden)

        delta_hp = hidden - _fixed_blur3x3(hidden)
        scalar_gate = self.residual_gate.to(dtype=feature_map.dtype) * self.residual_scale
        self._last_gate_map = gate_map
        return feature_map + gate_map * delta_hp * scalar_gate

    def gate_l1_loss(self) -> torch.Tensor | None:
        if self._last_gate_map is None:
            return None
        return self._last_gate_map.abs().mean()

    def gate_tv_loss(self) -> torch.Tensor | None:
        if self._last_gate_map is None:
            return None
        gate = self._last_gate_map
        tv_h = (gate[:, :, 1:, :] - gate[:, :, :-1, :]).abs().mean()
        tv_w = (gate[:, :, :, 1:] - gate[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w


__all__ = ["SpatialHFBrushHead"]
