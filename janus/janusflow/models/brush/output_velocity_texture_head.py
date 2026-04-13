"""Output-space texture head for JanusFlow-Art velocity prediction."""

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


class OutputVelocityTextureHead(nn.Module):
    """Apply a gated high-frequency residual directly on velocity outputs."""

    def __init__(
        self,
        *,
        in_channels: int = 4,
        hidden_channels: int = 64,
        kernel_size: int = 5,
        residual_scale: float = 0.35,
        dropout: float = 0.02,
        gate_bias_init: float = -2.2,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.residual_scale = float(residual_scale)

        self.gate_branch = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=in_channels),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )
        nn.init.constant_(self.gate_branch[-1].bias, float(gate_bias_init))

        self.texture_branch = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=in_channels),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=hidden_channels,
            ),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
        )
        self._last_gate_map: torch.Tensor | None = None

    def forward(self, pred_velocity: torch.Tensor) -> torch.Tensor:
        gate_map = torch.sigmoid(self.gate_branch(pred_velocity))
        delta = self.texture_branch(pred_velocity)
        delta_hp = delta - _fixed_blur3x3(delta)
        self._last_gate_map = gate_map
        return pred_velocity + gate_map * delta_hp * self.residual_scale

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


__all__ = ["OutputVelocityTextureHead"]
