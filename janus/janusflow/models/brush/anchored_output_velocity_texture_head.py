"""Structure-anchored output-space texture head for JanusFlow-Art."""

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


def _gaussian_blur2d(
    tensor: torch.Tensor,
    *,
    kernel_size: int,
    sigma: float,
) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=tensor.device, dtype=tensor.dtype)
    coords = coords - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-(coords * coords) / max(2.0 * sigma * sigma, 1.0e-6))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1.0e-6)
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, kernel_size, kernel_size)
    weight = kernel_2d.repeat(tensor.shape[1], 1, 1, 1)
    padding = kernel_size // 2
    return F.conv2d(tensor, weight, padding=padding, groups=tensor.shape[1])


class AnchoredOutputVelocityTextureHead(nn.Module):
    """Apply a high-frequency velocity residual with an explicit structure-protect mask.

    The protection mask is computed from the low-frequency magnitude of the incoming
    base velocity field. Regions with strong low-frequency structure receive less
    edit budget, encouraging the head to spend capacity on texture/background areas
    instead of rewriting the main subject silhouette.
    """

    def __init__(
        self,
        *,
        in_channels: int = 4,
        hidden_channels: int = 96,
        kernel_size: int = 5,
        residual_scale: float = 0.35,
        dropout: float = 0.02,
        gate_bias_init: float = -2.2,
        anchor_kernel_size: int = 9,
        anchor_sigma: float = 2.0,
        anchor_strength: float = 0.85,
        anchor_threshold: float = 0.30,
        anchor_sharpness: float = 12.0,
        min_edit_scale: float = 0.10,
        center_prior_strength: float = 0.0,
        center_prior_sigma_x: float = 0.22,
        center_prior_sigma_y: float = 0.18,
        center_prior_y_offset: float = -0.10,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.residual_scale = float(residual_scale)
        self.anchor_kernel_size = int(anchor_kernel_size)
        self.anchor_sigma = float(anchor_sigma)
        self.anchor_strength = float(anchor_strength)
        self.anchor_threshold = float(anchor_threshold)
        self.anchor_sharpness = float(anchor_sharpness)
        self.min_edit_scale = float(min_edit_scale)
        self.center_prior_strength = float(center_prior_strength)
        self.center_prior_sigma_x = float(center_prior_sigma_x)
        self.center_prior_sigma_y = float(center_prior_sigma_y)
        self.center_prior_y_offset = float(center_prior_y_offset)

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
        self._last_structure_mask: torch.Tensor | None = None

    def _center_prior(
        self,
        *,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        prior = torch.exp(
            -0.5
            * (
                (grid_x / max(self.center_prior_sigma_x, 1.0e-6)) ** 2
                + ((grid_y - self.center_prior_y_offset) / max(self.center_prior_sigma_y, 1.0e-6)) ** 2
            )
        )
        prior = prior / prior.max().clamp_min(1.0e-6)
        return prior.view(1, 1, height, width)

    def forward(self, pred_velocity: torch.Tensor) -> torch.Tensor:
        gate_map = torch.sigmoid(self.gate_branch(pred_velocity))
        delta = self.texture_branch(pred_velocity)
        delta_hp = delta - _fixed_blur3x3(delta)

        # [batch, 4, h, w] -> [batch, 1, h, w] low-frequency structure energy.
        structure_energy = pred_velocity.abs().mean(dim=1, keepdim=True)
        structure_energy = _gaussian_blur2d(
            structure_energy,
            kernel_size=self.anchor_kernel_size,
            sigma=self.anchor_sigma,
        )
        structure_energy = structure_energy / structure_energy.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
        structure_mask = torch.sigmoid(
            (structure_energy - self.anchor_threshold) * self.anchor_sharpness
        )
        if self.center_prior_strength > 0.0:
            center_prior = self._center_prior(
                height=pred_velocity.shape[-2],
                width=pred_velocity.shape[-1],
                device=pred_velocity.device,
                dtype=pred_velocity.dtype,
            )
            structure_mask = (structure_mask + self.center_prior_strength * center_prior).clamp_(0.0, 1.0)
        edit_scale = 1.0 - self.anchor_strength * structure_mask
        edit_scale = edit_scale.clamp(min=self.min_edit_scale, max=1.0)

        self._last_gate_map = gate_map
        self._last_structure_mask = structure_mask
        return pred_velocity + gate_map * edit_scale * delta_hp * self.residual_scale

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

    def protected_gate_l1_loss(self) -> torch.Tensor | None:
        if self._last_gate_map is None or self._last_structure_mask is None:
            return None
        return (self._last_gate_map * self._last_structure_mask).abs().mean()


__all__ = ["AnchoredOutputVelocityTextureHead"]
