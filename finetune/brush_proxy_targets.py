"""Proxy targets for texture-statistics and stroke-field brush supervision."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from finetune.janusflow_art_losses import gaussian_blur2d


def _rgb_to_grayscale(images: torch.Tensor) -> torch.Tensor:
    """Convert normalized RGB images `[-1, 1]` to grayscale `[0, 1]`."""

    rgb = ((images.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    return gray


def _sobel_gradients(gray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    return grad_x, grad_y


def _band_energy(gray: torch.Tensor) -> torch.Tensor:
    low = gaussian_blur2d(gray, kernel_size=9, sigma=2.0)
    mid = gaussian_blur2d(gray, kernel_size=5, sigma=1.0)
    high = gray - gaussian_blur2d(gray, kernel_size=3, sigma=0.7)
    bands = torch.cat(
        [
            (gray - low).abs(),
            (mid - low).abs(),
            high.abs(),
        ],
        dim=1,
    )
    return bands


def build_brush_proxy_targets(
    images: torch.Tensor,
    *,
    grid_size: int = 24,
    orientation_bins: int = 8,
) -> Dict[str, torch.Tensor]:
    """Build low-cost proxy supervision maps from target images.

    Returns:
    - `orientation_target`: `[B, H, W]` integer bin ids
    - `band_target`: `[B, 3, H, W]`
    - `density_target`: `[B, 1, H, W]`
    - `pigment_target`: `[B, 1, H, W]`
    """

    gray = _rgb_to_grayscale(images)
    gray = F.interpolate(gray, size=(grid_size, grid_size), mode="bilinear", align_corners=False)
    grad_x, grad_y = _sobel_gradients(gray)
    angle = torch.atan2(grad_y, grad_x + 1.0e-6)
    angle = torch.remainder(angle + math.pi, math.pi)
    orientation = torch.clamp((angle / math.pi * orientation_bins).floor().long(), 0, orientation_bins - 1)
    orientation = orientation.squeeze(1)

    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-6)
    local_complexity = gaussian_blur2d(magnitude, kernel_size=3, sigma=0.8)
    density = (magnitude + local_complexity)
    density = density / density.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)

    local_mean = gaussian_blur2d(gray, kernel_size=5, sigma=1.0)
    local_var = gaussian_blur2d((gray - local_mean).square(), kernel_size=5, sigma=1.0)
    pigment = torch.sqrt(local_var + 1.0e-6)
    pigment = pigment / pigment.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)

    bands = _band_energy(gray)
    bands = bands / bands.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)

    return {
        "orientation_target": orientation,
        "band_target": bands,
        "density_target": density,
        "pigment_target": pigment,
    }


def build_stroke_proxy_targets(
    images: torch.Tensor,
    *,
    grid_size: int = 24,
    prototype_orientations: Optional[torch.Tensor] = None,
    prototype_lengths: Optional[torch.Tensor] = None,
    prototype_widths: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Build stroke-oriented proxy supervision from target images.

    Returns:
    - `angle_target`: `[B, 1, H, W]` dominant axial orientation in radians
    - `length_target`: `[B, 1, H, W]` longer in coherent edge regions
    - `width_target`: `[B, 1, H, W]` broader in low-coherence regions
    - `alpha_target`: `[B, 1, H, W]` local stroke occupancy/intensity proxy
    - `objectness_target`: `[B, 1, H, W]` sharper occupancy target
    - `blank_region_target`: `[B, 1, H, W]` blank/low-support suppression mask
    - `support_target`: `[B, 1, H, W]` local support evidence
    - `support_dilated_target`: `[B, 1, H, W]` support with slight dilation
    - `prototype_target`: `[B, H, W]` nearest prototype id when anchors are supplied
    """

    gray = _rgb_to_grayscale(images)
    gray = F.interpolate(gray, size=(grid_size, grid_size), mode="bilinear", align_corners=False)
    grad_x, grad_y = _sobel_gradients(gray)
    angle = torch.atan2(grad_y, grad_x + 1.0e-6)
    angle = torch.remainder(angle + math.pi, math.pi)

    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-6)
    j_xx = gaussian_blur2d(grad_x.square(), kernel_size=5, sigma=1.0)
    j_xy = gaussian_blur2d(grad_x * grad_y, kernel_size=5, sigma=1.0)
    j_yy = gaussian_blur2d(grad_y.square(), kernel_size=5, sigma=1.0)
    trace = j_xx + j_yy
    det_term = torch.sqrt((j_xx - j_yy).square() + 4.0 * j_xy.square() + 1.0e-6)
    lambda_1 = 0.5 * (trace + det_term)
    lambda_2 = 0.5 * (trace - det_term)
    coherence = (lambda_1 - lambda_2).abs() / (lambda_1 + lambda_2 + 1.0e-6)
    coherence = coherence.clamp(0.0, 1.0)

    local_complexity = gaussian_blur2d(magnitude, kernel_size=3, sigma=0.8)
    density = magnitude + local_complexity
    density = density / density.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)

    local_mean = gaussian_blur2d(gray, kernel_size=5, sigma=1.0)
    local_var = gaussian_blur2d((gray - local_mean).square(), kernel_size=5, sigma=1.0)
    pigment = torch.sqrt(local_var + 1.0e-6)
    pigment = pigment / pigment.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)

    length_target = 0.45 + 0.60 * coherence
    width_target = 0.14 + 0.14 * (1.0 - coherence)
    alpha_target = (0.55 * density + 0.45 * pigment).clamp(0.0, 1.0)
    objectness_target = torch.clamp((alpha_target - 0.30) / 0.70, min=0.0, max=1.0)
    support_target = torch.maximum(density, pigment)
    blank_region_target = torch.clamp((0.22 - support_target) / 0.22, min=0.0, max=1.0)
    blank_region_target = gaussian_blur2d(blank_region_target, kernel_size=3, sigma=0.8).clamp(0.0, 1.0)
    support_dilated_target = F.max_pool2d(support_target, kernel_size=5, stride=1, padding=2).clamp(0.0, 1.0)

    output = {
        "angle_target": angle,
        "length_target": length_target,
        "width_target": width_target,
        "alpha_target": alpha_target,
        "objectness_target": objectness_target,
        "blank_region_target": blank_region_target,
        "support_target": support_target,
        "support_dilated_target": support_dilated_target,
    }

    if (
        prototype_orientations is not None
        and prototype_lengths is not None
        and prototype_widths is not None
    ):
        anchors_theta = prototype_orientations.to(device=gray.device, dtype=gray.dtype).view(1, -1, 1, 1)
        anchors_length = prototype_lengths.to(device=gray.device, dtype=gray.dtype).view(1, -1, 1, 1)
        anchors_width = prototype_widths.to(device=gray.device, dtype=gray.dtype).view(1, -1, 1, 1)
        angle_expanded = angle.expand(-1, anchors_theta.shape[1], -1, -1)
        axial_distance = 1.0 - torch.abs(torch.cos(angle_expanded - anchors_theta))
        length_distance = ((length_target.expand_as(axial_distance) - anchors_length) ** 2) / 0.08
        width_distance = ((width_target.expand_as(axial_distance) - anchors_width) ** 2) / 0.01
        prototype_target = torch.argmin(axial_distance + length_distance + width_distance, dim=1)
        output["prototype_target"] = prototype_target

    return output


__all__ = ["build_brush_proxy_targets", "build_stroke_proxy_targets"]
