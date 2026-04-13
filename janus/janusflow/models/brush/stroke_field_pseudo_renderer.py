"""Feature-conditioned pseudo-renderer for stroke-field probes."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StrokeFieldPseudoRenderer(nn.Module):
    """Compose a directional stroke residual from explicit stroke fields.

    The renderer stays dependency-free, but it is more explicit than the first
    probe version:

    - a projected carrier map is extracted from the decoder feature map
    - a bank of oriented stroke prototypes is applied with spatial convolution
    - per-pixel `theta/length/width/prototype_logits` reweight those responses
    - a small learned post-mix turns the scalar stroke features back into a
      feature-space residual
    """

    def __init__(
        self,
        *,
        feature_channels: int = 768,
        prototype_count: int = 8,
        kernel_size: int = 9,
        render_channels: int = 8,
        topk_prototypes: int = 2,
        support_gate_enabled: bool = False,
        support_threshold: float = 0.20,
        support_temperature: float = 0.08,
        support_dilation: int = 5,
        support_mix: float = 0.65,
        support_hard_mask_enabled: bool = False,
        support_opening_kernel: int = 0,
    ) -> None:
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.prototype_count = int(prototype_count)
        self.kernel_size = int(kernel_size)
        self.render_channels = int(render_channels)
        self.topk_prototypes = max(int(topk_prototypes), 1)
        self.support_gate_enabled = bool(support_gate_enabled)
        self.support_threshold = float(support_threshold)
        self.support_temperature = max(float(support_temperature), 1.0e-4)
        dilation = max(int(support_dilation), 1)
        self.support_dilation = dilation if dilation % 2 == 1 else dilation + 1
        self.support_mix = float(support_mix)
        self.support_hard_mask_enabled = bool(support_hard_mask_enabled)
        opening_kernel = max(int(support_opening_kernel), 0)
        if opening_kernel > 0 and opening_kernel % 2 == 0:
            opening_kernel += 1
        self.support_opening_kernel = opening_kernel
        self.feature_project = nn.Sequential(
            nn.Conv2d(self.feature_channels, self.render_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.render_channels, self.render_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.post_mix = nn.Sequential(
            nn.Conv2d(self.render_channels + 4, self.render_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.render_channels, self.feature_channels, kernel_size=1),
        )
        prototype_bank, orientation_anchors, length_anchors, width_anchors = self._build_prototype_bank()
        self.register_buffer("prototype_bank", prototype_bank, persistent=False)
        self.register_buffer("orientation_anchors", orientation_anchors, persistent=False)
        self.register_buffer("length_anchors", length_anchors, persistent=False)
        self.register_buffer("width_anchors", width_anchors, persistent=False)
        self.prototype_delta = nn.Parameter(torch.zeros_like(prototype_bank))

    @staticmethod
    def _normalize_map(tensor: torch.Tensor) -> torch.Tensor:
        """Normalize a BCHW map to `[0, 1]` per sample."""

        min_value = tensor.amin(dim=(-2, -1), keepdim=True)
        max_value = tensor.amax(dim=(-2, -1), keepdim=True)
        return (tensor - min_value) / (max_value - min_value).clamp_min(1.0e-6)

    def _build_prototype_bank(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a fixed oriented stroke bank plus anchor metadata.

        Shapes:
        - prototype bank: `[prototype_count, 1, kernel, kernel]`
        - anchor tensors: `[prototype_count]`
        """

        axis = torch.linspace(-1.0, 1.0, self.kernel_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        kernels = []
        orientation_anchors = []
        length_anchors = []
        width_anchors = []
        length_levels = torch.tensor([0.45, 0.75, 1.05], dtype=torch.float32)
        width_levels = torch.tensor([0.14, 0.20, 0.28], dtype=torch.float32)
        for index in range(self.prototype_count):
            angle = 2.0 * math.pi * float(index) / max(self.prototype_count, 1)
            length_anchor = float(length_levels[index % len(length_levels)])
            width_anchor = float(width_levels[(index // len(length_levels)) % len(width_levels)])
            x_rot = xx * math.cos(angle) + yy * math.sin(angle)
            y_rot = -xx * math.sin(angle) + yy * math.cos(angle)
            taper = (1.0 - torch.clamp(x_rot.abs() / max(length_anchor, 1e-3), min=0.0, max=1.0)) ** 1.5
            core = torch.exp(
                -0.5 * ((x_rot / max(length_anchor, 1e-3)) ** 2 + (y_rot / max(width_anchor, 1e-3)) ** 2)
            )
            ridge = taper * core * torch.clamp(1.0 - (y_rot / max(width_anchor, 1e-3)) ** 2, min=0.0)
            kernel = ridge - ridge.mean()
            kernel = kernel / kernel.abs().sum().clamp_min(1e-6)
            kernels.append(kernel)
            orientation_anchors.append(angle)
            length_anchors.append(length_anchor)
            width_anchors.append(width_anchor)
        bank = torch.stack(kernels, dim=0).unsqueeze(1)
        return (
            bank,
            torch.tensor(orientation_anchors, dtype=torch.float32),
            torch.tensor(length_anchors, dtype=torch.float32),
            torch.tensor(width_anchors, dtype=torch.float32),
        )

    @staticmethod
    def _binary_open(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Apply a lightweight morphological opening to remove isolated blobs."""

        if kernel_size <= 1:
            return mask
        padding = kernel_size // 2
        # BCHW binary erosion via min-pooling, then dilation via max-pooling.
        eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=padding)
        opened = F.max_pool2d(eroded, kernel_size=kernel_size, stride=1, padding=padding)
        return opened.clamp_(0.0, 1.0)

    def forward(
        self,
        feature_map: torch.Tensor,
        *,
        theta: torch.Tensor,
        length: torch.Tensor,
        width: torch.Tensor,
        curvature: torch.Tensor,
        alpha: torch.Tensor,
        objectness: torch.Tensor,
        prototype_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Render a stroke residual shaped `[B, C, H, W]`.

        Input shapes:
        - `feature_map`: `[B, C, H, W]`
        - predicted fields: `[B, 1, H, W]` except `prototype_logits` `[B, P, H, W]`
        """

        carrier = self.feature_project(feature_map)
        carrier_mean = carrier.mean(dim=1, keepdim=True)
        carrier_hp = carrier_mean - F.avg_pool2d(carrier_mean, kernel_size=3, stride=1, padding=1)

        kernels = (self.prototype_bank + 0.05 * self.prototype_delta).to(
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        responses = F.conv2d(carrier_hp, kernels, padding=self.kernel_size // 2)

        theta_angle = torch.tanh(theta) * math.pi
        theta_delta = theta_angle - self.orientation_anchors.view(1, self.prototype_count, 1, 1).to(
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        theta_score = torch.cos(theta_delta)

        length_value = torch.sigmoid(length) * 0.85 + 0.25
        width_value = torch.sigmoid(width) * 0.22 + 0.08
        length_score = -((length_value - self.length_anchors.view(1, self.prototype_count, 1, 1).to(
            device=feature_map.device,
            dtype=feature_map.dtype,
        )) ** 2) / 0.08
        width_score = -((width_value - self.width_anchors.view(1, self.prototype_count, 1, 1).to(
            device=feature_map.device,
            dtype=feature_map.dtype,
        )) ** 2) / 0.01

        prototype_bias = prototype_logits + 1.25 * theta_score + length_score + width_score
        prototype_weights = torch.softmax(prototype_bias, dim=1)
        if self.topk_prototypes < self.prototype_count:
            topk_values, topk_indices = torch.topk(
                prototype_weights,
                k=self.topk_prototypes,
                dim=1,
            )
            sparse_weights = torch.zeros_like(prototype_weights)
            sparse_weights.scatter_(1, topk_indices, topk_values)
            prototype_weights = sparse_weights / sparse_weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)

        alpha_map = torch.sigmoid(alpha)
        objectness_map = torch.sigmoid(objectness)
        curvature_map = torch.tanh(curvature)
        stroke_scalar = (responses * prototype_weights).sum(dim=1, keepdim=True)
        stroke_scalar = stroke_scalar - F.avg_pool2d(stroke_scalar, kernel_size=3, stride=1, padding=1)
        density = carrier.abs().mean(dim=1, keepdim=True)
        density = density / density.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        direction_x = torch.cos(theta_angle)
        direction_y = torch.sin(theta_angle)
        objectness_blob = F.avg_pool2d(objectness_map, kernel_size=3, stride=1, padding=1)
        if self.support_gate_enabled:
            carrier_support = self._normalize_map(
                F.avg_pool2d(carrier_hp.abs(), kernel_size=3, stride=1, padding=1)
            )
            density_support = self._normalize_map(density)
            support_evidence = self.support_mix * carrier_support + (1.0 - self.support_mix) * density_support
            support_gate = torch.sigmoid((support_evidence - self.support_threshold) / self.support_temperature)
            if self.support_dilation > 1:
                padding = self.support_dilation // 2
                support_gate = F.max_pool2d(
                    support_gate,
                    kernel_size=self.support_dilation,
                    stride=1,
                    padding=padding,
                )
            if self.support_hard_mask_enabled:
                support_mask = (support_evidence >= self.support_threshold).to(feature_map.dtype)
                if self.support_opening_kernel > 1:
                    support_mask = self._binary_open(support_mask, self.support_opening_kernel)
                if self.support_dilation > 1:
                    padding = self.support_dilation // 2
                    support_mask = F.max_pool2d(
                        support_mask,
                        kernel_size=self.support_dilation,
                        stride=1,
                        padding=padding,
                    )
                support_gate = support_gate * support_mask
            # Detach the runtime support gate so the branch cannot optimize the
            # support measure itself as an easy workaround for blank suppression.
            objectness_blob = objectness_blob * support_gate.detach()
        stroke_gate = alpha_map * objectness_blob * (0.70 + 0.30 * density.clamp(0.0, 2.0))
        stroke_scalar = stroke_gate * stroke_scalar * (1.0 + 0.15 * curvature_map)

        stroke_features = torch.cat(
            [
                carrier * stroke_scalar,
                stroke_scalar,
                density,
                direction_x,
                direction_y,
            ],
            dim=1,
        )
        residual = self.post_mix(stroke_features)
        residual = residual - F.avg_pool2d(residual, kernel_size=3, stride=1, padding=1)
        residual = residual * objectness_blob
        return residual


__all__ = ["StrokeFieldPseudoRenderer"]
