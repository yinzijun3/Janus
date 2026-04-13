"""Support-locked primitive renderer for sparse stroke-object experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PrimitiveRendererAux:
    """Cached renderer-side maps for supervision and audit.

    Shapes:
    - `anchor_map`: `[B, 1, H, W]`
    - `support_mask`: `[B, 1, H, W]`
    - `occupancy_scores`: `[B, 1, H, W]`
    """

    anchor_map: torch.Tensor
    support_mask: torch.Tensor
    occupancy_scores: torch.Tensor
    small_component_mask: torch.Tensor
    exclusion_map: torch.Tensor
    component_overflow_score: torch.Tensor


class SupportLockedPrimitiveRenderer(nn.Module):
    """Render sparse elongated primitives anchored to support-backed peaks.

    Unlike the earlier semi-implicit stroke-field renderer, this module first
    selects a sparse set of anchors with NMS + top-k, then stamps one elongated
    primitive per anchor. Primitive footprints are clipped to support-backed
    regions so blank areas cannot be filled by a dense objectness blob.
    """

    def __init__(
        self,
        *,
        feature_channels: int = 768,
        prototype_count: int = 16,
        render_channels: int = 12,
        kernel_size: int = 13,
        top_k_anchors: int = 24,
        support_threshold: float = 0.28,
        support_mix: float = 0.80,
        support_opening_kernel: int = 3,
        support_dilation: int = 3,
        nms_kernel: int = 3,
        component_aware_enabled: bool = False,
        min_component_area: int = 4,
        large_component_area: int = 20,
        max_anchors_per_component_small: int = 1,
        max_anchors_per_component_large: int = 2,
        subject_safe_exclusion_enabled: bool = False,
        exclusion_logit_bias: float = -6.0,
        exclusion_upper_center_weight: float = 1.0,
        exclusion_dilation: int = 3,
    ) -> None:
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.prototype_count = int(prototype_count)
        self.render_channels = int(render_channels)
        self.kernel_size = int(kernel_size)
        self.top_k_anchors = int(top_k_anchors)
        self.support_threshold = float(support_threshold)
        self.support_mix = float(support_mix)
        opening_kernel = max(int(support_opening_kernel), 0)
        if opening_kernel > 0 and opening_kernel % 2 == 0:
            opening_kernel += 1
        self.support_opening_kernel = opening_kernel
        dilation = max(int(support_dilation), 1)
        self.support_dilation = dilation if dilation % 2 == 1 else dilation + 1
        nms_kernel = max(int(nms_kernel), 1)
        self.nms_kernel = nms_kernel if nms_kernel % 2 == 1 else nms_kernel + 1
        self.component_aware_enabled = bool(component_aware_enabled)
        self.min_component_area = max(int(min_component_area), 1)
        self.large_component_area = max(int(large_component_area), self.min_component_area)
        self.max_anchors_per_component_small = max(int(max_anchors_per_component_small), 0)
        self.max_anchors_per_component_large = max(
            int(max_anchors_per_component_large),
            self.max_anchors_per_component_small,
        )
        self.subject_safe_exclusion_enabled = bool(subject_safe_exclusion_enabled)
        self.exclusion_logit_bias = float(exclusion_logit_bias)
        self.exclusion_upper_center_weight = float(exclusion_upper_center_weight)
        exclusion_dilation = max(int(exclusion_dilation), 1)
        self.exclusion_dilation = exclusion_dilation if exclusion_dilation % 2 == 1 else exclusion_dilation + 1

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
        self.prototype_embedding = nn.Parameter(
            torch.randn(self.prototype_count, self.render_channels) * 0.02
        )
        orientation_anchors, length_anchors, width_anchors = self._build_anchor_metadata()
        self.register_buffer("orientation_anchors", orientation_anchors, persistent=False)
        self.register_buffer("length_anchors", length_anchors, persistent=False)
        self.register_buffer("width_anchors", width_anchors, persistent=False)
        axis = torch.linspace(-1.0, 1.0, self.kernel_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("grid_x", xx, persistent=False)
        self.register_buffer("grid_y", yy, persistent=False)
        self._last_aux: PrimitiveRendererAux | None = None

    def _build_anchor_metadata(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build prototype anchor metadata for proxy-target lookup."""

        orientation = torch.linspace(0.0, math.pi, self.prototype_count + 1, dtype=torch.float32)[:-1]
        length_levels = torch.tensor([0.40, 0.70, 1.00, 1.25], dtype=torch.float32)
        width_levels = torch.tensor([0.12, 0.18, 0.24, 0.30], dtype=torch.float32)
        length = torch.stack(
            [length_levels[index % len(length_levels)] for index in range(self.prototype_count)]
        )
        width = torch.stack(
            [width_levels[(index // len(length_levels)) % len(width_levels)] for index in range(self.prototype_count)]
        )
        return orientation, length, width

    @staticmethod
    def _normalize_map(tensor: torch.Tensor) -> torch.Tensor:
        """Normalize BCHW maps to `[0, 1]` per sample."""

        min_value = tensor.amin(dim=(-2, -1), keepdim=True)
        max_value = tensor.amax(dim=(-2, -1), keepdim=True)
        return (tensor - min_value) / (max_value - min_value).clamp_min(1.0e-6)

    @staticmethod
    def _binary_open(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Apply binary opening to prune isolated tiny supports."""

        if kernel_size <= 1:
            return mask
        padding = kernel_size // 2
        eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=padding)
        opened = F.max_pool2d(eroded, kernel_size=kernel_size, stride=1, padding=padding)
        return opened.clamp_(0.0, 1.0)

    def _build_support_mask(self, carrier_hp: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """Convert carrier evidence into a binary support mask."""

        carrier_support = self._normalize_map(
            F.avg_pool2d(carrier_hp.abs(), kernel_size=3, stride=1, padding=1)
        )
        density_support = self._normalize_map(density)
        support_evidence = self.support_mix * carrier_support + (1.0 - self.support_mix) * density_support
        support_mask = (support_evidence >= self.support_threshold).to(dtype=carrier_hp.dtype)
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
        return support_mask.clamp_(0.0, 1.0)

    def _select_sparse_anchors(
        self,
        occupancy_scores: torch.Tensor,
        support_mask: torch.Tensor,
        *,
        exclusion_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run NMS + top-k anchor selection over the occupancy map."""

        scores = occupancy_scores * support_mask
        if exclusion_map is not None:
            scores = scores * (1.0 - exclusion_map)
        local_max = F.max_pool2d(scores, kernel_size=self.nms_kernel, stride=1, padding=self.nms_kernel // 2)
        peak_mask = (scores >= local_max) & (scores > 0.05)
        peak_scores = scores * peak_mask.to(dtype=scores.dtype)
        if not self.component_aware_enabled:
            flat_scores = peak_scores.flatten(start_dim=1)
            topk_values, topk_indices = torch.topk(
                flat_scores,
                k=min(self.top_k_anchors, flat_scores.shape[1]),
                dim=1,
            )
            anchor_map = torch.zeros_like(flat_scores)
            anchor_map.scatter_(1, topk_indices, topk_values)
            anchor_map = anchor_map.view_as(scores)
            small_component_mask = torch.zeros_like(scores)
            component_overflow_score = torch.zeros(scores.shape[0], device=scores.device, dtype=scores.dtype)
            return anchor_map, topk_values, topk_indices, small_component_mask, component_overflow_score
        return self._select_sparse_anchors_component_aware(peak_scores, support_mask)

    @staticmethod
    def _connected_components(mask_2d: torch.Tensor) -> list[list[tuple[int, int]]]:
        """Extract 4-neighbor connected components from a binary 2D mask."""

        mask = mask_2d.detach().to(dtype=torch.bool, device="cpu").numpy()
        height, width = mask.shape
        visited = [[False] * width for _ in range(height)]
        components: list[list[tuple[int, int]]] = []
        for y in range(height):
            for x in range(width):
                if not mask[y, x] or visited[y][x]:
                    continue
                stack = [(y, x)]
                visited[y][x] = True
                component: list[tuple[int, int]] = []
                while stack:
                    cy, cx = stack.pop()
                    component.append((cy, cx))
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny][nx]:
                            visited[ny][nx] = True
                            stack.append((ny, nx))
                components.append(component)
        return components

    def _select_sparse_anchors_component_aware(
        self,
        peak_scores: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select sparse anchors with connected-support capacity constraints."""

        batch_size, _, height, width = peak_scores.shape
        flat_size = height * width
        anchor_map = torch.zeros((batch_size, 1, height, width), device=peak_scores.device, dtype=peak_scores.dtype)
        topk_values = torch.zeros(
            (batch_size, min(self.top_k_anchors, flat_size)),
            device=peak_scores.device,
            dtype=peak_scores.dtype,
        )
        topk_indices = torch.zeros(
            (batch_size, min(self.top_k_anchors, flat_size)),
            device=peak_scores.device,
            dtype=torch.long,
        )
        small_component_mask = torch.zeros_like(anchor_map)
        component_overflow_score = torch.zeros(batch_size, device=peak_scores.device, dtype=peak_scores.dtype)

        for batch_index in range(batch_size):
            components = self._connected_components(support_mask[batch_index, 0] > 0.5)
            selected: list[tuple[float, int]] = []
            for component in components:
                area = len(component)
                comp_mask = torch.zeros((height, width), device=peak_scores.device, dtype=peak_scores.dtype)
                ys = []
                xs = []
                for y, x in component:
                    ys.append(y)
                    xs.append(x)
                    comp_mask[y, x] = 1.0
                if area < self.min_component_area:
                    small_component_mask[batch_index, 0] = torch.maximum(
                        small_component_mask[batch_index, 0],
                        comp_mask,
                    )
                    component_overflow_score[batch_index] += (
                        peak_scores[batch_index, 0] * comp_mask
                    ).sum()
                    continue
                allowed = (
                    self.max_anchors_per_component_large
                    if area >= self.large_component_area
                    else self.max_anchors_per_component_small
                )
                comp_scores = (peak_scores[batch_index, 0] * comp_mask).flatten()
                positive_scores = comp_scores[comp_scores > 0.0]
                if positive_scores.numel() > allowed:
                    sorted_scores, _ = torch.sort(positive_scores, descending=True)
                    component_overflow_score[batch_index] += sorted_scores[allowed:].sum()
                if allowed <= 0:
                    continue
                values, indices = torch.topk(comp_scores, k=min(allowed, comp_scores.numel()))
                for value, index in zip(values.tolist(), indices.tolist()):
                    if value <= 0.0:
                        continue
                    selected.append((float(value), int(index)))
            selected.sort(key=lambda item: item[0], reverse=True)
            selected = selected[: self.top_k_anchors]
            for slot, (value, flat_index) in enumerate(selected):
                y = flat_index // width
                x = flat_index % width
                anchor_map[batch_index, 0, y, x] = value
                topk_values[batch_index, slot] = value
                topk_indices[batch_index, slot] = flat_index
        return anchor_map, topk_values, topk_indices, small_component_mask, component_overflow_score

    def _primitive_kernel(
        self,
        *,
        theta_value: torch.Tensor,
        length_value: torch.Tensor,
        width_value: torch.Tensor,
    ) -> torch.Tensor:
        """Build one elongated primitive kernel clipped to local support."""

        x_rot = self.grid_x * torch.cos(theta_value) + self.grid_y * torch.sin(theta_value)
        y_rot = -self.grid_x * torch.sin(theta_value) + self.grid_y * torch.cos(theta_value)
        length_norm = length_value.clamp_min(1.0e-3)
        width_norm = width_value.clamp_min(1.0e-3)
        taper = (1.0 - torch.clamp(x_rot.abs() / length_norm, min=0.0, max=1.0)) ** 1.5
        body = torch.exp(-0.5 * ((x_rot / length_norm) ** 2 + (y_rot / width_norm) ** 2))
        ridge = taper * body * torch.clamp(1.0 - (y_rot / width_norm) ** 2, min=0.0)
        return ridge

    def _build_subject_exclusion_map(
        self,
        feature_map: torch.Tensor,
        subject_exclusion_hints: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build an upper-center portrait-safe exclusion map from feature energy."""

        batch_size, _, height, width = feature_map.shape
        exclusion = feature_map.new_zeros((batch_size, 1, height, width))
        if not self.subject_safe_exclusion_enabled or subject_exclusion_hints is None:
            return exclusion
        hints = subject_exclusion_hints.to(device=feature_map.device, dtype=feature_map.dtype).view(batch_size, 1, 1, 1)
        if float(hints.max().item()) <= 0.0:
            return exclusion
        feature_energy = feature_map.abs().mean(dim=1, keepdim=True)
        low_frequency = F.avg_pool2d(feature_energy, kernel_size=5, stride=1, padding=2)
        saliency = self._normalize_map(low_frequency)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=feature_map.device, dtype=feature_map.dtype),
            torch.linspace(-1.0, 1.0, width, device=feature_map.device, dtype=feature_map.dtype),
            indexing="ij",
        )
        center_x = xx.view(1, 1, height, width)
        center_y = yy.view(1, 1, height, width)
        ellipse = torch.exp(
            -0.5 * ((center_x / 0.42) ** 2 + ((center_y + 0.30) / 0.34) ** 2)
        )
        intersection = ((ellipse * self.exclusion_upper_center_weight) > 0.30).to(feature_map.dtype)
        intersection = intersection * (saliency > 0.40).to(feature_map.dtype)
        if self.exclusion_dilation > 1:
            padding = self.exclusion_dilation // 2
            intersection = F.max_pool2d(
                intersection,
                kernel_size=self.exclusion_dilation,
                stride=1,
                padding=padding,
            )
        return (intersection * hints).clamp_(0.0, 1.0)

    def forward(
        self,
        feature_map: torch.Tensor,
        *,
        theta: torch.Tensor,
        length: torch.Tensor,
        width: torch.Tensor,
        alpha: torch.Tensor,
        occupancy_logits: torch.Tensor,
        prototype_logits: torch.Tensor,
        subject_exclusion_hints: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Render a sparse primitive residual with the same BCHW shape as `feature_map`."""

        # `feature_map`: [B, C, H, W]
        batch_size, _, height, width_size = feature_map.shape
        carrier = self.feature_project(feature_map)  # [B, R, H, W]
        carrier_mean = carrier.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        carrier_hp = carrier_mean - F.avg_pool2d(carrier_mean, kernel_size=3, stride=1, padding=1)
        density = carrier.abs().mean(dim=1, keepdim=True)
        density = density / density.mean(dim=(2, 3), keepdim=True).clamp_min(1.0e-6)
        support_mask = self._build_support_mask(carrier_hp, density)
        exclusion_map = self._build_subject_exclusion_map(feature_map, subject_exclusion_hints)
        effective_logits = occupancy_logits + exclusion_map * self.exclusion_logit_bias
        occupancy_scores = torch.sigmoid(effective_logits)
        available_support = support_mask * (1.0 - exclusion_map)
        (
            anchor_map,
            topk_values,
            topk_indices,
            small_component_mask,
            component_overflow_score,
        ) = self._select_sparse_anchors(
            occupancy_scores,
            available_support,
            exclusion_map=exclusion_map,
        )

        rendered = torch.zeros(
            batch_size,
            self.render_channels,
            height,
            width_size,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        padding = self.kernel_size // 2
        theta_map = torch.tanh(theta) * math.pi
        length_map = torch.sigmoid(length) * 0.95 + 0.30
        width_map = torch.sigmoid(width) * 0.24 + 0.08
        alpha_map = torch.sigmoid(alpha)
        prototype_ids = torch.argmax(prototype_logits, dim=1)

        for batch_index in range(batch_size):
            for slot in range(topk_indices.shape[1]):
                flat_index = int(topk_indices[batch_index, slot].item())
                score = topk_values[batch_index, slot]
                if float(score.item()) <= 0.0:
                    continue
                anchor_y = flat_index // width_size
                anchor_x = flat_index % width_size
                if available_support[batch_index, 0, anchor_y, anchor_x] <= 0.0:
                    continue
                if exclusion_map[batch_index, 0, anchor_y, anchor_x] > 0.0:
                    continue
                y0 = max(anchor_y - padding, 0)
                y1 = min(anchor_y + padding + 1, height)
                x0 = max(anchor_x - padding, 0)
                x1 = min(anchor_x + padding + 1, width_size)
                ky0 = padding - (anchor_y - y0)
                ky1 = ky0 + (y1 - y0)
                kx0 = padding - (anchor_x - x0)
                kx1 = kx0 + (x1 - x0)
                support_patch = available_support[batch_index, 0, y0:y1, x0:x1]
                primitive = self._primitive_kernel(
                    theta_value=theta_map[batch_index, 0, anchor_y, anchor_x],
                    length_value=length_map[batch_index, 0, anchor_y, anchor_x],
                    width_value=width_map[batch_index, 0, anchor_y, anchor_x],
                )[ky0:ky1, kx0:kx1] * support_patch
                prototype_id = int(prototype_ids[batch_index, anchor_y, anchor_x].item())
                proto_vec = self.prototype_embedding[prototype_id].to(dtype=feature_map.dtype)
                carrier_vec = carrier[batch_index, :, anchor_y, anchor_x]
                channel_vec = carrier_vec * (0.65 + 0.35 * torch.tanh(proto_vec)) + 0.20 * proto_vec
                amplitude = alpha_map[batch_index, 0, anchor_y, anchor_x] * score
                rendered[batch_index, :, y0:y1, x0:x1] += channel_vec[:, None, None] * primitive[None] * amplitude

        aux_map = torch.cat(
            [
                rendered,
                support_mask,
                anchor_map,
                density,
                self._normalize_map(carrier_hp.abs()),
            ],
            dim=1,
        )
        residual = self.post_mix(aux_map)
        residual = residual - F.avg_pool2d(residual, kernel_size=3, stride=1, padding=1)
        residual = residual * available_support
        self._last_aux = PrimitiveRendererAux(
            anchor_map=anchor_map.detach(),
            support_mask=available_support.detach(),
            occupancy_scores=occupancy_scores.detach(),
            small_component_mask=small_component_mask.detach(),
            exclusion_map=exclusion_map.detach(),
            component_overflow_score=component_overflow_score.detach(),
        )
        return residual

    def latest_aux(self) -> PrimitiveRendererAux | None:
        """Return the latest sparse-anchor supervision maps."""

        return self._last_aux


__all__ = ["PrimitiveRendererAux", "SupportLockedPrimitiveRenderer"]
