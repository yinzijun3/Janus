"""Slot-based anchor-set renderer for explicit primitive-object experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .support_locked_primitive_renderer import SupportLockedPrimitiveRenderer


@dataclass
class SlotRendererAux:
    """Cached slot-level and dense audit tensors.

    Shapes:
    - `anchor_map`: `[B, 1, H, W]`
    - `support_mask`: `[B, 1, H, W]`
    - `occupancy_scores`: `[B, 1, H, W]`
    - `slot_valid_logits`: `[B, K]`
    - `slot_valid_probs`: `[B, K]`
    - `slot_centers_xy`: `[B, K, 2]` in pixel coordinates `(x, y)`
    - `slot_support_mass`: `[B, K]`
    - `slot_exclusion_mass`: `[B, K]`
    - `slot_component_ids`: `[B, K]`
    - `slot_component_conflict_score`: `[B]`
    """

    anchor_map: torch.Tensor
    support_mask: torch.Tensor
    occupancy_scores: torch.Tensor
    exclusion_map: torch.Tensor
    slot_valid_logits: torch.Tensor
    slot_valid_probs: torch.Tensor
    slot_centers_xy: torch.Tensor
    slot_support_mass: torch.Tensor
    slot_exclusion_mass: torch.Tensor
    slot_component_ids: torch.Tensor
    slot_component_conflict_score: torch.Tensor


class SlotBasedAnchorSetRenderer(SupportLockedPrimitiveRenderer):
    """Render a small set of explicit stroke anchors instead of dense occupancy.

    This branch keeps the same primitive-stamping backend as the current
    support-locked primitive family, but it replaces the dense per-pixel
    occupancy map with a fixed number of slots that predict:

    - whether a stroke anchor is valid
    - where the anchor lives
    - which primitive prototype it should use
    - which support component owns the anchor
    """

    def __init__(
        self,
        *,
        feature_channels: int = 768,
        hidden_channels: int = 512,
        slot_count: int = 16,
        prototype_count: int = 16,
        render_channels: int = 12,
        support_threshold: float = 0.28,
        support_mix: float = 0.80,
        support_opening_kernel: int = 3,
        support_dilation: int = 3,
        min_component_area: int = 4,
        large_component_area: int = 20,
        max_anchors_per_component_small: int = 1,
        max_anchors_per_component_large: int = 2,
        subject_safe_exclusion_enabled: bool = True,
        exclusion_logit_bias: float = -6.0,
        exclusion_upper_center_weight: float = 1.0,
        exclusion_dilation: int = 3,
        valid_anchor_threshold: float = 0.35,
    ) -> None:
        super().__init__(
            feature_channels=feature_channels,
            prototype_count=prototype_count,
            render_channels=render_channels,
            kernel_size=13,
            top_k_anchors=slot_count,
            support_threshold=support_threshold,
            support_mix=support_mix,
            support_opening_kernel=support_opening_kernel,
            support_dilation=support_dilation,
            component_aware_enabled=True,
            min_component_area=min_component_area,
            large_component_area=large_component_area,
            max_anchors_per_component_small=max_anchors_per_component_small,
            max_anchors_per_component_large=max_anchors_per_component_large,
            subject_safe_exclusion_enabled=subject_safe_exclusion_enabled,
            exclusion_logit_bias=exclusion_logit_bias,
            exclusion_upper_center_weight=exclusion_upper_center_weight,
            exclusion_dilation=exclusion_dilation,
        )
        self.hidden_channels = int(hidden_channels)
        self.slot_count = int(slot_count)
        self.valid_anchor_threshold = float(valid_anchor_threshold)

        self.slot_stem = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=feature_channels),
            nn.Conv2d(feature_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.slot_key = nn.Conv2d(self.hidden_channels, render_channels, kernel_size=1)
        self.slot_queries = nn.Parameter(torch.randn(self.slot_count, render_channels) * 0.02)
        self.slot_valid = nn.Linear(render_channels, 1)
        self.slot_theta = nn.Linear(render_channels, 1)
        self.slot_length = nn.Linear(render_channels, 1)
        self.slot_width = nn.Linear(render_channels, 1)
        self.slot_alpha = nn.Linear(render_channels, 1)
        self.slot_prototype = nn.Linear(render_channels, prototype_count)
        self._last_aux: SlotRendererAux | None = None

    @staticmethod
    def _safe_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute a numerically safe masked softmax over the last dimension."""

        masked_logits = logits.masked_fill(~mask, -1.0e4)
        weights = torch.softmax(masked_logits, dim=-1)
        has_support = mask.any(dim=-1, keepdim=True)
        return torch.where(has_support, weights, torch.zeros_like(weights))

    @staticmethod
    def _normalized_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Build full-resolution `[-1, 1]` coordinate grids."""

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        return xx, yy

    def _collect_component_masks(
        self,
        support_mask: torch.Tensor,
    ) -> tuple[list[list[torch.Tensor]], torch.Tensor]:
        """Build per-sample connected-component masks and id maps."""

        batch_size, _, height, width = support_mask.shape
        component_masks: list[list[torch.Tensor]] = []
        component_ids = torch.full(
            (batch_size, 1, height, width),
            fill_value=-1,
            device=support_mask.device,
            dtype=torch.long,
        )
        for batch_index in range(batch_size):
            components = self._connected_components(support_mask[batch_index, 0] > 0.5)
            masks_for_batch: list[torch.Tensor] = []
            component_index = 0
            for component in components:
                area = len(component)
                if area < self.min_component_area:
                    continue
                mask = torch.zeros((height, width), device=support_mask.device, dtype=support_mask.dtype)
                for y, x in component:
                    mask[y, x] = 1.0
                    component_ids[batch_index, 0, y, x] = component_index
                masks_for_batch.append(mask)
                component_index += 1
            component_masks.append(masks_for_batch)
        return component_masks, component_ids

    def _render_slot_primitive(
        self,
        *,
        center_x: torch.Tensor,
        center_y: torch.Tensor,
        theta_value: torch.Tensor,
        length_value: torch.Tensor,
        width_value: torch.Tensor,
        amplitude: torch.Tensor,
        component_mask: torch.Tensor,
        prototype_vector: torch.Tensor,
        full_grid_x: torch.Tensor,
        full_grid_y: torch.Tensor,
    ) -> torch.Tensor:
        """Render one full-image elongated primitive clipped to one support component."""

        x_shift = full_grid_x - center_x
        y_shift = full_grid_y - center_y
        x_rot = x_shift * torch.cos(theta_value) + y_shift * torch.sin(theta_value)
        y_rot = -x_shift * torch.sin(theta_value) + y_shift * torch.cos(theta_value)
        taper = (1.0 - torch.clamp(x_rot.abs() / length_value.clamp_min(1.0e-3), min=0.0, max=1.0)) ** 1.5
        body = torch.exp(
            -0.5
            * (
                (x_rot / length_value.clamp_min(1.0e-3)) ** 2
                + (y_rot / width_value.clamp_min(1.0e-3)) ** 2
            )
        )
        ridge = taper * body * torch.clamp(
            1.0 - (y_rot / width_value.clamp_min(1.0e-3)) ** 2,
            min=0.0,
        )
        primitive = ridge * component_mask * amplitude
        return prototype_vector[:, None, None] * primitive[None]

    def forward(
        self,
        feature_map: torch.Tensor,
        *,
        subject_exclusion_hints: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Render slot-based sparse primitives and return dense audit maps.

        Returns:
        - rendered residual: `[B, C, H, W]`
        - theta/length/width/alpha/occupancy maps: `[B, 1, H, W]`
        - prototype logits: `[B, P, H, W]`
        - anchor map: `[B, 1, H, W]`
        """

        batch_size, _, height, width = feature_map.shape
        carrier = self.feature_project(feature_map)  # [B, R, H, W]
        carrier_mean = carrier.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        carrier_hp = carrier_mean - F.avg_pool2d(carrier_mean, kernel_size=3, stride=1, padding=1)
        density = carrier.abs().mean(dim=1, keepdim=True)
        density = density / density.mean(dim=(2, 3), keepdim=True).clamp_min(1.0e-6)
        support_mask = self._build_support_mask(carrier_hp.abs(), density)
        exclusion_map = self._build_subject_exclusion_map(feature_map, subject_exclusion_hints)
        available_support = support_mask * (1.0 - exclusion_map)
        component_masks, component_id_map = self._collect_component_masks(available_support)

        hidden = self.slot_stem(feature_map)  # [B, Hc, H, W]
        key_map = self.slot_key(hidden)  # [B, R, H, W]
        flat_keys = key_map.flatten(start_dim=2).transpose(1, 2)  # [B, HW, R]
        query = self.slot_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, R]
        attn_logits = torch.einsum("bkd,bnd->bkn", query, flat_keys) / math.sqrt(float(self.render_channels))
        support_flat = available_support.flatten(start_dim=2).squeeze(1) > 0.5  # [B, HW]
        attn_weights = self._safe_softmax(attn_logits, support_flat.unsqueeze(1).expand(-1, self.slot_count, -1))
        slot_features = torch.einsum("bkn,bnd->bkd", attn_weights, flat_keys)  # [B, K, R]

        valid_logits = self.slot_valid(slot_features).squeeze(-1)  # [B, K]
        theta_slots = self.slot_theta(slot_features).squeeze(-1)  # [B, K]
        length_slots = self.slot_length(slot_features).squeeze(-1)  # [B, K]
        width_slots = self.slot_width(slot_features).squeeze(-1)  # [B, K]
        alpha_slots = self.slot_alpha(slot_features).squeeze(-1)  # [B, K]
        prototype_logits_slots = self.slot_prototype(slot_features)  # [B, K, P]
        valid_probs = torch.sigmoid(valid_logits)  # [B, K]

        rendered = torch.zeros(
            batch_size,
            self.render_channels,
            height,
            width,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        anchor_map = torch.zeros((batch_size, 1, height, width), device=feature_map.device, dtype=feature_map.dtype)
        occupancy_map = torch.zeros_like(anchor_map)
        theta_map = torch.zeros_like(anchor_map)
        length_map = torch.zeros_like(anchor_map)
        width_map = torch.zeros_like(anchor_map)
        alpha_map = torch.zeros_like(anchor_map)
        prototype_map = torch.zeros(
            (batch_size, self.prototype_count, height, width),
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        slot_centers_xy = torch.full(
            (batch_size, self.slot_count, 2),
            fill_value=-1.0,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        slot_support_mass = torch.zeros((batch_size, self.slot_count), device=feature_map.device, dtype=feature_map.dtype)
        slot_exclusion_mass = torch.zeros_like(slot_support_mass)
        slot_component_ids = torch.full(
            (batch_size, self.slot_count),
            fill_value=-1,
            device=feature_map.device,
            dtype=torch.long,
        )
        slot_component_conflict_values = [
            feature_map.new_zeros(())
            for _ in range(batch_size)
        ]
        full_grid_x, full_grid_y = self._normalized_grid(height, width, feature_map.device, feature_map.dtype)

        for batch_index in range(batch_size):
            component_slot_counts: dict[int, int] = {}
            flat_exclusion = exclusion_map[batch_index, 0].flatten()
            for slot_index in range(self.slot_count):
                valid_prob = valid_probs[batch_index, slot_index]
                attention = attn_weights[batch_index, slot_index].view(height, width)
                slot_exclusion_mass[batch_index, slot_index] = (attention.flatten() * flat_exclusion).sum()
                if float(valid_prob.item()) < self.valid_anchor_threshold:
                    continue
                masks_for_batch = component_masks[batch_index]
                if not masks_for_batch:
                    continue
                masses = torch.stack([(attention * mask).sum() for mask in masks_for_batch])
                component_mass, component_choice = torch.max(masses, dim=0)
                slot_support_mass[batch_index, slot_index] = component_mass
                if float(component_mass.item()) <= 1.0e-6:
                    continue
                component_index = int(component_choice.item())
                component_mask = masks_for_batch[component_index]
                component_count = component_slot_counts.get(component_index, 0)
                component_area = int(component_mask.sum().item())
                allowed = (
                    self.max_anchors_per_component_large
                    if component_area >= self.large_component_area
                    else self.max_anchors_per_component_small
                )
                if component_count >= allowed:
                    slot_component_conflict_values[batch_index] = (
                        slot_component_conflict_values[batch_index] + valid_prob
                    )
                    continue
                component_slot_counts[component_index] = component_count + 1
                localized_attention = attention * component_mask
                localized_attention = localized_attention / localized_attention.sum().clamp_min(1.0e-6)
                center_y_index = (localized_attention * torch.arange(height, device=feature_map.device, dtype=feature_map.dtype)[:, None]).sum()
                center_x_index = (localized_attention * torch.arange(width, device=feature_map.device, dtype=feature_map.dtype)[None, :]).sum()
                slot_centers_xy[batch_index, slot_index, 0] = center_x_index
                slot_centers_xy[batch_index, slot_index, 1] = center_y_index
                slot_component_ids[batch_index, slot_index] = component_index

                anchor_x = int(torch.round(center_x_index).clamp(0, width - 1).item())
                anchor_y = int(torch.round(center_y_index).clamp(0, height - 1).item())
                anchor_map[batch_index, 0, anchor_y, anchor_x] = valid_prob
                occupancy_map[batch_index, 0, anchor_y, anchor_x] = valid_logits[batch_index, slot_index]
                theta_map[batch_index, 0, anchor_y, anchor_x] = theta_slots[batch_index, slot_index]
                length_map[batch_index, 0, anchor_y, anchor_x] = length_slots[batch_index, slot_index]
                width_map[batch_index, 0, anchor_y, anchor_x] = width_slots[batch_index, slot_index]
                alpha_map[batch_index, 0, anchor_y, anchor_x] = alpha_slots[batch_index, slot_index]
                prototype_map[batch_index, :, anchor_y, anchor_x] = prototype_logits_slots[batch_index, slot_index]

                theta_value = torch.tanh(theta_slots[batch_index, slot_index]) * math.pi
                length_value = torch.sigmoid(length_slots[batch_index, slot_index]) * 0.95 + 0.30
                width_value = torch.sigmoid(width_slots[batch_index, slot_index]) * 0.24 + 0.08
                alpha_value = torch.sigmoid(alpha_slots[batch_index, slot_index]) * valid_prob
                prototype_probs = torch.softmax(prototype_logits_slots[batch_index, slot_index], dim=-1)
                proto_vec = prototype_probs @ self.prototype_embedding.to(dtype=feature_map.dtype)
                carrier_vec = carrier[batch_index, :, anchor_y, anchor_x]
                prototype_vector = carrier_vec * (0.65 + 0.35 * torch.tanh(proto_vec)) + 0.20 * proto_vec
                center_x = (center_x_index / max(width - 1, 1)) * 2.0 - 1.0
                center_y = (center_y_index / max(height - 1, 1)) * 2.0 - 1.0
                rendered[batch_index] += self._render_slot_primitive(
                    center_x=center_x,
                    center_y=center_y,
                    theta_value=theta_value,
                    length_value=length_value,
                    width_value=width_value,
                    amplitude=alpha_value,
                    component_mask=component_mask,
                    prototype_vector=prototype_vector,
                    full_grid_x=full_grid_x,
                    full_grid_y=full_grid_y,
                )

        aux_map = torch.cat(
            [
                rendered,
                available_support,
                anchor_map,
                density,
                self._normalize_map(carrier_hp.abs()),
            ],
            dim=1,
        )
        residual = self.post_mix(aux_map)
        residual = residual - F.avg_pool2d(residual, kernel_size=3, stride=1, padding=1)
        residual = residual * available_support
        slot_component_conflict = torch.stack(slot_component_conflict_values, dim=0)
        self._last_aux = SlotRendererAux(
            anchor_map=anchor_map.detach(),
            support_mask=available_support.detach(),
            occupancy_scores=torch.sigmoid(occupancy_map).detach(),
            exclusion_map=exclusion_map.detach(),
            slot_valid_logits=valid_logits.detach(),
            slot_valid_probs=valid_probs.detach(),
            slot_centers_xy=slot_centers_xy.detach(),
            slot_support_mass=slot_support_mass.detach(),
            slot_exclusion_mass=slot_exclusion_mass.detach(),
            slot_component_ids=slot_component_ids.detach(),
            slot_component_conflict_score=slot_component_conflict.detach(),
        )
        return (
            residual,
            theta_map,
            length_map,
            width_map,
            alpha_map,
            occupancy_map,
            prototype_map,
            anchor_map,
            exclusion_map,
        )

    def latest_aux(self) -> SlotRendererAux | None:
        """Return cached slot-level supervision tensors."""

        return self._last_aux


__all__ = ["SlotBasedAnchorSetRenderer", "SlotRendererAux"]
