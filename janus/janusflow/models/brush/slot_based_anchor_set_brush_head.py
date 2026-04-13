"""Slot-based anchor-set brush head for explicit primitive-object experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .slot_based_anchor_set_renderer import SlotBasedAnchorSetRenderer


@dataclass
class SlotBasedAnchorSetOutput:
    """Most recent slot-based primitive predictions.

    Shapes:
    - `theta/length/width/alpha/occupancy_logits`: `[B, 1, H, W]`
    - `prototype_logits`: `[B, P, H, W]`
    - `anchor_map/support_mask/exclusion_map`: `[B, 1, H, W]`
    - `slot_valid_logits/slot_valid_probs/slot_support_mass/slot_exclusion_mass`: `[B, K]`
    - `slot_centers_xy`: `[B, K, 2]`
    - `slot_component_ids`: `[B, K]`
    - `slot_component_conflict_score`: `[B]`
    """

    theta: torch.Tensor
    length: torch.Tensor
    width: torch.Tensor
    alpha: torch.Tensor
    occupancy_logits: torch.Tensor
    prototype_logits: torch.Tensor
    anchor_map: torch.Tensor
    support_mask: torch.Tensor
    exclusion_map: torch.Tensor
    slot_valid_logits: torch.Tensor
    slot_valid_probs: torch.Tensor
    slot_centers_xy: torch.Tensor
    slot_support_mass: torch.Tensor
    slot_exclusion_mass: torch.Tensor
    slot_component_ids: torch.Tensor
    slot_component_conflict_score: torch.Tensor

    @property
    def objectness(self) -> torch.Tensor:
        """Compatibility alias for legacy stroke-field callers."""

        return self.occupancy_logits


class SlotBasedAnchorSetBrushHead(nn.Module):
    """Predict a small explicit set of primitive anchors and render them locally."""

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
        residual_scale: float = 0.50,
        dropout: float = 0.02,
    ) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.dropout = nn.Dropout2d(float(dropout))
        self.renderer = SlotBasedAnchorSetRenderer(
            feature_channels=feature_channels,
            hidden_channels=hidden_channels,
            slot_count=slot_count,
            prototype_count=prototype_count,
            render_channels=render_channels,
            support_threshold=support_threshold,
            support_mix=support_mix,
            support_opening_kernel=support_opening_kernel,
            support_dilation=support_dilation,
            min_component_area=min_component_area,
            large_component_area=large_component_area,
            max_anchors_per_component_small=max_anchors_per_component_small,
            max_anchors_per_component_large=max_anchors_per_component_large,
            subject_safe_exclusion_enabled=subject_safe_exclusion_enabled,
            exclusion_logit_bias=exclusion_logit_bias,
            exclusion_upper_center_weight=exclusion_upper_center_weight,
            exclusion_dilation=exclusion_dilation,
            valid_anchor_threshold=valid_anchor_threshold,
        )
        self._last_output: SlotBasedAnchorSetOutput | None = None

    def forward(
        self,
        feature_map: torch.Tensor,
        *,
        subject_exclusion_hints: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply slot-rendered residuals to the decoder feature map."""

        dropped_feature_map = self.dropout(feature_map)
        (
            rendered,
            theta,
            length,
            width,
            alpha,
            occupancy_logits,
            prototype_logits,
            anchor_map,
            exclusion_map,
        ) = self.renderer(
            dropped_feature_map,
            subject_exclusion_hints=subject_exclusion_hints,
        )
        aux = self.renderer.latest_aux()
        support_mask = (
            aux.support_mask
            if aux is not None
            else occupancy_logits.new_zeros(occupancy_logits.shape)
        )
        slot_valid_logits = (
            aux.slot_valid_logits
            if aux is not None
            else occupancy_logits.new_zeros((occupancy_logits.shape[0], self.renderer.slot_count))
        )
        slot_valid_probs = (
            aux.slot_valid_probs
            if aux is not None
            else occupancy_logits.new_zeros((occupancy_logits.shape[0], self.renderer.slot_count))
        )
        slot_centers_xy = (
            aux.slot_centers_xy
            if aux is not None
            else occupancy_logits.new_full((occupancy_logits.shape[0], self.renderer.slot_count, 2), -1.0)
        )
        slot_support_mass = (
            aux.slot_support_mass
            if aux is not None
            else occupancy_logits.new_zeros((occupancy_logits.shape[0], self.renderer.slot_count))
        )
        slot_exclusion_mass = (
            aux.slot_exclusion_mass
            if aux is not None
            else occupancy_logits.new_zeros((occupancy_logits.shape[0], self.renderer.slot_count))
        )
        slot_component_ids = (
            aux.slot_component_ids
            if aux is not None
            else torch.full(
                (occupancy_logits.shape[0], self.renderer.slot_count),
                fill_value=-1,
                device=occupancy_logits.device,
                dtype=torch.long,
            )
        )
        slot_component_conflict_score = (
            aux.slot_component_conflict_score
            if aux is not None
            else occupancy_logits.new_zeros((occupancy_logits.shape[0],))
        )
        self._last_output = SlotBasedAnchorSetOutput(
            theta=theta,
            length=length,
            width=width,
            alpha=alpha,
            occupancy_logits=occupancy_logits,
            prototype_logits=prototype_logits,
            anchor_map=anchor_map,
            support_mask=support_mask,
            exclusion_map=exclusion_map,
            slot_valid_logits=slot_valid_logits,
            slot_valid_probs=slot_valid_probs,
            slot_centers_xy=slot_centers_xy,
            slot_support_mass=slot_support_mass,
            slot_exclusion_mass=slot_exclusion_mass,
            slot_component_ids=slot_component_ids,
            slot_component_conflict_score=slot_component_conflict_score,
        )
        return feature_map + rendered * self.residual_scale

    def latest_output(self) -> SlotBasedAnchorSetOutput | None:
        """Return cached slot-based predictions from the latest forward pass."""

        return self._last_output

    def prototype_anchor_metadata(self) -> dict[str, torch.Tensor]:
        """Return prototype anchor metadata for proxy-supervision helpers."""

        return {
            "orientation": self.renderer.orientation_anchors,
            "length": self.renderer.length_anchors,
            "width": self.renderer.width_anchors,
        }

    def basis_usage_entropy_loss(self) -> torch.Tensor | None:
        """Encourage sharper slot-level prototype routing."""

        if self._last_output is None:
            return None
        probs = torch.softmax(self._last_output.prototype_logits, dim=1)
        entropy = -(probs * probs.clamp_min(1.0e-6).log()).sum(dim=1)
        return entropy.mean()

    def gate_l1_loss(self) -> torch.Tensor | None:
        """Keep the runtime regularization interface uniform across brush heads."""

        return None

    def gate_tv_loss(self) -> torch.Tensor | None:
        """Keep the runtime regularization interface uniform across brush heads."""

        return None


__all__ = ["SlotBasedAnchorSetBrushHead", "SlotBasedAnchorSetOutput"]
