"""Sparse support-locked primitive brush head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .support_locked_primitive_renderer import SupportLockedPrimitiveRenderer


@dataclass
class SupportLockedPrimitiveOutput:
    """Most recent sparse primitive predictions.

    Shapes:
    - `theta/length/width/alpha/occupancy_logits`: `[B, 1, H, W]`
    - `prototype_logits`: `[B, P, H, W]`
    - `anchor_map/support_mask`: `[B, 1, H, W]`
    """

    theta: torch.Tensor
    length: torch.Tensor
    width: torch.Tensor
    alpha: torch.Tensor
    occupancy_logits: torch.Tensor
    prototype_logits: torch.Tensor
    anchor_map: torch.Tensor
    support_mask: torch.Tensor
    small_component_mask: torch.Tensor
    exclusion_map: torch.Tensor
    component_overflow_score: torch.Tensor

    @property
    def objectness(self) -> torch.Tensor:
        """Compatibility alias for legacy stroke-field callers."""

        return self.occupancy_logits


class SupportLockedPrimitiveBrushHead(nn.Module):
    """Predict sparse anchor fields and render support-locked primitives."""

    def __init__(
        self,
        *,
        feature_channels: int = 768,
        hidden_channels: int = 512,
        prototype_count: int = 16,
        render_channels: int = 12,
        renderer_kernel_size: int = 13,
        top_k_anchors: int = 24,
        support_threshold: float = 0.28,
        support_mix: float = 0.80,
        support_opening_kernel: int = 3,
        support_dilation: int = 3,
        component_aware_enabled: bool = False,
        min_component_area: int = 4,
        large_component_area: int = 20,
        max_anchors_per_component_small: int = 1,
        max_anchors_per_component_large: int = 2,
        subject_safe_exclusion_enabled: bool = False,
        exclusion_logit_bias: float = -6.0,
        exclusion_upper_center_weight: float = 1.0,
        exclusion_dilation: int = 3,
        residual_scale: float = 0.5,
        dropout: float = 0.02,
    ) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.stem = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=feature_channels),
            nn.Conv2d(feature_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.occupancy_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.theta_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.length_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.width_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.alpha_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.prototype_head = nn.Conv2d(hidden_channels, prototype_count, kernel_size=1)
        self.renderer = SupportLockedPrimitiveRenderer(
            feature_channels=feature_channels,
            prototype_count=prototype_count,
            render_channels=render_channels,
            kernel_size=renderer_kernel_size,
            top_k_anchors=top_k_anchors,
            support_threshold=support_threshold,
            support_mix=support_mix,
            support_opening_kernel=support_opening_kernel,
            support_dilation=support_dilation,
            component_aware_enabled=component_aware_enabled,
            min_component_area=min_component_area,
            large_component_area=large_component_area,
            max_anchors_per_component_small=max_anchors_per_component_small,
            max_anchors_per_component_large=max_anchors_per_component_large,
            subject_safe_exclusion_enabled=subject_safe_exclusion_enabled,
            exclusion_logit_bias=exclusion_logit_bias,
            exclusion_upper_center_weight=exclusion_upper_center_weight,
            exclusion_dilation=exclusion_dilation,
        )
        self._last_output: SupportLockedPrimitiveOutput | None = None

    def forward(
        self,
        feature_map: torch.Tensor,
        *,
        subject_exclusion_hints: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply sparse primitive rendering to the decoder feature map."""

        hidden = self.stem(feature_map)
        occupancy_logits = self.occupancy_head(hidden)
        theta = self.theta_head(hidden)
        length = self.length_head(hidden)
        width = self.width_head(hidden)
        alpha = self.alpha_head(hidden)
        prototype_logits = self.prototype_head(hidden)
        rendered = self.renderer(
            feature_map,
            theta=theta,
            length=length,
            width=width,
            alpha=alpha,
            occupancy_logits=occupancy_logits,
            prototype_logits=prototype_logits,
            subject_exclusion_hints=subject_exclusion_hints,
        )
        aux = self.renderer.latest_aux()
        anchor_map = aux.anchor_map if aux is not None else occupancy_logits.new_zeros(occupancy_logits.shape)
        support_mask = aux.support_mask if aux is not None else occupancy_logits.new_zeros(occupancy_logits.shape)
        small_component_mask = (
            aux.small_component_mask if aux is not None else occupancy_logits.new_zeros(occupancy_logits.shape)
        )
        exclusion_map = aux.exclusion_map if aux is not None else occupancy_logits.new_zeros(occupancy_logits.shape)
        component_overflow_score = (
            aux.component_overflow_score
            if aux is not None
            else occupancy_logits.new_zeros((occupancy_logits.shape[0],))
        )
        self._last_output = SupportLockedPrimitiveOutput(
            theta=theta,
            length=length,
            width=width,
            alpha=alpha,
            occupancy_logits=occupancy_logits,
            prototype_logits=prototype_logits,
            anchor_map=anchor_map,
            support_mask=support_mask,
            small_component_mask=small_component_mask,
            exclusion_map=exclusion_map,
            component_overflow_score=component_overflow_score,
        )
        return feature_map + rendered * self.residual_scale

    def latest_output(self) -> SupportLockedPrimitiveOutput | None:
        """Return cached sparse-primitive predictions from the latest forward pass."""

        return self._last_output

    def prototype_anchor_metadata(self) -> dict[str, torch.Tensor]:
        """Return prototype-anchor metadata for proxy-supervision helpers."""

        return {
            "orientation": self.renderer.orientation_anchors,
            "length": self.renderer.length_anchors,
            "width": self.renderer.width_anchors,
        }

    def basis_usage_entropy_loss(self) -> torch.Tensor | None:
        """Encourage low-entropy prototype routing."""

        if self._last_output is None:
            return None
        probs = torch.softmax(self._last_output.prototype_logits, dim=1)
        entropy = -(probs * probs.clamp_min(1.0e-6).log()).sum(dim=1)
        return entropy.mean()

    def gate_l1_loss(self) -> torch.Tensor | None:
        """Keep the brush-head regularization interface consistent."""

        return None

    def gate_tv_loss(self) -> torch.Tensor | None:
        """Keep the brush-head regularization interface consistent."""

        return None


__all__ = ["SupportLockedPrimitiveBrushHead", "SupportLockedPrimitiveOutput"]
