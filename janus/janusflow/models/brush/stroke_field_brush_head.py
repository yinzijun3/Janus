"""Stroke-field brush head skeleton for V2 probes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .stroke_field_pseudo_renderer import StrokeFieldPseudoRenderer


@dataclass
class StrokeFieldOutput:
    """Most recent stroke-field predictions."""

    theta: torch.Tensor
    length: torch.Tensor
    width: torch.Tensor
    curvature: torch.Tensor
    alpha: torch.Tensor
    objectness: torch.Tensor
    prototype_logits: torch.Tensor


class StrokeFieldBrushHead(nn.Module):
    """Predict stroke fields and compose a pseudo-rendered probe residual."""

    def __init__(
        self,
        *,
        feature_channels: int = 768,
        hidden_channels: int = 384,
        prototype_count: int = 8,
        renderer_kernel_size: int = 9,
        render_channels: int = 8,
        topk_prototypes: int = 2,
        support_gate_enabled: bool = False,
        support_threshold: float = 0.20,
        support_temperature: float = 0.08,
        support_dilation: int = 5,
        support_mix: float = 0.65,
        support_hard_mask_enabled: bool = False,
        support_opening_kernel: int = 0,
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
        self.theta_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.length_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.width_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.curvature_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.alpha_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.objectness_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.prototype_head = nn.Conv2d(hidden_channels, prototype_count, kernel_size=1)
        self.renderer = StrokeFieldPseudoRenderer(
            feature_channels=feature_channels,
            prototype_count=prototype_count,
            kernel_size=renderer_kernel_size,
            render_channels=render_channels,
            topk_prototypes=topk_prototypes,
            support_gate_enabled=support_gate_enabled,
            support_threshold=support_threshold,
            support_temperature=support_temperature,
            support_dilation=support_dilation,
            support_mix=support_mix,
            support_hard_mask_enabled=support_hard_mask_enabled,
            support_opening_kernel=support_opening_kernel,
        )
        self._last_output: StrokeFieldOutput | None = None

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Apply a pseudo-rendered stroke residual to the feature map."""

        hidden = self.stem(feature_map)
        theta = self.theta_head(hidden)
        length = self.length_head(hidden)
        width = self.width_head(hidden)
        curvature = self.curvature_head(hidden)
        alpha = self.alpha_head(hidden)
        objectness = self.objectness_head(hidden)
        prototype_logits = self.prototype_head(hidden)
        rendered = self.renderer(
            feature_map,
            theta=theta,
            length=length,
            width=width,
            curvature=curvature,
            alpha=alpha,
            objectness=objectness,
            prototype_logits=prototype_logits,
        )
        self._last_output = StrokeFieldOutput(
            theta=theta,
            length=length,
            width=width,
            curvature=curvature,
            alpha=alpha,
            objectness=objectness,
            prototype_logits=prototype_logits,
        )
        return feature_map + rendered * self.residual_scale

    def latest_output(self) -> StrokeFieldOutput | None:
        """Return cached stroke-field predictions from the latest forward pass."""

        return self._last_output

    def prototype_anchor_metadata(self) -> dict[str, torch.Tensor]:
        """Return renderer prototype anchors for supervision helpers."""

        return {
            "orientation": self.renderer.orientation_anchors,
            "length": self.renderer.length_anchors,
            "width": self.renderer.width_anchors,
        }

    def basis_usage_entropy_loss(self) -> torch.Tensor | None:
        """Encourage sharper local prototype selection when requested."""

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


__all__ = ["StrokeFieldBrushHead", "StrokeFieldOutput"]
