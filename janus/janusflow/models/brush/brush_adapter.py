"""Decoder-side brush adapters for JanusFlow-Art."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .patch_reference_encoder import BrushReferenceOutput


class BrushAdapter(nn.Module):
    """Small residual adapter for decoder tokens near the image decoder.

    The adapter operates on the `[batch, num_tokens, hidden]` decoder-token map
    as a local 2D feature field. Its residual gate is initialized conservatively
    so early training favors quality preservation over aggressive texture drift.
    """

    def __init__(
        self,
        *,
        decoder_hidden_size: int = 768,
        style_hidden_size: int = 512,
        bottleneck_channels: int = 192,
        kernel_size: int = 3,
        residual_init: float = 0.02,
        residual_scale: float = 1.0,
        dropout: float = 0.0,
        expert_count: int = 0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.expert_count = expert_count
        self.residual_scale = float(residual_scale)
        self.norm = nn.LayerNorm(decoder_hidden_size)
        self.in_proj = nn.Linear(decoder_hidden_size, bottleneck_channels)
        self.depthwise = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=bottleneck_channels,
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(bottleneck_channels, decoder_hidden_size)
        self.film = nn.Linear(style_hidden_size, decoder_hidden_size * 2)
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_init)))
        if expert_count > 1:
            self.expert_residuals = nn.ModuleList(
                nn.Linear(decoder_hidden_size, decoder_hidden_size, bias=False)
                for _ in range(expert_count)
            )
            self.router = nn.Linear(style_hidden_size, expert_count)
        else:
            self.expert_residuals = None
            self.router = None

    def _conditioning_embedding(
        self,
        brush_output: BrushReferenceOutput | None,
        decoder_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if brush_output is None:
            return decoder_tokens.new_zeros(decoder_tokens.shape[0], self.film.in_features)
        return brush_output.coarse_tokens.mean(dim=1)

    def forward(
        self,
        decoder_tokens: torch.Tensor,
        brush_output: BrushReferenceOutput | None = None,
    ) -> torch.Tensor:
        """Return brush-enhanced decoder tokens.

        Shape:
        - `decoder_tokens`: `[batch, num_decoder_tokens, decoder_hidden]`
        """

        batch_size, token_count, hidden_size = decoder_tokens.shape
        side = int(math.sqrt(token_count))
        if side * side != token_count:
            raise ValueError(f"decoder token count must be square, got {token_count}")

        cond = self._conditioning_embedding(brush_output, decoder_tokens).to(dtype=decoder_tokens.dtype)
        scale, shift = self.film(cond).chunk(2, dim=-1)
        scale = torch.tanh(scale).unsqueeze(1)
        shift = torch.tanh(shift).unsqueeze(1)

        hidden = self.norm(decoder_tokens)
        hidden = self.in_proj(hidden)
        hidden = hidden.reshape(batch_size, side, side, -1).permute(0, 3, 1, 2)
        hidden = self.depthwise(hidden)
        hidden = hidden.permute(0, 2, 3, 1).reshape(batch_size, token_count, -1)
        hidden = self.out_proj(self.dropout(self.act(hidden)))
        hidden = hidden * (1.0 + scale) + shift

        if self.expert_residuals is not None and self.router is not None:
            route_weights = torch.softmax(self.router(cond), dim=-1)
            expert_delta = torch.stack(
                [expert(hidden) for expert in self.expert_residuals],
                dim=1,
            )
            hidden = hidden + (expert_delta * route_weights[:, :, None, None]).sum(dim=1)

        gate = self.residual_gate.to(dtype=decoder_tokens.dtype) * self.residual_scale
        return decoder_tokens + hidden * gate
