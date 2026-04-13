"""Decoder-local brush conditioning modules for JanusFlow-Art."""

from __future__ import annotations

import torch
import torch.nn as nn

from .patch_reference_encoder import BrushReferenceOutput


class BrushConditionInjector(nn.Module):
    """Inject multi-scale brush reference tokens into decoder tokens only."""

    def __init__(
        self,
        *,
        style_hidden_size: int = 512,
        decoder_hidden_size: int = 768,
        use_mid_tokens: bool = True,
        use_fine_tokens: bool = True,
        cross_attention_scale: float = 0.25,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_mid_tokens = use_mid_tokens
        self.use_fine_tokens = use_fine_tokens
        self.cross_attention_scale = cross_attention_scale
        self.query = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        self.key = nn.Linear(style_hidden_size, decoder_hidden_size)
        self.value = nn.Linear(style_hidden_size, decoder_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        decoder_tokens: torch.Tensor,
        brush_output: BrushReferenceOutput | None,
    ) -> torch.Tensor:
        """Apply local texture cross-attention before the decoder model.

        Shape:
        - `decoder_tokens`: `[batch, num_decoder_tokens, decoder_hidden]`
        """

        if brush_output is None:
            return decoder_tokens
        token_groups = []
        if self.use_mid_tokens:
            token_groups.append(brush_output.mid_tokens)
        if self.use_fine_tokens:
            token_groups.append(brush_output.fine_tokens)
        if not token_groups:
            return decoder_tokens

        brush_tokens = torch.cat(token_groups, dim=1)
        query = self.query(decoder_tokens)
        key = self.key(brush_tokens)
        value = self.value(brush_tokens)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / max(query.shape[-1] ** 0.5, 1.0)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, value)
        return decoder_tokens + self.dropout(context) * self.cross_attention_scale
