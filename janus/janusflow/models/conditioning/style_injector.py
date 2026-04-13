"""Style injection utilities for JanusFlow-Art."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class StyleConditionInjector(nn.Module):
    """Inject global style and local texture features into JanusFlow generation."""

    def __init__(
        self,
        *,
        style_hidden_size: int = 512,
        llm_hidden_size: int = 2048,
        decoder_hidden_size: int = 768,
        global_scale: float = 1.0,
        local_scale: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.global_scale = global_scale
        self.local_scale = local_scale
        self.global_to_z = nn.Linear(style_hidden_size, llm_hidden_size)
        self.global_to_t = nn.Linear(style_hidden_size, llm_hidden_size)
        self.decoder_query = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        self.local_key = nn.Linear(style_hidden_size, decoder_hidden_size)
        self.local_value = nn.Linear(style_hidden_size, decoder_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def inject_generation_context(
        self,
        z_tokens: torch.Tensor,
        t_embed: torch.Tensor,
        *,
        global_tokens: Optional[torch.Tensor] = None,
        global_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inject early global style features before the language model.

        Shapes:
        - `z_tokens`: `[batch, num_latent_tokens, llm_hidden]`
        - `t_embed`: `[batch, llm_hidden]`
        - `global_tokens`: `[batch, num_global_tokens, style_hidden]`
        - `global_embedding`: `[batch, style_hidden]`
        """

        if global_embedding is None and global_tokens is not None:
            global_embedding = global_tokens.mean(dim=1)
        if global_embedding is None:
            return z_tokens, t_embed

        z_delta = torch.tanh(self.global_to_z(global_embedding)).unsqueeze(1)
        t_delta = torch.tanh(self.global_to_t(global_embedding))
        return (
            z_tokens + self.dropout(z_delta) * self.global_scale,
            t_embed + self.dropout(t_delta) * self.global_scale,
        )

    def inject_decoder_tokens(
        self,
        decoder_tokens: torch.Tensor,
        *,
        local_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inject later local texture features before the JanusFlow decoder.

        Shapes:
        - `decoder_tokens`: `[batch, num_latent_tokens, decoder_hidden]`
        - `local_tokens`: `[batch, num_local_tokens, style_hidden]`
        """

        if local_tokens is None or local_tokens.numel() == 0:
            return decoder_tokens
        query = self.decoder_query(decoder_tokens)
        key = self.local_key(local_tokens)
        value = self.local_value(local_tokens)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / max(query.shape[-1] ** 0.5, 1.0)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, value)
        return decoder_tokens + self.dropout(context) * self.local_scale
