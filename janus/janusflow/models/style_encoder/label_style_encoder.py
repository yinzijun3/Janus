"""Label-driven style encoder for JanusFlow-Art."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from janus.janusflow.models.style_encoder.common import StyleEncoderOutput


class LabelStyleEncoder(nn.Module):
    """Encode style, period, and medium IDs into global and local style tokens."""

    def __init__(
        self,
        *,
        num_style_labels: int,
        num_period_labels: int,
        num_medium_labels: int,
        hidden_size: int = 512,
        num_global_tokens: int = 4,
        num_local_tokens: int = 8,
        dropout: float = 0.0,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_global_tokens = num_global_tokens
        self.num_local_tokens = num_local_tokens
        self.style_embedding = nn.Embedding(num_style_labels, hidden_size, padding_idx=padding_idx)
        self.period_embedding = nn.Embedding(num_period_labels, hidden_size, padding_idx=padding_idx)
        self.medium_embedding = nn.Embedding(num_medium_labels, hidden_size, padding_idx=padding_idx)
        self.missing_embedding = nn.Parameter(torch.zeros(hidden_size))
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.global_token_head = nn.Linear(hidden_size, hidden_size * num_global_tokens)
        self.local_token_head = nn.Linear(hidden_size, hidden_size * num_local_tokens)
        self.style_classifier = nn.Linear(hidden_size, num_style_labels)

    def forward(
        self,
        style_label_ids: torch.Tensor,
        period_label_ids: torch.Tensor,
        medium_label_ids: torch.Tensor,
    ) -> StyleEncoderOutput:
        """Encode integer labels into global and local style representations.

        Shapes:
        - `style_label_ids`: `[batch]`
        - `period_label_ids`: `[batch]`
        - `medium_label_ids`: `[batch]`
        """

        style_embed = self.style_embedding(style_label_ids)
        period_embed = self.period_embedding(period_label_ids)
        medium_embed = self.medium_embedding(medium_label_ids)

        embeds = torch.stack([style_embed, period_embed, medium_embed], dim=1)
        valid_mask = torch.stack(
            [
                style_label_ids.ne(0),
                period_label_ids.ne(0),
                medium_label_ids.ne(0),
            ],
            dim=1,
        ).float()
        valid_count = valid_mask.sum(dim=1, keepdim=True)
        fused = (embeds * valid_mask.unsqueeze(-1)).sum(dim=1)
        fused = fused / valid_count.clamp_min(1.0)
        missing_rows = valid_count.squeeze(-1).eq(0)
        if missing_rows.any():
            fused = fused.clone()
            fused[missing_rows] = self.missing_embedding
        fused = self.fusion(fused)

        global_tokens = self.global_token_head(fused).view(
            fused.shape[0],
            self.num_global_tokens,
            self.hidden_size,
        )
        local_tokens = self.local_token_head(fused).view(
            fused.shape[0],
            self.num_local_tokens,
            self.hidden_size,
        )
        return StyleEncoderOutput(
            global_embedding=fused,
            global_tokens=global_tokens,
            local_tokens=local_tokens,
            style_logits=self.style_classifier(fused),
        )
