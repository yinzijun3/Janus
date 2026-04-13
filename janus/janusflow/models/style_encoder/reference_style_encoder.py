"""Reference-image style encoder for JanusFlow-Art."""

from __future__ import annotations

import torch
import torch.nn as nn

from janus.janusflow.models.style_encoder.common import StyleEncoderOutput


class ReferenceStyleImageEncoder(nn.Module):
    """Encode a reference style image into global and local style features."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        hidden_size: int = 512,
        num_global_tokens: int = 4,
        num_local_tokens: int = 8,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_global_tokens = num_global_tokens
        self.num_local_tokens = num_local_tokens
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels * 4, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.local_pool = nn.AdaptiveAvgPool2d((2, max(1, num_local_tokens // 2)))
        self.global_token_head = nn.Linear(hidden_size, hidden_size * num_global_tokens)

    def forward(self, reference_images: torch.Tensor) -> StyleEncoderOutput:
        """Encode style-reference images.

        Shape:
        - `reference_images`: `[batch, 3, height, width]`
        """

        features = self.stem(reference_images)
        pooled = self.global_pool(features).flatten(1)
        global_tokens = self.global_token_head(pooled).view(
            pooled.shape[0],
            self.num_global_tokens,
            self.hidden_size,
        )
        local_map = self.local_pool(features)
        # `[batch, hidden, 2, 4]` -> `[batch, 8, hidden]` for the default configuration.
        local_tokens = local_map.flatten(2).transpose(1, 2)
        if local_tokens.shape[1] > self.num_local_tokens:
            local_tokens = local_tokens[:, : self.num_local_tokens, :]
        return StyleEncoderOutput(
            global_embedding=pooled,
            global_tokens=global_tokens,
            local_tokens=local_tokens,
            style_logits=None,
        )
