"""Common datatypes for style encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class StyleEncoderOutput:
    """Style encoder outputs for JanusFlow-Art conditioning and losses."""

    global_embedding: torch.Tensor
    global_tokens: torch.Tensor
    local_tokens: torch.Tensor
    style_logits: Optional[torch.Tensor] = None
