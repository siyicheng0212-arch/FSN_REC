"""Exact three core TCN classes, without the private dataset/orchestrator dependencies."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class TCNConfig:
    global_dim: int
    local_dim: int
    hidden_dim: int = 64
    blocks: int = 3
    dropout: float = .1

    def __post_init__(self):
        if min(self.global_dim, self.local_dim, self.hidden_dim, self.blocks) < 1:
            raise ValueError("TCN dimensions and blocks must be positive")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")

    @property
    def receptive_field_clips(self):
        # One width-three convolution per residual block, dilation 1, 2, 4...
        return 1 + 2 * (2**self.blocks - 1)


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        update = self.mix(self.dropout(F.gelu(self.conv(sequence))))
        # Mask after every block: padding cannot leak back into a valid clip.
        return (sequence + update) * valid[:, None, :]


class FrozenEvidenceTCN(nn.Module):
    """A logits plus a trainable TCN correction; initial output equals A."""

    def __init__(self, config: TCNConfig):
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(nn.Linear(config.global_dim + config.local_dim + 7,
                                            config.hidden_dim),
                                  nn.LayerNorm(config.hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList(TemporalBlock(config.hidden_dim, 2**level, config.dropout)
                                    for level in range(config.blocks))
        self.classifier = nn.Conv1d(config.hidden_dim, 7, 1)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features: torch.Tensor, a_logits: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        if (features.ndim != 3 or a_logits.shape != (*features.shape[:2], 7)
                or valid.shape != features.shape[:2]
                or features.shape[-1] != self.config.global_dim + self.config.local_dim + 7):
            raise ValueError("TCN input shapes do not match its frozen evidence config")
        if valid.dtype != torch.bool or not bool(valid.any(dim=1).all()):
            raise ValueError("every sequence must have at least one valid clip")
        if not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(a_logits).all()):
            raise ValueError("frozen evidence must be finite")
        hidden = self.stem(features).transpose(1, 2) * valid[:, None, :]
        for block in self.blocks:
            hidden = block(hidden, valid)
        delta = self.classifier(hidden).transpose(1, 2)
        return a_logits + delta * valid[:, :, None]

