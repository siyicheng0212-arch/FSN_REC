"""Dual-timescale modules for clip-internal evidence and inter-clip context.

This is an experimental frozen-feature head. It never changes or re-trains the
VideoMAE backbone. Each input clip contains its fixed sequence of sampled
VideoMAE tubelet features; clip sequences are split at the sealed source
protocol's structural cuts before temporal convolutions are applied.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


NUM_CLASSES = 7
def _validate_dilations(blocks: int, dilations: tuple[int, ...] | None) -> tuple[int, ...]:
    schedule = tuple(2 ** level for level in range(blocks)) if dilations is None else tuple(dilations)
    if len(schedule) != blocks or any(type(item) is not int or item < 1 for item in schedule):
        raise ValueError("dilations must contain one positive integer per block")
    return schedule


def temporal_pyramid_pool(tokens: torch.Tensor, levels: tuple[int, ...] = (1, 2, 4)) -> torch.Tensor:
    """Pool [batch, clips, sampled_positions, channels] at 1/2/4 temporal scales.

    Temporal bins are based on normalized positions within each already-cut
    action clip, not on neighboring clips or on the source-video duration.
    The returned feature concatenates the seven pooled bins (1 + 2 + 4).
    """
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 4:
        raise ValueError("tokens must have shape [batch, clips, positions, channels]")
    if not levels or any(type(level) is not int or level < 1 for level in levels):
        raise ValueError("levels must be a nonempty tuple of positive integers")
    positions = tokens.shape[2]
    if positions < max(levels) or any(level < 1 for level in levels):
        raise ValueError("each temporal pyramid level must fit the sampled positions")
    pooled = []
    for bins in levels:
        # Equal-width bins in normalized clip time. Require divisibility to
        # avoid silently unequal windows in this fixed protocol.
        if positions % bins:
            raise ValueError(f"{positions} sampled positions are not divisible by {bins} bins")
        width = positions // bins
        for index in range(bins):
            start, end = index * width, (index + 1) * width
            pooled.append(tokens[:, :, start:end, :].mean(dim=2))
    return torch.cat(pooled, dim=-1)


class MultiScaleClipEncoder(nn.Module):
    """Learn a clip embedding and a zero-initialized seven-class correction."""

    def __init__(
        self,
        global_dim: int = 768,
        local_dim: int = 768,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        levels: tuple[int, ...] = (1, 2, 4),
        pooling_mode: str = "multiscale",
    ):
        super().__init__()
        if min(global_dim, local_dim, embedding_dim) < 1:
            raise ValueError("feature dimensions must be positive")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not levels or any(type(level) is not int or level < 1 for level in levels):
            raise ValueError("levels must be positive integers")
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.embedding_dim = embedding_dim
        if pooling_mode not in {"multiscale", "mean_control"}:
            raise ValueError("pooling_mode must be 'multiscale' or 'mean_control'")
        self.levels = tuple(levels)
        self.pooling_mode = pooling_mode
        self.token_projection = nn.Sequential(
            nn.Linear(global_dim + local_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        pooled_width = sum(self.levels) * embedding_dim
        self.fusion = nn.Sequential(
            nn.Linear(pooled_width, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.class_correction = nn.Linear(embedding_dim, NUM_CLASSES)
        # Initially the new head preserves the frozen VideoMAE decision exactly.
        nn.init.zeros_(self.class_correction.weight)
        nn.init.zeros_(self.class_correction.bias)

    def forward(
        self,
        global_tokens: torch.Tensor,
        local_tokens: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if global_tokens.ndim != 4 or local_tokens.ndim != 4:
            raise ValueError("global/local tokens must be [batch, clips, positions, channels]")
        if global_tokens.shape[:3] != local_tokens.shape[:3]:
            raise ValueError("global/local token batch, clip and position dimensions must match")
        if global_tokens.shape[-1] != self.global_dim or local_tokens.shape[-1] != self.local_dim:
            raise ValueError("global/local channel dimensions differ from encoder configuration")
        if valid.shape != global_tokens.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean [batch, clips] mask")
        if global_tokens.shape[2] < max(self.levels):
            raise ValueError("sampled positions do not support the requested temporal pyramid")
        tokens = torch.cat((global_tokens, local_tokens), dim=-1)
        projected = self.token_projection(tokens)
        if self.pooling_mode == "multiscale":
            pooled = temporal_pyramid_pool(projected, self.levels)
        else:
            # Same projection and fusion layers as the multiscale model, but
            # repeat the same global temporal mean into each of the seven bins.
            # This is a parameter-matched control for extra head capacity.
            mean = projected.mean(dim=2)
            pooled = mean.unsqueeze(2).expand(
                *mean.shape[:2], sum(self.levels), mean.shape[-1]
            ).reshape(*mean.shape[:2], -1)
        embedding = self.fusion(pooled)
        correction = self.class_correction(embedding)
        mask = valid.unsqueeze(-1).to(embedding.dtype)
        return embedding * mask, correction * mask


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.mix = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        update = self.mix(self.dropout(F.gelu(self.conv(hidden))))
        # Re-mask after each block so padded positions cannot leak back inward.
        return (hidden + update) * valid[:, None, :].to(hidden.dtype)


class TemporalContextHead(nn.Module):
    """Offline inter-clip TCN that adds a residual to the incoming class logits."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        blocks: int = 3,
        dropout: float = 0.1,
        dilations: tuple[int, ...] | None = None,
    ):
        super().__init__()
        if min(feature_dim, hidden_dim, blocks) < 1:
            raise ValueError("feature dimensions and block count must be positive")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.dilations = _validate_dilations(blocks, dilations)
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.blocks_count = blocks
        self.dropout_rate = dropout
        self.stem = nn.Sequential(
            nn.Linear(feature_dim + NUM_CLASSES, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            TemporalBlock(hidden_dim, dilation, dropout) for dilation in self.dilations
        )
        self.class_correction = nn.Conv1d(hidden_dim, NUM_CLASSES, kernel_size=1)
        nn.init.zeros_(self.class_correction.weight)
        nn.init.zeros_(self.class_correction.bias)

    @property
    def receptive_field_clips(self) -> int:
        return 1 + 2 * sum(self.dilations)

    def forward(
        self,
        clip_features: torch.Tensor,
        base_logits: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if clip_features.ndim != 3:
            raise ValueError("clip_features must have shape [batch, clips, channels]")
        if clip_features.shape[-1] != self.feature_dim:
            raise ValueError("clip feature dimension differs from TCN configuration")
        if base_logits.shape != (*clip_features.shape[:2], NUM_CLASSES):
            raise ValueError("base_logits must have shape [batch, clips, 7]")
        if valid.shape != clip_features.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean [batch, clips] mask")
        if not bool(valid.any(dim=1).all()):
            raise ValueError("each sequence must contain at least one valid clip")
        hidden = self.stem(torch.cat((clip_features, base_logits), dim=-1)).transpose(1, 2)
        hidden = hidden * valid[:, None, :].to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, valid)
        correction = self.class_correction(hidden).transpose(1, 2)
        return base_logits + correction * valid.unsqueeze(-1).to(base_logits.dtype)


class DualTimescaleModel(nn.Module):
    """M1S/M1/M2/M2S/M3 ablations; M0 is the unmodified frozen A logits."""

    VALID_VARIANTS = ("M1S", "M1", "M2", "M2S", "M3")

    def __init__(
        self,
        variant: str,
        global_dim: int = 768,
        local_dim: int = 768,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        blocks: int = 3,
        dropout: float = 0.1,
        dilations: tuple[int, ...] | None = None,
    ):
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"variant must be one of {self.VALID_VARIANTS}")
        self.variant = variant
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.blocks_count = blocks
        self.dropout_rate = dropout
        self.dilations = _validate_dilations(blocks, dilations)
        uses_clip_head = variant in {"M1S", "M1", "M2S", "M3"}
        pooling_mode = "mean_control" if variant in {"M1S", "M2S"} else "multiscale"
        self.clip_encoder = (
            MultiScaleClipEncoder(
                global_dim, local_dim, embedding_dim, dropout,
                pooling_mode=pooling_mode,
            )
            if uses_clip_head else None
        )
        if variant == "M2":
            context_dim = global_dim + local_dim
        elif variant in {"M2S", "M3"}:
            context_dim = global_dim + local_dim + embedding_dim
        else:
            context_dim = 0
        self.context_head = (
            TemporalContextHead(context_dim, hidden_dim, blocks, dropout, self.dilations)
            if variant in {"M2", "M2S", "M3"} else None
        )

    def forward(
        self,
        global_tokens: torch.Tensor,
        local_tokens: torch.Tensor,
        a_logits: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if global_tokens.ndim != 4 or local_tokens.ndim != 4:
            raise ValueError("global/local tokens must have shape [batch, clips, positions, channels]")
        if global_tokens.shape[:3] != local_tokens.shape[:3]:
            raise ValueError("global/local token axes do not match")
        if a_logits.shape != (*global_tokens.shape[:2], NUM_CLASSES):
            raise ValueError("A logits must have shape [batch, clips, 7]")
        if valid.shape != global_tokens.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean [batch, clips] mask")

        # A simple uniform within-clip aggregate is retained for the inter-only
        # control. M1/M3 additionally see the normalized temporal pyramid.
        global_mean = global_tokens.mean(dim=2)
        local_mean = local_tokens.mean(dim=2)
        base_features = torch.cat((global_mean, local_mean), dim=-1)
        base_logits = a_logits
        embedding = None

        if self.clip_encoder is not None:
            embedding, clip_delta = self.clip_encoder(global_tokens, local_tokens, valid)
            base_logits = a_logits + clip_delta

        if self.variant in {"M1S", "M1"}:
            return base_logits
        if self.variant == "M2":
            return self.context_head(base_features, base_logits, valid)
        if self.variant in {"M2S", "M3"}:
            context_features = torch.cat((base_features, embedding), dim=-1)
            return self.context_head(context_features, base_logits, valid)
        raise AssertionError("validated variant became unreachable")
