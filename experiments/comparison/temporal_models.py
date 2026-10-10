"""Temporal heads for controlled, frozen-VideoMAE feature comparisons.

All models use identical per-clip mean-pooled global/local features and the same
seven frozen A logits. Sequence models receive only source-protocol-eligible
chains and never cross structural cuts. MS-TCN++/ASFormer are deliberately
labelled adaptations for clip-level sequence classification, not reproductions
of dense frame-wise action-segmentation systems.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from experiments.relation.tcn_baseline import FrozenEvidenceTCN, TCNConfig


NUM_CLASSES = 7
VALID_MODELS = ("mlp", "bigru", "tcn", "mstcnpp_adapted", "local_transformer")


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        delta = self.mix(self.dropout(F.gelu(self.conv(x))))
        return (x + delta) * valid[:, None, :].to(x.dtype)


class TemporalRefinementStage(nn.Module):
    """One TCN refinement stage, with clip-wise residual logits."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            ResidualTemporalBlock(hidden_dim, dilation, dropout)
            for dilation in (1, 2, 4)
        )
        self.classifier = nn.Conv1d(hidden_dim, NUM_CLASSES, 1)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor, base_logits: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(x).transpose(1, 2)
        hidden = hidden * valid[:, None, :].to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, valid)
        correction = self.classifier(hidden).transpose(1, 2)
        return base_logits + correction * valid.unsqueeze(-1).to(base_logits.dtype)


class TemporalComparisonModel(nn.Module):
    """A common interface for MLP, BiGRU, TCN, MS-TCN++-style and local Transformer heads."""

    def __init__(
        self,
        model_name: str,
        global_dim: int = 768,
        local_dim: int = 768,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        tcn_dilations: tuple[int, ...] = (1, 2, 4),
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        local_radius: int = 4,
        max_sequence_length: int = 128,
    ):
        super().__init__()
        if model_name not in VALID_MODELS:
            raise ValueError(f"model_name must be in {VALID_MODELS}")
        if min(global_dim, local_dim, hidden_dim, transformer_layers,
               transformer_heads, max_sequence_length) < 1:
            raise ValueError("all model dimensions must be positive")
        if hidden_dim % transformer_heads != 0:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0,1)")
        if local_radius < 0:
            raise ValueError("local_radius must be nonnegative")
        self.model_name = model_name
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.input_dim = global_dim + local_dim + NUM_CLASSES
        self.max_sequence_length = max_sequence_length
        self.local_radius = local_radius

        if model_name == "mlp":
            self.clip_head = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, NUM_CLASSES),
            )
            nn.init.zeros_(self.clip_head[-1].weight)
            nn.init.zeros_(self.clip_head[-1].bias)

        elif model_name == "bigru":
            self.input_projection = nn.Sequential(
                nn.Linear(self.input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.sequence_model = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=dropout,
            )
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, NUM_CLASSES))
            nn.init.zeros_(self.classifier[-1].weight)
            nn.init.zeros_(self.classifier[-1].bias)

        elif model_name == "tcn":
            config = TCNConfig(
                global_dim=global_dim,
                local_dim=local_dim,
                hidden_dim=hidden_dim,
                blocks=len(tcn_dilations),
                dropout=dropout,
                dilations=tcn_dilations,
            )
            self.tcn = FrozenEvidenceTCN(config)

        elif model_name == "mstcnpp_adapted":
            # Adapted multi-stage sequence refinement for clip labels.
            # This is not the official dense frame-wise MS-TCN++ implementation.
            self.stage1 = TemporalRefinementStage(self.input_dim, hidden_dim, dropout)
            self.refiners = nn.ModuleList(
                TemporalRefinementStage(NUM_CLASSES, hidden_dim, dropout)
                for _ in range(2)
            )

        elif model_name == "local_transformer":
            self.input_projection = nn.Sequential(
                nn.Linear(self.input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            self.position_embedding = nn.Parameter(
                torch.zeros(1, max_sequence_length, hidden_dim)
            )
            nn.init.trunc_normal_(self.position_embedding, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=transformer_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sequence_model = nn.TransformerEncoder(
                layer, num_layers=transformer_layers, enable_nested_tensor=False
            )
            self.norm = nn.LayerNorm(hidden_dim)
            self.classifier = nn.Linear(hidden_dim, NUM_CLASSES)
            nn.init.zeros_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)

    @staticmethod
    def _mean_features(global_tokens: torch.Tensor, local_tokens: torch.Tensor,
                       a_logits: torch.Tensor) -> torch.Tensor:
        return torch.cat((
            global_tokens.mean(dim=2),
            local_tokens.mean(dim=2),
            a_logits,
        ), dim=-1)

    def forward_all(
        self,
        global_tokens: torch.Tensor,
        local_tokens: torch.Tensor,
        a_logits: torch.Tensor,
        valid: torch.Tensor,
    ) -> list[torch.Tensor]:
        if global_tokens.ndim != 4 or local_tokens.ndim != 4:
            raise ValueError("global/local tokens must be [batch, clips, positions, channels]")
        if global_tokens.shape[:3] != local_tokens.shape[:3]:
            raise ValueError("global/local token axes do not match")
        if global_tokens.shape[-1] != self.global_dim or local_tokens.shape[-1] != self.local_dim:
            raise ValueError("global/local feature dimensions do not match configuration")
        if a_logits.shape != (*global_tokens.shape[:2], NUM_CLASSES):
            raise ValueError("a_logits must be [batch, clips, 7]")
        if valid.shape != global_tokens.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean [batch, clips] mask")
        if not bool(valid.any(dim=1).all()):
            raise ValueError("every segment must contain at least one real clip")
        x = self._mean_features(global_tokens, local_tokens, a_logits)
        mask = valid.unsqueeze(-1).to(x.dtype)

        if self.model_name == "mlp":
            return [a_logits + self.clip_head(x) * mask]

        if self.model_name == "bigru":
            hidden = self.input_projection(x) * mask
            lengths = valid.sum(dim=1).to(device="cpu", dtype=torch.long)
            packed = pack_padded_sequence(
                hidden, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.sequence_model(packed)
            hidden, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=x.shape[1]
            )
            correction = self.classifier(hidden) * mask
            return [a_logits + correction]

        if self.model_name == "tcn":
            return [self.tcn(x, a_logits, valid)]

        if self.model_name == "mstcnpp_adapted":
            outputs = [self.stage1(x, a_logits, valid)]
            for refiner in self.refiners:
                previous = outputs[-1]
                # MS-TCN-family refinement stages consume previous per-class scores.
                refinement_input = previous.softmax(dim=-1)
                outputs.append(refiner(refinement_input, previous, valid))
            return outputs

        if self.model_name == "local_transformer":
            length = x.shape[1]
            if length > self.max_sequence_length:
                raise ValueError(
                    f"sequence length {length} exceeds max_sequence_length={self.max_sequence_length}"
                )
            positions = self.position_embedding[:, :length]
            hidden = (self.input_projection(x) + positions) * mask
            distance = torch.arange(length, device=x.device)
            # True means attention is blocked. Only local temporal neighborhoods
            # are visible, following the inductive-bias idea in ASFormer.
            attention_mask = (distance[:, None] - distance[None, :]).abs() > self.local_radius
            hidden = self.sequence_model(
                hidden,
                mask=attention_mask,
                src_key_padding_mask=~valid,
            )
            correction = self.classifier(self.norm(hidden)) * mask
            return [a_logits + correction]

        raise AssertionError("validated model name became unreachable")

    def forward(self, global_tokens, local_tokens, a_logits, valid):
        return self.forward_all(global_tokens, local_tokens, a_logits, valid)[-1]

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
