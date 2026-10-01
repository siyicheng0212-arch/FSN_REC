"""Frozen-evidence relation models; outputs are binary logits, not flow gates.

The dual prototype reads the last/first *sampled* global tokens. Their positions
are normalized cache indices, not verified frame presentation timestamps or
precise boundary windows. Global and local evidence is detached by construction.
The model never accepts action logits, ground-truth actions, or source labels.
"""

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RelationConfig:
    global_dim: int = 1280
    local_dim: int = 2048
    dim: int = 64
    heads: int = 4
    endpoint_tokens: int = 2
    dropout: float = 0.1
    mode: str = "dual"

    def __post_init__(self):
        for name in ("global_dim", "local_dim", "dim", "heads", "endpoint_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.mode not in {"mlp", "dual"}:
            raise ValueError("mode must be 'mlp' or 'dual'")


def _pair(left: Tensor, right: Tensor) -> Tensor:
    # Concatenating the two original vectors preserves left/right direction.
    return torch.cat((left, right, (left - right).abs(), left * right), dim=-1)


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class RelationNet(nn.Module):
    """Estimate annotation-defined connection reliability from a clip pair.

    Each side must supply ``global_tokens[B,Tg,Cg]``,
    ``local_tokens[B,Tl,Cl]``, ``global_positions[B,Tg]``, and
    ``local_positions[B,Tl]``. Positions must be sorted, normalized to [0,1],
    and may repeat. This network does not assert that its inputs encode enough
    evidence to recover clinical order. Calibration/thresholding is external.
    """

    def __init__(self, config: RelationConfig = RelationConfig()):
        super().__init__()
        self.config = config
        d = config.dim
        self.global_projection = nn.Sequential(nn.Linear(config.global_dim, d), nn.LayerNorm(d))
        self.local_projection = nn.Sequential(nn.Linear(config.local_dim, d), nn.LayerNorm(d))
        self.scene_head = _mlp(8 * d, d, 2 * d, config.dropout)

        if config.mode == "dual":
            self.position_embedding = _mlp(1, d, d, config.dropout)
            self.side_embedding = nn.Embedding(2, d)
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=config.heads, dim_feedforward=4 * d,
                dropout=config.dropout, activation="gelu", batch_first=True,
                norm_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(
                layer, num_layers=1, norm=nn.LayerNorm(d), enable_nested_tensor=False,
            )
            self.temporal_head = _mlp(4 * d, d, 2 * d, config.dropout)
            classifier_dim = 2 * d
        else:
            classifier_dim = d
        self.classifier = _mlp(classifier_dim, 1, d, config.dropout)

    def _evidence(self, side: Mapping[str, Tensor], name: str):
        required = ("global_tokens", "local_tokens", "global_positions", "local_positions")
        missing = [key for key in required if key not in side]
        if missing:
            raise ValueError(f"{name} evidence is missing: {', '.join(missing)}")
        detached = {}
        for key in required:
            value = side[key]
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise ValueError(f"{name}.{key} must be a floating-point tensor")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name}.{key} must be finite")
            detached[key] = value.detach()

        global_tokens, local_tokens = detached["global_tokens"], detached["local_tokens"]
        for key, channels in (("global_tokens", self.config.global_dim),
                              ("local_tokens", self.config.local_dim)):
            tokens = detached[key]
            if tokens.ndim != 3 or tokens.shape[0] < 1 or tokens.shape[1] < 1 or tokens.shape[2] != channels:
                raise ValueError(f"{name}.{key} must have shape [B,T,{channels}] with B,T > 0")
        if global_tokens.shape[0] != local_tokens.shape[0]:
            raise ValueError(f"{name} global/local batch sizes differ")
        if len({value.device for value in detached.values()}) != 1:
            raise ValueError(f"{name} evidence tensors must share a device")
        for branch in ("global", "local"):
            positions = detached[f"{branch}_positions"]
            if positions.shape != detached[f"{branch}_tokens"].shape[:2]:
                raise ValueError(f"{name}.{branch}_positions must have shape [B,T]")
            if bool((positions < 0).any()) or bool((positions > 1).any()):
                raise ValueError(f"{name}.{branch}_positions must be in [0,1]")
            if bool((positions[:, 1:] < positions[:, :-1]).any()):
                raise ValueError(f"{name}.{branch}_positions must be ordered")
        return detached

    def forward(self, left: Mapping[str, Tensor], right: Mapping[str, Tensor]) -> Tensor:
        left, right = self._evidence(left, "left"), self._evidence(right, "right")
        if left["global_tokens"].shape[0] != right["global_tokens"].shape[0]:
            raise ValueError("left/right batch sizes differ")
        if left["global_tokens"].device != right["global_tokens"].device:
            raise ValueError("left/right evidence must share a device")

        global_left = self.global_projection(left["global_tokens"])
        global_right = self.global_projection(right["global_tokens"])
        local_left = self.local_projection(left["local_tokens"])
        local_right = self.local_projection(right["local_tokens"])
        scene = self.scene_head(torch.cat((
            _pair(global_left.mean(1), global_right.mean(1)),
            _pair(local_left.mean(1), local_right.mean(1)),
        ), dim=-1))

        if self.config.mode == "dual":
            k = self.config.endpoint_tokens
            tail, head = global_left[:, -k:], global_right[:, :k]
            tail_pos = left["global_positions"][:, -k:]
            head_pos = right["global_positions"][:, :k]
            tokens = torch.cat((tail, head), dim=1)
            positions = torch.cat((tail_pos, head_pos), dim=1).to(dtype=tokens.dtype)
            sides = torch.cat((
                torch.zeros(tail.shape[1], dtype=torch.long, device=tokens.device),
                torch.ones(head.shape[1], dtype=torch.long, device=tokens.device),
            ))
            tokens = tokens + self.position_embedding(positions.unsqueeze(-1))
            tokens = tokens + self.side_embedding(sides).unsqueeze(0)
            temporal_tokens = self.temporal_encoder(tokens)
            split = tail.shape[1]
            temporal = self.temporal_head(_pair(
                temporal_tokens[:, :split].mean(1), temporal_tokens[:, split:].mean(1),
            ))
            scene = torch.cat((scene, temporal), dim=-1)

        return self.classifier(scene).squeeze(-1)
