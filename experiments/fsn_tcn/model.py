"""Single-stage reference TCN and an architecture-independent hard-cut wrapper.

The wrapper cuts BEFORE temporal inference: every connected segment is passed
through the same base model independently. It never masks only the final output.
The reference TCN is a new reference implementation, not a reproduction of a
private or unavailable ``tcn_baseline.py``. It is bidirectional/offline.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TCNConfig:
    input_dim: int
    width: int = 64
    layers: int = 4
    kernel_size: int = 3
    dropout: float = 0.1
    num_classes: int = 7

    def __post_init__(self):
        for name in ("input_dim", "width", "layers", "kernel_size", "num_classes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd to preserve temporal length")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, Real):
            raise ValueError("dropout must be a finite number in [0, 1)")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be a finite number in [0, 1)")


def _validate_inputs(features: Tensor, a_logits: Tensor) -> None:
    if not isinstance(features, Tensor) or not isinstance(a_logits, Tensor):
        raise ValueError("features and a_logits must be tensors")
    if features.ndim != 2 or min(features.shape) <= 0:
        raise ValueError("features must have nonempty shape [T, F]")
    if a_logits.ndim != 2 or min(a_logits.shape) <= 0:
        raise ValueError("a_logits must have nonempty shape [T, C]")
    if features.shape[0] != a_logits.shape[0]:
        raise ValueError("features and a_logits must have the same temporal length")
    if not features.is_floating_point() or not a_logits.is_floating_point():
        raise ValueError("features and a_logits must be real floating-point tensors")
    if features.device != a_logits.device:
        raise ValueError("features and a_logits must be on the same device")
    if not torch.isfinite(features).all().item() or not torch.isfinite(a_logits).all().item():
        raise ValueError("features and a_logits must be finite")


def _open_flags(open_edges: Sequence[bool] | Tensor, expected: int) -> list[bool]:
    if isinstance(open_edges, Tensor):
        if open_edges.dtype != torch.bool or open_edges.ndim != 1:
            raise ValueError("open_edges must be a one-dimensional boolean tensor")
        flags = open_edges.detach().cpu().tolist()
    else:
        if isinstance(open_edges, (str, bytes)):
            raise ValueError("open_edges must be a sequence of booleans")
        try:
            flags = list(open_edges)
        except TypeError as exc:
            raise ValueError("open_edges must be a sequence of booleans") from exc
        if any(type(flag) is not bool for flag in flags):
            raise ValueError("open_edges must contain booleans, not numeric masks")
    if len(flags) != expected:
        raise ValueError(f"open_edges must have length T-1 ({expected})")
    return flags


class _DilatedResidualBlock(nn.Module):
    def __init__(self, config: TCNConfig, dilation: int):
        super().__init__()
        self.temporal = nn.Conv1d(
            config.width, config.width, config.kernel_size,
            padding=dilation * (config.kernel_size - 1) // 2, dilation=dilation,
        )
        self.activation = nn.ReLU()
        self.pointwise = nn.Conv1d(config.width, config.width, 1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, values: Tensor) -> Tensor:
        residual = self.pointwise(self.activation(self.temporal(values)))
        return values + self.dropout(residual)


class TemporalResidualTCN(nn.Module):
    """Single-stage dilated residual TCN over frozen A features.

    Input is one unpadded chain ``features[T,F]`` and ``a_logits[T,C]``.
    Dilation is 1, 2, 4, ...; no batch or temporal normalization is used.
    A zero-initialized final projection starts as an exact A-logit residual.
    Singleton chains always return the original A logits, even after training.
    """

    def __init__(self, config: TCNConfig):
        super().__init__()
        if not isinstance(config, TCNConfig):
            raise ValueError("config must be a TCNConfig")
        self.config = config
        self.input_projection = nn.Conv1d(config.input_dim, config.width, 1)
        self.activation = nn.ReLU()
        self.blocks = nn.Sequential(*[
            _DilatedResidualBlock(config, 2 ** layer) for layer in range(config.layers)
        ])
        self.output_projection = nn.Conv1d(config.width, config.num_classes, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, features: Tensor, a_logits: Tensor) -> Tensor:
        _validate_inputs(features, a_logits)
        if features.shape[1] != self.config.input_dim:
            raise ValueError("feature width does not match input_dim")
        if a_logits.shape[1] != self.config.num_classes:
            raise ValueError("A logit width does not match num_classes")
        if features.shape[0] == 1:
            return a_logits
        hidden = self.activation(self.input_projection(features.T.unsqueeze(0)))
        hidden = self.blocks(hidden)
        delta = self.output_projection(hidden).squeeze(0).T
        logits = a_logits + delta.to(dtype=a_logits.dtype)
        if not torch.isfinite(logits).all().item():
            raise ValueError("TCN produced nonfinite logits")
        return logits


class SegmentedTCN(nn.Module):
    """Run one shared base TCN independently on segments separated by cuts.

    ``base.forward(features[T,F], a_logits[T,C])`` must return logits[T,C].
    This works with the new reference TCN or a user-supplied compatible model.
    A supplied base must not carry recurrent state between forward calls.
    No features, intermediate activations, or temporal convolutions cross a cut.
    All-open chains of length >= 2 invoke exactly the unchanged base forward.
    All-closed chains and individual isolated clips return exact A logits.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        if not isinstance(base, nn.Module):
            raise ValueError("base must be an nn.Module")
        self.base = base

    def _segment(self, features: Tensor, a_logits: Tensor) -> Tensor:
        if features.shape[0] == 1:
            return a_logits
        logits = self.base(features, a_logits)
        if not isinstance(logits, Tensor) or logits.shape != a_logits.shape:
            raise ValueError("base must return logits with the same [T,C] shape as A")
        if logits.device != a_logits.device or not logits.is_floating_point():
            raise ValueError("base logits must be floating-point on the A device")
        if not torch.isfinite(logits).all().item():
            raise ValueError("base produced nonfinite logits")
        return logits

    def forward(self, features: Tensor, a_logits: Tensor,
                open_edges: Sequence[bool] | Tensor) -> Tensor:
        _validate_inputs(features, a_logits)
        flags = _open_flags(open_edges, features.shape[0] - 1)
        if not flags or not any(flags):
            return a_logits
        if all(flags):
            return self._segment(features, a_logits)
        boundaries = [0, *[i + 1 for i, flag in enumerate(flags) if not flag], features.shape[0]]
        outputs = [
            self._segment(features[start:end], a_logits[start:end])
            for start, end in zip(boundaries, boundaries[1:])
        ]
        return torch.cat(outputs, dim=0)
