"""Target-conditioned modulation of explicitly selected temporal Conv1d taps.

This wraps an existing two-input scorer rather than rebuilding a private legacy
architecture.  The native Conv1d result is retained and off-centre corrections
are added as ``(g - 1) * tap_contribution``.  Thus zero-initialized gates recover
native evaluation exactly while retaining gate gradients.  Gates modulate direct
messages at each layer; they do not prove end-to-end breakpoint isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.utils import parametrize


@dataclass
class _ForwardContext:
    active: bool = False
    length: int = 0
    coefficients: dict[str, dict[int, Tensor]] = field(default_factory=dict)
    called_paths: list[str] = field(default_factory=list)


class _VisualGate(nn.Module):
    def __init__(self, feature_dim: int, dim: int):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(feature_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.network = nn.Sequential(nn.Linear(3 * dim + 1, dim), nn.GELU(), nn.Linear(dim, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, projected: Tensor, targets: Tensor, neighbors: Tensor,
                normalized_offset: float) -> Tensor:
        current = projected.index_select(0, targets)
        neighbor = projected.index_select(0, neighbors)
        signed_offset = current.new_full((len(targets), 1), normalized_offset)
        evidence = torch.cat((current, neighbor, (current - neighbor).abs(), signed_offset), dim=-1)
        return 2.0 * torch.sigmoid(self.network(evidence).squeeze(-1))


def _validate_conv(conv: nn.Module, path: str) -> nn.Conv1d:
    if type(conv) is not nn.Conv1d:
        raise ValueError(f"{path}: temporal_paths must name unwrapped native nn.Conv1d modules")
    if parametrize.is_parametrized(conv) or hasattr(conv, "weight_orig") or hasattr(conv, "weight_g"):
        raise ValueError(f"{path}: parametrized/weight-normalized Conv1d is unsupported; do not strip it")
    if conv._forward_pre_hooks or conv._forward_hooks:
        raise ValueError(f"{path}: Conv1d forward hooks are unsupported because tap contributions may differ")
    if conv.stride != (1,) or conv.kernel_size[0] < 3 or conv.kernel_size[0] % 2 != 1:
        raise ValueError(f"{path}: require stride=1 and odd temporal kernel >= 3")
    required_padding = conv.dilation[0] * (conv.kernel_size[0] - 1) // 2
    if conv.padding != (required_padding,) and conv.padding != "same":
        raise ValueError(f"{path}: require symmetric same-length padding, not causal/custom padding")
    if conv.padding_mode != "zeros":
        raise ValueError(f"{path}: only zero padding is supported")
    return conv


class _UtilityConv(nn.Module):
    """The original Conv1d plus differentiable, off-centre tap corrections."""

    def __init__(self, conv: nn.Conv1d, path: str, context: _ForwardContext):
        super().__init__()
        self.conv = conv
        self.path = path
        self.context = context  # A plain object, shared with the owner, not an nn.Module.

    def forward(self, values: Tensor) -> Tensor:
        if not self.context.active:
            raise RuntimeError("wrapped temporal Conv1d must run through ContextUtilityTCN.forward")
        if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != self.context.length:
            raise ValueError(f"{self.path}: temporal input must align with immutable A evidence [1,C,T]")
        self.context.called_paths.append(self.path)
        # Keep the backend's native arithmetic, bias and central contribution.
        output = self.conv(values)
        kernel = self.conv.kernel_size[0]
        dilation = self.conv.dilation[0]
        padding = dilation * (kernel - 1) // 2
        padded = F.pad(values, (padding, padding))
        for tap in range(kernel):
            offset = (tap - kernel // 2) * dilation
            if offset == 0:
                continue
            start = tap * dilation
            source = padded[:, :, start:start + self.context.length]
            contribution = F.conv1d(source, self.conv.weight[:, :, tap:tap + 1],
                                    bias=None, stride=1, padding=0, dilation=1,
                                    groups=self.conv.groups)
            coefficient = self.context.coefficients[self.path][offset].to(dtype=contribution.dtype)
            output = output + (coefficient.view(1, 1, -1) - 1.0) * contribution
        return output


class ContextUtilityTCN(nn.Module):
    """Adapt an actual TCN while preserving its input/output and singleton policy.

    ``base(features[T,F], a_logits[T,7])`` must be a stateless, single-chain
    scorer.  Load the base checkpoint STRICTLY before constructing this wrapper.
    ``temporal_paths`` explicitly selects supported same-length Conv1d modules;
    no architecture discovery, semantic repair or hidden legacy reconstruction
    occurs.  ``plain`` leaves base modules untouched; ``scalar`` shares one
    coefficient across all selected taps; ``dynamic`` uses visual evidence.

    Dynamic coefficients depend on immutable input visual features, ordered
    target/neighbor evidence, their absolute difference, and signed offset.  No
    source name, action truth, identity, or A class score enters the gate.
    ``2*sigmoid(s)`` lies in (0,2), so these are strengths, not probabilities.
    """

    def __init__(self, base: nn.Module, feature_dim: int,
                 temporal_paths: Sequence[str], mode: str = "dynamic", dim: int = 64,
                 gate_seed: int = 42):
        super().__init__()
        if not isinstance(base, nn.Module):
            raise ValueError("base must be an nn.Module")
        for name, value in (("feature_dim", feature_dim), ("dim", dim)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(gate_seed) is not int or gate_seed < 0:
            raise ValueError("gate_seed must be a nonnegative integer")
        if mode not in {"plain", "scalar", "dynamic"}:
            raise ValueError("mode must be plain, scalar or dynamic")
        if isinstance(temporal_paths, (str, bytes)):
            raise ValueError("temporal_paths must be an explicit sequence of module paths")
        paths = list(temporal_paths)
        if not paths or any(not isinstance(path, str) or not path for path in paths):
            raise ValueError("temporal_paths must be nonempty string paths")
        if len(set(paths)) != len(paths):
            raise ValueError("temporal_paths must not contain duplicates")
        convs = []
        for path in paths:
            try:
                candidate = base.get_submodule(path)
            except AttributeError as exc:
                raise ValueError(f"missing temporal module: {path}") from exc
            convs.append(_validate_conv(candidate, path))
        if len({id(conv) for conv in convs}) != len(convs):
            raise ValueError("temporal_paths must not alias the same Conv1d")
        self.base = base
        self.feature_dim = feature_dim
        self.dim = dim
        self.mode = mode
        self.temporal_paths = tuple(paths)
        self._offsets = {
            path: tuple((tap - conv.kernel_size[0] // 2) * conv.dilation[0]
                        for tap in range(conv.kernel_size[0]) if tap != conv.kernel_size[0] // 2)
            for path, conv in zip(paths, convs)
        }
        self._max_offset = max(abs(offset) for offsets in self._offsets.values() for offset in offsets)
        self._context = _ForwardContext()
        self._last_coefficients: list[dict[str, Any]] = []
        self._last_called_paths: list[str] = []
        self._last_gate_mode: str | None = None
        self.gate: _VisualGate | None = None
        self.register_parameter("scalar_logit", None)
        # Gate initialization consumes neither CPU nor already-initialized CUDA
        # random streams.  It therefore cannot perturb data/dropout randomness.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(gate_seed)
            if mode == "dynamic":
                self.gate = _VisualGate(feature_dim, dim)
            elif mode == "scalar":
                self.scalar_logit = nn.Parameter(torch.zeros(()))
        if mode != "plain":
            for path, conv in zip(paths, convs):
                parent_name, _, name = path.rpartition(".")
                parent = base.get_submodule(parent_name) if parent_name else base
                setattr(parent, name, _UtilityConv(conv, path, self._context))

    def added_parameter_count(self) -> int:
        """Parameters introduced by this wrapper, excluding every base weight."""
        if self.mode == "plain":
            return 0
        if self.mode == "scalar":
            return 1
        return sum(parameter.numel() for parameter in self.gate.parameters())

    def _coefficients(self, features: Tensor, gate_mode: str, permutation_seed: int):
        length = len(features)
        projected = None
        if self.mode == "dynamic" and gate_mode != "unit":
            # Evidence is immutable; only the new projection/gate is trainable.
            projected = self.gate.projection(features.detach())
        coefficients: dict[str, dict[int, Tensor]] = {}
        entries = []
        for path in self.temporal_paths:
            coefficients[path] = {}
            for offset in self._offsets[path]:
                start, end = max(0, -offset), min(length, length - offset)
                targets = torch.arange(start, max(start, end), device=features.device)
                if gate_mode == "unit":
                    values = features.new_ones(len(targets))
                elif self.mode == "scalar":
                    values = (2 * torch.sigmoid(self.scalar_logit)).expand(len(targets))
                else:
                    values = self.gate(projected, targets, targets + offset, offset / self._max_offset)
                original = values
                shuffled_positions = 0
                if gate_mode == "permuted" and len(values) > 1:
                    encoded = f"{permutation_seed}:{path}:{offset}".encode("utf-8")
                    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63 - 1)
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                    order = torch.randperm(len(values), generator=generator).to(features.device)
                    shuffled_positions = int((order != torch.arange(len(values), device=features.device)).sum())
                    values = values.index_select(0, order)
                # Invalid padded positions remain unit strength and are never
                # evaluated by the gate network as invented visual neighbors.
                full = features.new_ones(length).to(dtype=values.dtype)
                coefficients[path][offset] = full.index_copy(0, targets, values)
                entries.append({"path": path, "offset": offset, "values": values.detach().clone(),
                                "target_positions": targets.detach().clone(),
                                "original_values": original.detach().clone(),
                                "shuffled_positions": shuffled_positions})
        return coefficients, entries

    def forward(self, features: Tensor, a_logits: Tensor, *, gate_mode: str = "learned",
                permutation_seed: int = 0) -> Tensor:
        self._last_coefficients = []
        self._last_called_paths = []
        self._last_gate_mode = None
        if gate_mode not in {"learned", "unit", "permuted"}:
            raise ValueError("gate_mode must be learned, unit or permuted")
        if type(permutation_seed) is not int or permutation_seed < 0:
            raise ValueError("permutation_seed must be a nonnegative integer")
        if not isinstance(features, Tensor) or not isinstance(a_logits, Tensor):
            raise ValueError("features and A logits must be tensors")
        if features.ndim != 2 or features.shape[0] <= 0 or features.shape[1] != self.feature_dim:
            raise ValueError("features must have nonempty shape [T, feature_dim]")
        if a_logits.shape != (len(features), 7):
            raise ValueError("A logits must have shape [T,7]")
        if not features.is_floating_point() or not a_logits.is_floating_point():
            raise ValueError("features and A logits must be floating point")
        if features.device != a_logits.device:
            raise ValueError("features and A logits must be on the same device")
        if not torch.isfinite(features).all().item() or not torch.isfinite(a_logits).all().item():
            raise ValueError("features and A logits must be finite")
        if self._context.active:
            raise RuntimeError("ContextUtilityTCN cannot be used reentrantly or concurrently")
        self._last_gate_mode = gate_mode
        try:
            if self.mode == "plain":
                output = self.base(features, a_logits)
            else:
                coefficients, entries = self._coefficients(features, gate_mode, permutation_seed)
                self._context.coefficients = coefficients
                self._context.length = len(features)
                self._context.called_paths = []
                self._context.active = True
                output = self.base(features, a_logits)
                if len(features) > 1 and set(self._context.called_paths) != set(self.temporal_paths):
                    raise ValueError("some declared temporal_paths were not called by this base")
                # A singleton may skip every convolution, preserving the base's
                # original fallback rather than forcing a new A-only policy.
                called = set(self._context.called_paths)
                self._last_coefficients = [entry for entry in entries if entry["path"] in called]
                self._last_called_paths = list(self._context.called_paths)
            if not isinstance(output, Tensor) or output.shape != a_logits.shape:
                raise ValueError("base must return logits with the same [T,7] shape as A")
            if output.device != a_logits.device or not output.is_floating_point():
                raise ValueError("base output must be floating point on the A device")
            if not torch.isfinite(output).all().item():
                raise ValueError("base returned nonfinite logits")
            return output
        except Exception:
            self._last_coefficients = []
            self._last_called_paths = []
            self._last_gate_mode = None
            raise
        finally:
            self._context.active = False
            self._context.length = 0
            self._context.coefficients = {}
            self._context.called_paths = []

    def last_coefficients(self) -> list[dict[str, Any]]:
        """Detached diagnostic tensors; local positions are not public IDs."""
        return [{key: value.clone() if isinstance(value, Tensor) else value
                 for key, value in entry.items()} for entry in self._last_coefficients]

    def last_gate_audit(self) -> dict[str, Any]:
        """JSON-safe aggregates with no visual evidence, identities or positions."""
        groups = []
        for entry in self._last_coefficients:
            values = entry["values"].double()
            original = entry["original_values"].double()
            count = values.numel()
            groups.append({"path": entry["path"], "offset": entry["offset"],
                           "direction": "left" if entry["offset"] < 0 else "right",
                           "count": count, "mean": float(values.mean()) if count else None,
                           "min": float(values.min()) if count else None,
                           "max": float(values.max()) if count else None,
                           "sum": float(values.sum()), "sum_sq": float(values.square().sum()),
                           "changed_positions_permuted": int((values != original).sum()),
                           "shuffled_positions": entry["shuffled_positions"]})
        return {"mode": self.mode, "gate_mode": self._last_gate_mode,
                "coefficient_range": [0, 2], "groups": groups,
                "total_coefficients": sum(group["count"] for group in groups),
                "changed_positions_permuted": sum(group["changed_positions_permuted"] for group in groups),
                "shuffled_positions": sum(group["shuffled_positions"] for group in groups),
                "called_paths": list(self._last_called_paths)}
