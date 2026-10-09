"""Small acceptance gates. No visual feature, identity, source or truth input."""
from __future__ import annotations

import math
import torch
from torch import nn

VARIANTS = ("scalar", "class_conditioned", "direction", "direction_confidence")
Q_NAMES = ("A_margin", "TCN_margin", "A_entropy", "TCN_entropy",
           "probability_total_variation", "TCN_target_probability_change")
SHRINK_PSEUDOCOUNT = 20.0  # predeclared train-only support shrinkage, not val-tuned
INITIAL_ALPHA = 0.95


def check(a, t):
    if a.shape != t.shape or a.ndim != 2 or a.shape[1] != 7:
        raise ValueError("paired logits must be [N,7]")
    if not bool(torch.isfinite(a).all() and torch.isfinite(t).all()):
        raise ValueError("paired logits must be finite")


def confidence(a, t):
    check(a, t)
    pa, pt = a.softmax(-1), t.softmax(-1)
    topa, topt = pa.topk(2, dim=-1).values, pt.topk(2, dim=-1).values
    ea = -(pa * a.log_softmax(-1)).sum(-1) / math.log(7)
    et = -(pt * t.log_softmax(-1)).sum(-1) / math.log(7)
    target = t.argmax(-1, keepdim=True)
    return torch.stack((topa[:, 0] - topa[:, 1], topt[:, 0] - topt[:, 1], ea, et,
                        (pa - pt).abs().sum(-1) / 2,
                        (pt - pa).gather(1, target).squeeze(1)), dim=-1)


def train_statistics(a, t):
    check(a, t)
    counts = torch.bincount(a.argmax(-1) * 7 + t.argmax(-1), minlength=49).reshape(7, 7)
    q = confidence(a, t)
    return {"pair_counts": counts.tolist(), "q_mean": q.mean(0).tolist(),
            "q_scale": q.std(0, unbiased=False).clamp_min(0.05).tolist()}


class AcceptanceGate(nn.Module):
    def __init__(self, variant, statistics):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("unknown gate variant")
        self.variant = variant
        counts = torch.as_tensor(statistics["pair_counts"], dtype=torch.float32)
        mean = torch.as_tensor(statistics["q_mean"], dtype=torch.float32)
        scale = torch.as_tensor(statistics["q_scale"], dtype=torch.float32)
        if counts.shape != (7, 7) or mean.shape != (6,) or scale.shape != (6,) or \
                bool((counts < 0).any() or (scale <= 0).any()) or \
                not bool(torch.isfinite(counts).all() and torch.isfinite(mean).all()
                         and torch.isfinite(scale).all()):
            raise ValueError("invalid train-only gate statistics")
        self.register_buffer("shrink", counts / (counts + SHRINK_PSEUDOCOUNT))
        self.register_buffer("q_mean", mean)
        self.register_buffer("q_scale", scale)
        initial = math.log(INITIAL_ALPHA / (1 - INITIAL_ALPHA))
        self.bias = nn.Parameter(torch.full((1 if variant == "scalar" else 7,), initial))
        if variant in {"direction", "direction_confidence"}:
            self.pair_offset = nn.Parameter(torch.zeros(7, 7))
        if variant == "direction_confidence":
            self.confidence_weight = nn.Parameter(torch.zeros(6))

    def forward(self, a, t, mode="learned"):
        check(a, t)
        # Frozen evidence is deliberately detached, including q normalization.
        a, t = a.detach(), t.detach()
        if mode == "unit":
            return t, a.new_ones(len(a))
        if mode == "zero":
            return a, a.new_zeros(len(a))
        if mode != "learned":
            raise ValueError("unknown gate audit mode")
        ca, ct = a.argmax(-1), t.argmax(-1)
        s = self.bias.expand(len(a)) if self.variant == "scalar" else self.bias[ca]
        if self.variant in {"direction", "direction_confidence"}:
            s = s + self.shrink[ca, ct] * self.pair_offset[ca, ct]
        if self.variant == "direction_confidence":
            q = (confidence(a, t) - self.q_mean) / self.q_scale
            s = s + q @ self.confidence_weight
        alpha = s.sigmoid()
        return a + alpha[:, None] * (t - a), alpha
