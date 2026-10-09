"""Separate a TCN's center-only correction from its incremental context.

This is an offline, candidate-chain model, not an edit-boundary detector.
Both passes use the same parameters AND dropout mask. A singleton's contextual
increment is exactly zero. No source, identity or action truth enters forward.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from experiments.direction_gate.model import confidence
from experiments.relation.tcn_baseline import FrozenEvidenceTCN, TCNConfig

VARIANTS = ("clip_only_matched", "legacy_continue", "legacy_aug",
            "context_delta_continue", "context_delta_aug", "context_delta_gate_aug")
CONTEXT_VARIANTS = frozenset(VARIANTS[3:])
AUGMENTED = frozenset(("legacy_aug", "context_delta_aug", "context_delta_gate_aug"))


class ContextEvidenceTCN(nn.Module):
    def __init__(self, config, variant, statistics=None):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("unknown context-evidence variant")
        self.variant = variant
        if variant == "clip_only_matched":
            # Width 74 is selected by parameter matching, NOT by val performance:
            # 148229 versus the ordinary width64/kernel3 model's 148935 (-0.47%).
            config = TCNConfig(config.global_dim, config.local_dim, 74, 3, config.dropout)
        self.base = FrozenEvidenceTCN(config)
        if variant == "clip_only_matched":
            for block in self.base.blocks:
                block.conv = nn.Conv1d(config.hidden_dim, config.hidden_dim, 1)
        self.config = config
        if variant in CONTEXT_VARIANTS:
            # This bias cancels algebraically; do not pretend it is trained.
            self.base.classifier.bias.requires_grad_(False)
        if variant == "context_delta_gate_aug":
            if statistics is None:
                raise ValueError("gated context requires train-only confidence statistics")
            mean = torch.tensor(statistics["q_mean"], dtype=torch.float32)
            scale = torch.tensor(statistics["q_scale"], dtype=torch.float32)
            if mean.shape != (6,) or scale.shape != (6,) or not bool(
                    torch.isfinite(mean).all() and torch.isfinite(scale).all() and (scale > 0).all()):
                raise ValueError("invalid train confidence statistics")
            self.register_buffer("q_mean", mean)
            self.register_buffer("q_scale", scale)
            self.gate_bias = nn.Parameter(torch.zeros(()))
            self.gate_weight = nn.Parameter(torch.zeros(6))

    def load_old(self, state):
        if self.variant == "clip_only_matched":
            raise ValueError("the different-width clip-only control is initialized fresh")
        self.base.load_state_dict(state, strict=True)

    def streams(self, features, a_logits, valid):
        if (features.ndim != 3 or features.shape[-1] != self.config.global_dim + self.config.local_dim + 7
                or a_logits.shape != (*features.shape[:2], 7) or valid.shape != features.shape[:2]
                or valid.dtype != torch.bool or not bool(valid.any(1).all())):
            raise ValueError("invalid frozen-evidence shapes or valid mask")
        if not bool(torch.isfinite(features).all() and torch.isfinite(a_logits).all()):
            raise ValueError("nonfinite frozen evidence")
        full = self.base.stem(features).transpose(1, 2) * valid[:, None, :]
        center = full
        for block in self.base.blocks:
            activation = F.gelu(block.conv(full))
            # The exact single-clip counterfactual: only the central kernel tap,
            # original bias, activation, pointwise mix and residual remain.
            weight = block.conv.weight[:, :, block.conv.kernel_size[0] // 2:][:, :, :1]
            center_activation = F.gelu(F.conv1d(center, weight, block.conv.bias))
            mask = F.dropout(torch.ones_like(activation), block.dropout.p, self.training)
            full = (full + block.mix(activation * mask)) * valid[:, None, :]
            center = (center + block.mix(center_activation * mask)) * valid[:, None, :]
        full_logits = a_logits + self.base.classifier(full).transpose(1, 2) * valid[:, :, None]
        center_logits = a_logits + self.base.classifier(center).transpose(1, 2) * valid[:, :, None]
        # Bias cancels. Calculate the increment stably without subtracting two
        # large classification outputs. Singleton rows are explicitly exact A.
        delta = F.conv1d(full - center, self.base.classifier.weight).transpose(1, 2)
        delta = delta * valid[:, :, None]
        delta = torch.where((valid.sum(1) == 1)[:, None, None], torch.zeros_like(delta), delta)
        return full_logits, center_logits, delta

    def forward(self, features, a_logits, valid, mode="learned", return_details=False):
        if mode not in {"learned", "unit", "zero"}:
            raise ValueError("unknown coefficient audit mode")
        if self.variant not in CONTEXT_VARIANTS and not return_details:
            # Real baseline training/inference must not pay for an unused
            # center-counterfactual pass merely to favor the proposed method.
            return self.base(features, a_logits, valid)
        full, center, delta = self.streams(features, a_logits, valid)
        coefficient = a_logits.new_ones(valid.shape)
        if self.variant not in CONTEXT_VARIANTS:
            output = full
        else:
            if mode == "zero":
                coefficient = torch.zeros_like(coefficient)
            elif self.variant == "context_delta_gate_aug" and mode == "learned":
                candidate = a_logits + delta
                # Gate inputs are detached scores of this target clip. TCN
                # weights still get gradients through the contextual delta.
                q = confidence(a_logits.reshape(-1, 7).detach(), candidate.reshape(-1, 7).detach())
                q = (q - self.q_mean) / self.q_scale
                coefficient = (2 * (self.gate_bias + q @ self.gate_weight).sigmoid()).reshape(valid.shape)
            output = a_logits + coefficient[:, :, None] * delta
        if return_details:
            return output, {"full": full, "center": center, "context_delta": delta,
                            "coefficient": coefficient}
        return output

    @property
    def receptive_field_clips(self):
        return 1 if self.variant == "clip_only_matched" else self.config.receptive_field_clips
