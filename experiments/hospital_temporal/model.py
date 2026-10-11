"""Small falsifiable T′ variants; no relation classifier or learned gate.

The established A+TCN input, stem, residual blocks and seven-logit correction
are retained. Multispan variants apply the SAME learned kernel at d and2d,
averaging before activation, with the center and bias occurring once. Relative
variants replace valid noncenter h_j contributions with GELU(h_j-h_t). The
nonlinearity matters: a linear difference alone is largely a reparameterized
ordinary convolution, not a distinct interior representation. All three
T′ models share the protected ordinary TCN's parameter count and initialization.
These are experimental adaptations, not claims of novel multiscale convolution.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.relation.tcn_baseline import FrozenEvidenceTCN, TCNConfig

CONFIG = json.loads(Path(__file__).with_name("config.json").read_text())
VARIANTS = tuple(CONFIG["variants"])
REGISTERED_VARIANTS = ("plain_tcn", "short_tcn", "clip_only", "protected_tcn",
                       "multispan_tcn", "relative_tcn", "multispan_relative_tcn")
PROTECTED = frozenset(("protected_tcn", "multispan_tcn", "relative_tcn",
                       "multispan_relative_tcn"))


def shifted(sequence, valid, offset):
    length = valid.shape[1]
    indices = torch.arange(length, device=sequence.device) + offset
    inside = (indices >= 0) & (indices < length)
    indices = indices.clamp(0, length - 1)
    mask = valid & valid[:, indices] & inside[None]
    return sequence[:, :, indices], mask


def temporal_contribution(hidden, valid, conv, dilation, relative=False, center_only=False):
    if conv.kernel_size != (3,) or conv.stride != (1,) or conv.groups != 1:
        raise ValueError("the verified width-three temporal kernel is required")
    # Keep the center and original bias exactly once, including edge positions.
    result = F.conv1d(hidden, conv.weight[:, :, 1:2], conv.bias)
    if center_only:
        return result
    for tap, offset in ((0, -dilation), (2, dilation)):
        neighbor, mask = shifted(hidden, valid, offset)
        evidence = F.gelu(neighbor - hidden) if relative else neighbor
        # Missing/padded neighbors contribute zero, not minus the current clip.
        result = result + F.conv1d(evidence * mask[:, None], conv.weight[:, :, tap:tap+1])
    return result


class TemporalVariant(nn.Module):
    def __init__(self, variant, global_dim=768, local_dim=768, hidden_dim=64,
                 dropout=0.1, seed=42):
        super().__init__()
        if variant not in REGISTERED_VARIANTS:
            raise ValueError("unregistered temporal variant")
        self.variant = variant
        width = 74 if variant == "clip_only" and hidden_dim == 64 else hidden_dim
        self.config = TCNConfig(global_dim, local_dim, width, 3, dropout)
        # Architecture additions cannot consume the sampler's random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.base = FrozenEvidenceTCN(self.config)
            if variant == "short_tcn":
                dilations = tuple(CONFIG["model"]["short_tcn_dilations"])
                if dilations != (1, 1, 1):
                    raise ValueError("the registered short TCN dilations are exactly 1/1/1")
                # Only the temporal field changes. Keep weights, initialization,
                # three blocks, padding masks and singleton behavior identical.
                for block, dilation in zip(self.base.blocks, dilations):
                    block.conv.dilation = (dilation,)
                    block.conv.padding = (dilation,)
            if variant == "clip_only":
                for block in self.base.blocks:
                    block.conv = nn.Conv1d(width, width, 1)

    def forward(self, features, a_logits, valid, mode="native"):
        if mode not in {"native", "center_only", "zero_correction"}:
            raise ValueError("unsupported diagnostic mode")
        if features.ndim != 3 or features.shape[-1] != self.config.global_dim+self.config.local_dim+7 \
                or a_logits.shape != (*features.shape[:2], 7) or valid.shape != features.shape[:2] \
                or valid.dtype != torch.bool or not bool(valid.any(1).all()):
            raise ValueError("invalid frozen input shapes or empty segments")
        lengths = valid.sum(1)
        if not torch.equal(valid, torch.arange(valid.shape[1], device=valid.device)[None] < lengths[:, None]):
            raise ValueError("valid positions must be a contiguous left-aligned segment")
        if not bool(torch.isfinite(features).all() and torch.isfinite(a_logits).all()) \
                or not torch.equal(features[..., -7:][valid], a_logits[valid]):
            raise ValueError("finite visual features and paired raw A logits are required")
        # A is immutable evidence, never a member of the temporal optimizer.
        features, a_logits = features.detach(), a_logits.detach()
        if mode == "zero_correction":
            return a_logits
        if self.variant in {"plain_tcn", "short_tcn", "clip_only", "protected_tcn"} and mode == "native":
            output = self.base(features, a_logits, valid)
        else:
            hidden = self.base.stem(features).transpose(1, 2) * valid[:, None]
            relative = self.variant in {"relative_tcn", "multispan_relative_tcn"}
            multispan = self.variant in {"multispan_tcn", "multispan_relative_tcn"}
            for block in self.base.blocks:
                if self.variant == "clip_only":
                    update = block.conv(hidden)
                else:
                    dilation = block.conv.dilation[0]
                    update = temporal_contribution(hidden, valid, block.conv, dilation,
                                                   relative, mode == "center_only")
                    if multispan and mode != "center_only":
                        longer = temporal_contribution(hidden, valid, block.conv, 2*dilation, relative)
                        update = (update + longer) * 0.5
                hidden = (hidden + block.mix(block.dropout(F.gelu(update)))) * valid[:, None]
            output = a_logits + self.base.classifier(hidden).transpose(1, 2) * valid[..., None]
        if self.variant in PROTECTED:
            output = torch.where((lengths == 1)[:, None, None], a_logits, output)
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("nonfinite temporal output")
        return output

    @property
    def receptive_field_clips(self):
        multiplier = 2 if "multispan" in self.variant else 1
        return 1 + sum((block.conv.kernel_size[0] - 1) * block.conv.dilation[0] * multiplier
                       for block in self.base.blocks)

    def parameter_counts(self):
        return {"trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "A_trainable": 0, "receptive_field_candidate_clips": self.receptive_field_clips}
