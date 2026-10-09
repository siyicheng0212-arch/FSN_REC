"""Gate actual noncenter Conv1d contributions, retaining every center term.

New native-classifier experiments, NOT a replay of historical A+TCN training.
No source, IDs, clock metadata or labels are accepted by forward. The original
frozen feature vector (including seven raw A logits) is unchanged. Only the
gate's projection excludes those seven scores. Gates have no dropout, leaving
the common TCN dropout RNG schedule unchanged.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from experiments.relation.tcn_baseline import FrozenEvidenceTCN, TCNConfig

VARIANTS = ("plain", "plain_aug", "reference", "reference_aug", "scalar_aug")
AUGMENTED = frozenset(("plain_aug", "reference_aug", "scalar_aug"))


def neighbor_indices(length, offset, device):
    indices = torch.arange(length, device=device) + offset
    inside = (indices >= 0) & (indices < length)
    return indices.clamp(0, length - 1), inside


class PairReference(nn.Module):
    def __init__(self, visual_dim, projection_dim=32, hidden_dim=32):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(visual_dim, projection_dim),
                                     nn.LayerNorm(projection_dim), nn.GELU())
        self.score = nn.Sequential(nn.Linear(4 * projection_dim + 2, hidden_dim),
                                   nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, target, neighbor, offset):
        difference = neighbor - target
        cosine = F.cosine_similarity(target, neighbor, dim=-1)[..., None]
        span = target.new_full((*target.shape[:-1], 1), offset / 4.0)
        evidence = torch.cat((target, neighbor, difference, difference.abs(), cosine, span), -1)
        return 2 * self.score(evidence).squeeze(-1).sigmoid()


class AnchorReferenceTCN(nn.Module):
    def __init__(self, config: TCNConfig, variant, seed=42, projection_dim=32, gate_hidden_dim=32):
        super().__init__()
        if variant not in VARIANTS or config.blocks != 3:
            raise ValueError("use one registered variant and the verified three-block layout")
        self.config, self.variant = config, variant
        # Each variant's shared weights are byte-identical. Added modules use a
        # separate RNG stream and do not alter sampler or common dropout state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.base = FrozenEvidenceTCN(config)
            nn.init.normal_(self.base.classifier.weight, std=0.02)
            nn.init.zeros_(self.base.classifier.bias)
        if variant.startswith("reference"):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 104729)
                self.reference = PairReference(config.global_dim + config.local_dim,
                                               projection_dim, gate_hidden_dim)
        elif variant == "scalar_aug":
            self.scalar = nn.Parameter(torch.zeros(()))

    def forward(self, features, a_logits, valid, *, mode="learned", overrides=None, return_details=False):
        if mode not in ("learned", "unit", "zero"):
            raise ValueError("unregistered coefficient mode")
        if features.ndim != 3 or features.shape[-1] != self.config.global_dim + self.config.local_dim + 7 \
                or a_logits.shape != (*features.shape[:2], 7) or valid.shape != features.shape[:2] \
                or valid.dtype != torch.bool or not bool(valid.any(1).all()):
            raise ValueError("invalid frozen evidence shapes or empty segment")
        if not bool(torch.isfinite(features).all() and torch.isfinite(a_logits).all()):
            raise ValueError("nonfinite frozen evidence")
        lengths = valid.sum(1)
        if not torch.equal(valid, torch.arange(valid.shape[1], device=valid.device)[None] < lengths[:, None]):
            raise ValueError("only left-aligned contiguous valid segments are supported")
        if not torch.equal(features[..., -7:][valid], a_logits[valid]):
            raise ValueError("the A logits in the feature vector must match the paired A scores")
        hidden = self.base.stem(features).transpose(1, 2) * valid[:, None]
        projected = None
        if self.variant.startswith("reference") and mode == "learned" and overrides is None:
            projected = self.reference.project(features[..., :-7])
        coefficients, edge_masks = {}, {}
        for layer, block in enumerate(self.base.blocks):
            conv = block.conv
            if conv.kernel_size != (3,) or conv.padding != conv.dilation or conv.stride != (1,) or conv.groups != 1:
                raise RuntimeError("unsupported temporal convolution; do not silently adapt")
            # Bias occurs exactly once; the center tap is never scaled.
            update = F.conv1d(hidden, conv.weight[:, :, 1:2], conv.bias)
            for tap in (0, 2):
                offset = (tap - 1) * conv.dilation[0]
                indices, inside = neighbor_indices(hidden.shape[-1], offset, hidden.device)
                mask = valid & valid[:, indices] & inside[None]
                key = f"layer{layer}_offset{offset:+d}"
                if overrides is not None:
                    if key not in overrides or overrides[key].shape != valid.shape:
                        raise ValueError("coefficient overrides must match every actual kernel contribution")
                    coefficient = overrides[key].to(features)
                elif mode == "zero":
                    coefficient = features.new_zeros(valid.shape)
                elif mode == "unit" or self.variant.startswith("plain"):
                    coefficient = features.new_ones(valid.shape)
                elif self.variant == "scalar_aug":
                    coefficient = (2 * self.scalar.sigmoid()).expand(valid.shape)
                else:
                    coefficient = self.reference(projected, projected[:, indices], offset)
                if not bool(torch.isfinite(coefficient).all()) or bool(((coefficient < 0) | (coefficient > 2)).any()):
                    raise ValueError("invalid continuous reference coefficients")
                contribution = F.conv1d(hidden[:, :, indices], conv.weight[:, :, tap:tap+1])
                update = update + contribution * (coefficient * mask)[:, None]
                if return_details:
                    coefficients[key], edge_masks[key] = coefficient, mask
            hidden = (hidden + block.mix(block.dropout(F.gelu(update)))) * valid[:, None]
        logits = self.base.classifier(hidden).transpose(1, 2) * valid[..., None]
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("nonfinite native classifier output")
        if return_details:
            return logits, {"coefficients": coefficients, "edge_masks": edge_masks}
        return logits

    def parameter_counts(self):
        return {"parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "reference_parameters": sum(p.numel() for n,p in self.named_parameters() if not n.startswith("base.")),
                "A_parameters_trained": 0}
