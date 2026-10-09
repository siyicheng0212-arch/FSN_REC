"""Canonical classifier heads on frozen per-clip VideoMAE evidence.

No extra A residual is added to MS-TCN++/ASFormer: their final classifier is
used, as in their papers. A's raw logits remain part of the common input.
Each batch is grouped by exact sequence length, not padded inside a head.
This preserves singleton classification and prevents batch companions from
creating fake context or affecting temporal instance normalization.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.relation.tcn_baseline import FrozenEvidenceTCN, TCNConfig
from .vendor.mstcn2 import MS_TCN2
from .vendor.asformer import MyTransformer

CONFIG = json.loads(Path(__file__).with_name("config.json").read_text())
VARIANTS = tuple(CONFIG["variants"])


class TemporalHead(nn.Module):
    def __init__(self, variant, input_dim):
        super().__init__()
        if variant not in VARIANTS or input_dim != 1543:
            raise ValueError("fixed VideoMAE mean/max 768+768+7 input required")
        self.variant = variant
        if variant == "plain_tcn_fresh":
            c = CONFIG["plain_tcn"]
            self.head = FrozenEvidenceTCN(TCNConfig(768, 768, c["hidden"], c["blocks"], c["dropout"]))
        elif variant == "mstcnpp":
            c = CONFIG["mstcnpp"]
            self.head = MS_TCN2(c["PG_layers"], c["refinement_layers"], c["refinements"], c["hidden"], input_dim, 7)
        else:
            c = CONFIG["asformer"]
            self.head = MyTransformer(c["decoders"], c["layers"], c["r1"], c["r2"], c["hidden"], input_dim, 7, c["channel_mask_rate"])

    def forward(self, features, a_logits, valid, *, all_stages=False):
        if features.ndim != 3 or features.shape[-1] != 1543 or a_logits.shape != (*features.shape[:2], 7) or valid.shape != features.shape[:2]:
            raise ValueError("frozen input shape mismatch")
        if valid.dtype != torch.bool or not bool(valid.any(1).all()):
            raise ValueError("every candidate segment must contain a clip")
        lengths = valid.sum(1)
        expected = torch.arange(valid.shape[1], device=valid.device)[None, :] < lengths[:, None]
        if not torch.equal(valid, expected):
            raise ValueError("only contiguous left-aligned segment masks are supported")
        if not bool(torch.isfinite(features).all() and torch.isfinite(a_logits).all()):
            raise ValueError("nonfinite evidence")
        if self.variant == "plain_tcn_fresh":
            stages = self.head(features, a_logits, valid)[None]
        else:
            stages = features.new_zeros((4, *features.shape[:2], 7))
            for length in sorted(set(lengths.tolist())):
                rows = torch.where(lengths == length)[0]
                x = features[rows, :length].transpose(1, 2)
                mask = torch.ones((len(rows), 1, length), dtype=x.dtype, device=x.device)
                y = self.head(x) if self.variant == "mstcnpp" else self.head(x, mask)
                if y.shape != (4, len(rows), 7, length):
                    raise RuntimeError("upstream stage shape changed")
                # Assignment retains autograd to every upstream stage.
                stages[:, rows, :length] = y.permute(0, 1, 3, 2)
        if not bool(torch.isfinite(stages).all()):
            raise RuntimeError("nonfinite head logits")
        return stages if all_stages else stages[-1]


def objective(stages, targets, valid):
    """Stage CE + truncated log-probability smoothness on legal pairs only.

Unlike the upstream padded-tensor mean, pair normalization uses actual
adjacent pairs. No penalty crosses a segment, a split or a recording.
Singletons have finite zero smoothness, not an empty-tensor mean.
"""
    if stages.ndim != 4 or stages.shape[1:3] != targets.shape or valid.shape != targets.shape:
        raise ValueError("loss shape mismatch")
    if not bool(valid.any()) or not bool(torch.isfinite(stages).all()):
        raise ValueError("loss requires finite nonempty valid predictions")
    ce, smooth = stages.sum() * 0, stages.sum() * 0
    pair = valid[:, 1:] & valid[:, :-1]
    for pred in stages:
        ce = ce + F.cross_entropy(pred[valid], targets[valid])
        if len(stages) > 1 and bool(pair.any()):
            log = F.log_softmax(pred, dim=-1)
            error = (log[:, 1:] - log[:, :-1].detach()).square().clamp(0, CONFIG["multistage_loss"]["log_difference_square_cap"])
            smooth = smooth + error[pair].mean()
    loss = ce + CONFIG["multistage_loss"]["smooth_weight"] * smooth
    return loss, ce, smooth
