"""Native classifiers on frozen new A features, never old FSN evidence.

MS-TCN++/ASFormer use the pinned author models with documented clip-level
adaptations. Same-length grouping prevents padding or another segment from
changing context/instance normalization. BiGRU reads each legal chain only.
"""
from __future__ import annotations

import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .vendor.mstcn2 import MS_TCN2
from .vendor.asformer import MyTransformer

CONFIG = json.loads(Path(__file__).with_name("config.json").read_text())
VARIANTS = tuple(CONFIG["variants"])


class BiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        c = CONFIG["bigru"]
        self.stem = nn.Sequential(nn.Linear(1543, c["hidden"]), nn.LayerNorm(c["hidden"]), nn.GELU())
        self.gru = nn.GRU(c["hidden"], c["hidden"], c["layers"], batch_first=True,
                          bidirectional=True, dropout=c["dropout"])
        self.classifier = nn.Linear(2*c["hidden"], 7)

    def forward(self, x):
        hidden, _ = self.gru(self.stem(x))
        return self.classifier(hidden).transpose(1, 2)[None]


class TemporalHead(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("unregistered native temporal head")
        self.variant = variant
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(CONFIG["seed"])
            if variant == "bigru":
                self.head = BiGRU()
            elif variant == "mstcnpp":
                c = CONFIG[variant]
                self.head = MS_TCN2(c["PG_layers"],c["refinement_layers"],c["refinements"],c["hidden"],1543,7)
            else:
                c = CONFIG[variant]
                self.head = MyTransformer(c["decoders"],c["layers"],c["r1"],c["r2"],c["hidden"],1543,7,c["channel_mask_rate"])

    def forward(self, features, a_logits, valid, mode="native", *, all_stages=False):
        if mode not in {"native", "center_only"}:
            raise ValueError("native heads have no fabricated zero-correction/A fallback mode")
        if features.ndim != 3 or features.shape[-1] != 1543 or a_logits.shape != (*features.shape[:2],7) \
                or valid.shape != features.shape[:2] or valid.dtype != torch.bool or not bool(valid.any(1).all()):
            raise ValueError("invalid frozen input shapes or empty segments")
        lengths = valid.sum(1)
        if not torch.equal(valid, torch.arange(valid.shape[1],device=valid.device)[None] < lengths[:,None]):
            raise ValueError("valid positions must be contiguous left-aligned segments")
        if not bool(torch.isfinite(features).all() and torch.isfinite(a_logits).all()) \
                or not torch.equal(features[...,-7:][valid],a_logits[valid]):
            raise ValueError("finite paired raw A logits required")
        features = features.detach()
        stages = features.new_zeros((1 if self.variant == "bigru" else 4,*features.shape[:2],7))
        if mode == "center_only":
            isolated = features[valid][:,None,:]
            v = torch.ones(len(isolated),1,dtype=torch.bool,device=features.device)
            y = self.forward(isolated,isolated[...,-7:],v,all_stages=True)
            stages[:,valid] = y[:,:,0]
        else:
            for length in sorted(set(lengths.tolist())):
                rows = torch.where(lengths == length)[0]
                x = features[rows,:length]
                if self.variant == "bigru":
                    y = self.head(x)
                elif self.variant == "mstcnpp":
                    y = self.head(x.transpose(1,2))
                else:
                    mask = x.new_ones((len(rows),1,length))
                    y = self.head(x.transpose(1,2),mask)
                if y.shape != (stages.shape[0],len(rows),7,length):
                    raise RuntimeError("native stage shape changed")
                stages[:,rows,:length] = y.permute(0,1,3,2)
        if not bool(torch.isfinite(stages).all()):
            raise RuntimeError("nonfinite temporal output")
        return stages if all_stages else stages[-1]

    def parameter_counts(self):
        return {"trainable":sum(p.numel() for p in self.parameters()), "A_trainable":0,
                "classification":"native_final_stage", "capacity_matched_to_TCN":False}


def objective(stages, targets, valid):
    if stages.ndim != 4 or stages.shape[1:3] != targets.shape or stages.shape[-1] != 7 \
            or valid.shape != targets.shape or valid.dtype != torch.bool or not bool(valid.any()) \
            or not bool(torch.isfinite(stages).all()) or stages.shape[0] not in (1,4):
        raise ValueError("finite registered nonempty loss shapes required")
    ce, smooth = stages.sum()*0, stages.sum()*0
    pair = valid[:,1:] & valid[:,:-1]
    for pred in stages:
        ce = ce + F.cross_entropy(pred[valid],targets[valid])
        if len(stages)>1 and bool(pair.any()):
            log = F.log_softmax(pred,dim=-1)
            error = (log[:,1:]-log[:,:-1].detach()).square().clamp(0,CONFIG["multistage_loss"]["log_difference_square_cap"])
            smooth = smooth + error[pair].mean()
    loss = ce + CONFIG["multistage_loss"]["smooth_weight"]*smooth
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("nonfinite native temporal objective")
    return loss,ce,smooth


def select_model(metrics):
    order = CONFIG["selection_order"]
    if set(metrics) != set(order):
        raise ValueError("all five completed replayed candidates are required")
    for value in metrics.values():
        if any(not isinstance(value[k],(int,float)) or not 0 <= value[k] <= 1 for k in ("macro_f1","accuracy")):
            raise ValueError("finite fraction metrics required")
    return max(order,key=lambda name:metrics[name]["macro_f1"])
