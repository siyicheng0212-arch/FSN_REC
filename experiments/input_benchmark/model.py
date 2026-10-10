"""Native TCN and attributed mature heads on the same sealed clip evidence."""
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.input_evidence.model import InputEvidenceTCN
from experiments.relation.tcn_baseline import TCNConfig
from experiments.temporal_heads.vendor.mstcn2 import MS_TCN2
from experiments.temporal_heads.vendor.asformer import MyTransformer

CONFIG_PATH = Path(__file__).with_name("config.json")
CONFIG = json.loads(CONFIG_PATH.read_text())

# Deliberately finite registry: no architecture/seed search after seeing val.
TASKS = {
    "logits": {"head": "native_tcn", "scores": "raw", "augmented": False, "loss": "final_ce"},
    "logits_aug": {"head": "native_tcn", "scores": "raw", "augmented": True, "loss": "final_ce"},
    "visual": {"head": "native_tcn", "scores": "zero", "augmented": False, "loss": "final_ce"},
    "visual_aug": {"head": "native_tcn", "scores": "zero", "augmented": True, "loss": "final_ce"},
    "clip_logits": {"head": "center_only", "scores": "raw", "augmented": False, "loss": "final_ce"},
    "clip_visual": {"head": "center_only", "scores": "zero", "augmented": False, "loss": "final_ce"},
    "mstcnpp_logits": {"head": "mstcnpp", "scores": "raw", "augmented": False, "loss": "stage_ce_smooth"},
    "mstcnpp_visual": {"head": "mstcnpp", "scores": "zero", "augmented": False, "loss": "stage_ce_smooth"},
    "asformer_logits": {"head": "asformer", "scores": "raw", "augmented": False, "loss": "stage_ce_smooth"},
    "asformer_visual": {"head": "asformer", "scores": "zero", "augmented": False, "loss": "stage_ce_smooth"},
    "mstcnpp_logits_final_ce": {"head": "mstcnpp", "scores": "raw", "augmented": False, "loss": "final_ce"},
    "asformer_logits_final_ce": {"head": "asformer", "scores": "raw", "augmented": False, "loss": "final_ce"}
}


def specification(task):
    if task not in TASKS:
        raise ValueError("task is not in the preregistered finite comparison matrix")
    return deepcopy(TASKS[task])


class BenchmarkHead(nn.Module):
    def __init__(self, config: TCNConfig, task, seed=42):
        super().__init__()
        self.task, self.spec, self.config = task, specification(task), config
        self.input_dim = config.global_dim + config.local_dim + 7
        # Keep RNG independent of data sampling and identical within head family.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if self.spec["head"] in ("native_tcn", "center_only"):
                cell = "visual" if self.spec["scores"] == "zero" else "logits"
                self.core = InputEvidenceTCN(config, cell, seed=seed)
            elif self.spec["head"] == "mstcnpp":
                c = CONFIG["mstcnpp"]
                self.core = MS_TCN2(c["PG_layers"], c["refinement_layers"], c["refinements"], c["hidden"], self.input_dim, 7)
            else:
                c = CONFIG["asformer"]
                self.core = MyTransformer(c["decoders"], c["layers"], c["r1"], c["r2"], c["hidden"], self.input_dim, 7, c["channel_mask_rate"])

    def effective_inputs(self, features, a_logits, valid):
        if features.ndim != 3 or features.shape[-1] != self.input_dim or \
                a_logits.shape != (*features.shape[:2], 7) or valid.shape != features.shape[:2] or \
                valid.dtype != torch.bool or not bool(valid.any(1).all()):
            raise ValueError("finite sealed feature, score and valid-mask shapes required")
        lengths = valid.sum(1)
        if not torch.equal(valid, torch.arange(valid.shape[1], device=valid.device)[None] < lengths[:, None]):
            raise ValueError("only contiguous left-aligned candidate segments are supported")
        if not bool(torch.isfinite(features).all() and torch.isfinite(a_logits).all()) or \
                not torch.equal(features[..., -7:], a_logits):
            raise ValueError("raw evidence including donor scores must be finite and exactly paired")
        if self.spec["scores"] == "zero":
            a_logits = torch.zeros_like(a_logits)
            features = torch.cat((features[..., :-7], a_logits), -1)
        return features, a_logits

    def forward(self, features, a_logits, valid, *, all_stages=False):
        features, a_logits = self.effective_inputs(features, a_logits, valid)
        if self.spec["head"] in ("native_tcn", "center_only"):
            mode = "zero" if self.spec["head"] == "center_only" else "learned"
            stages = self.core(features, a_logits, valid, mode=mode)[None]
        else:
            # Never let padding or another recording alter normalization/attention.
            stages = features.new_zeros((4, *features.shape[:2], 7))
            lengths = valid.sum(1)
            for length in sorted(set(lengths.tolist())):
                rows = torch.where(lengths == length)[0]
                x = features[rows, :length].transpose(1, 2)
                mask = torch.ones((len(rows), 1, length), device=x.device, dtype=x.dtype)
                y = self.core(x) if self.spec["head"] == "mstcnpp" else self.core(x, mask)
                if y.shape != (4, len(rows), 7, length):
                    raise RuntimeError("attributed upstream head changed its stage layout")
                stages[:, rows, :length] = y.permute(0, 1, 3, 2)
        if not bool(torch.isfinite(stages).all()):
            raise RuntimeError("nonfinite classifier output")
        return stages if all_stages else stages[-1]

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        unused = 7 * self.config.hidden_dim if self.spec["scores"] == "zero" and \
            self.spec["head"] in ("native_tcn", "center_only") else 0
        if self.spec["head"] == "center_only":
            unused += sum(b.conv.weight[:, :, [0, 2]].numel() for b in self.core.base.blocks)
        return {"parameters": total, "trainable_parameters": total, "A_parameters_trained": 0,
                "unused_native_weight_elements": unused,
                "allocated_counts_do_not_imply_equal_effective_capacity": True,
                "singleton_classifier_not_forced_A": True, "task_specification": self.spec}


def objective(stages, targets, valid, spec, positions=None):
    if stages.ndim != 4 or stages.shape[1:3] != targets.shape or valid.shape != targets.shape or \
            valid.dtype != torch.bool or not bool(torch.isfinite(stages).all()):
        raise ValueError("classification loss shapes/nonfinite values invalid")
    if spec.get("loss") not in ("final_ce", "stage_ce_smooth"):
        raise ValueError("unregistered loss recipe")
    selected = valid.clone()
    if positions is not None:
        positions = torch.as_tensor(positions, device=targets.device)
        if positions.dtype != torch.int64 or positions.shape != (len(targets),) or bool((positions < 0).any()) or bool((positions >= targets.shape[1]).any()):
            raise ValueError("one valid preserved anchor per view required")
        selected.zero_()
        selected[torch.arange(len(targets), device=targets.device), positions] = True
        if bool((selected & ~valid).any()) or not torch.equal(targets >= 0, selected):
            raise ValueError("donor positions cannot receive original labels")
    if not bool(selected.any()) or bool(((targets[selected] < 0) | (targets[selected] >= 7)).any()):
        raise ValueError("only valid seven-class targets may be supervised")
    used = stages[-1:] if spec["loss"] == "final_ce" else stages
    ce = sum(F.cross_entropy(z[selected], targets[selected]) for z in used)
    smooth = stages.sum() * 0
    # Do not smooth across intentionally replaced neighbors, or supervise donors.
    if positions is None and spec["loss"] == "stage_ce_smooth":
        pairs = valid[:, 1:] & valid[:, :-1]
        if bool(pairs.any()):
            for z in stages:
                log = z.log_softmax(-1)
                error = (log[:, 1:] - log[:, :-1].detach()).square().clamp(0, CONFIG["multistage_loss"]["log_difference_square_cap"])
                smooth = smooth + error[pairs].mean()
    loss = ce + CONFIG["multistage_loss"]["smooth_weight"] * smooth
    return loss, ce, smooth
