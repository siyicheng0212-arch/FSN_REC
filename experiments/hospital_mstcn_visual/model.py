"""Original PG and R blocks; only R's input projection receives visual evidence."""
from __future__ import annotations
import copy
import torch
from torch import nn
from torch.nn import functional as F
from experiments.hospital_temporal_compare.model import CONFIG as ORIGINAL, TemporalHead, objective
from experiments.hospital_temporal_compare.vendor.mstcn2 import Prediction_Generation, Refinement

VARIANT = "mstcnpp_visual_r"
CONFIG = copy.deepcopy(ORIGINAL)
CONFIG.update(schema="fsn-hospital-mstcn-visual-refinement-v1", variants=[VARIANT],
    selection_order=[*ORIGINAL["selection_order"], VARIANT],
    refinement_input="frozen_A_visual_mean768_max768_plus_previous_stage_softmax7",
    additional_raw_A_logits_in_R=False, additional_gates_or_residuals=False,
    initialization="fresh_independent_seed42_author_constructor_order_no_trained_weights",
    parameters_matched_to_original=False, compute_matched_to_original=False)


class VisualRefinementMS(nn.Module):
    def __init__(self):
        super().__init__()
        c = CONFIG["mstcnpp"]
        self.PG = Prediction_Generation(c["PG_layers"], c["hidden"], 1543, 7)
        # Preserve the author's constructor order (each R independently initialized).
        self.Rs = nn.ModuleList([copy.deepcopy(Refinement(c["refinement_layers"],
            c["hidden"], 1543, 7)) for _ in range(c["refinements"])])

    def forward(self, x):
        out = self.PG(x)
        stages = [out]
        for refine in self.Rs:
            out = refine(torch.cat((x[:, :1536], F.softmax(out, dim=1)), dim=1))
            stages.append(out)
        return torch.stack(stages)


class VisualTemporalHead(TemporalHead):
    def __init__(self):
        nn.Module.__init__(self)
        # Inherit EXACT frozen-input validation, same-length grouping and masks.
        self.variant = "mstcnpp"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(CONFIG["seed"])
            self.head = VisualRefinementMS()

    def parameter_counts(self):
        return {**super().parameter_counts(), "R_visual_input": True,
                "parameters_matched_to_original": False, "compute_matched_to_original": False}


def select(metrics):
    order = CONFIG["selection_order"]
    if set(metrics) != set(order):
        raise ValueError("all six actually replayed registered candidates required")
    for item in metrics.values():
        for key in ("macro_f1", "accuracy"):
            value = item[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError("finite fraction development metrics required")
    return max(order, key=lambda name: metrics[name]["macro_f1"])
