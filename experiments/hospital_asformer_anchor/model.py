"""Bounded neighbor-versus-self bias, with all original ASFormer paths intact."""
from __future__ import annotations

from copy import deepcopy
import torch
from torch import nn

from experiments.hospital_asformer_interval.model import (
    CONFIG as INTERVAL_CONFIG,
    IntervalTemporalHead,
    PairCalibration,
    objective,
    plateau,
)

VARIANT = "asformer_interval_self_reference"
CONFIG = deepcopy(INTERVAL_CONFIG)
CONFIG.update(
    variants=[VARIANT],
    selection_order=[*INTERVAL_CONFIG["selection_order"], VARIANT],
)
CONFIG["interval_calibration"].update(
    score_transform="tanh(pair_score_minus_same_anchor_self_score)",
    fixed_absolute_bias_bound=1.0,
    output_linear_bias=False,
    output_bias_omitted_because_shared_scalar_cancels=True,
    same_query_self_reference_before_self_mask=True,
    forced_positive_neighbor=False,
    no_validation_tuned_temperature=True,
)


def relative_bias(pair_scores: torch.Tensor) -> torch.Tensor:
    """Reference each real key to the same query's self score, not a row mean.

    A common score offset has no effect. Positive/negative corrections are
    possible, not prescribed. All-negative learned *raw* scores do not imply
    all-negative neighbor corrections. Self is exactly zero. Virtual padding
    is not scored here; the unchanged attention adapter gives it zero bias.
    """
    if pair_scores.ndim != 3 or pair_scores.shape[-1] != pair_scores.shape[-2] \
            or pair_scores.shape[-1] < 1 or not pair_scores.is_floating_point() \
            or not bool(torch.isfinite(pair_scores).all()):
        raise ValueError("finite nonempty floating real-pair scores required")
    self_score = pair_scores.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
    return torch.tanh(pair_scores - self_score)


class SelfReferencedCalibration(PairCalibration):
    def __init__(self):
        # Preserve the input descriptor network and original ASFormer RNG.
        # A final scalar bias cancels in s_ij - s_ii: do not optimize a dead
        # parameter or misreport its identically-zero gradient as a failure.
        nn.Module.__init__(self)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.network = nn.Sequential(
                nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 1, bias=False)
            )
            nn.init.zeros_(self.network[-1].weight)

    def forward(self, intervals, hidden, probabilities):
        features = self.inputs(intervals, hidden, probabilities)
        return relative_bias(self.network(features).squeeze(-1))


class AnchoredTemporalHead(IntervalTemporalHead):
    """Reuse the accepted segment-local attention implementation unchanged.

    Only calibration is replaced. The encoder, decoder Q/K/V, legal windows,
    original virtual-padding softmax mass, values and native classifiers are
    inherited exactly, as are detachment of frozen A and per-length grouping.
    """
    def __init__(self):
        super().__init__()
        self.variant = VARIANT
        self.calibration = SelfReferencedCalibration()

    def load_registered_checkpoint(self, payload, expected_identity):
        if payload.get("variant") != VARIANT or payload.get("config") != CONFIG \
                or not expected_identity or payload.get("identity") != expected_identity:
            raise ValueError("the new variant/config and fixed run identity must match")
        state = payload.get("model", {})
        if not state or any(not torch.is_tensor(v) or not bool(torch.isfinite(v).all())
                            for v in state.values()):
            raise ValueError("complete finite new-variant tensors required")
        self.load_state_dict(state, strict=True)
        return self


def preparation_status():
    """Fail-closed handoff: local model tests are not a completed GPU run."""
    return {
        "variant": VARIANT,
        "model_implemented": True,
        "execution_pipeline_migrated": True,
        "server_deployed": False,
        "CUDA_acceptance_completed": False,
        "formal_training_started": False,
        "no_new_launcher_created": True,
    }
