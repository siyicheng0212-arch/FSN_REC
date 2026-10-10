"""One native TCN, four registered cells, and no hidden path to raw A scores."""
import torch

from experiments.anchor_reference.model import AnchorReferenceTCN

VARIANTS = ("logits", "logits_aug", "visual", "visual_aug")
AUGMENTED = frozenset(("logits_aug", "visual_aug"))
MASKED = frozenset(("visual", "visual_aug"))


def input_policy(variant):
    if variant not in VARIANTS:
        raise ValueError("require one of the four registered diagnostic cells")
    return "seven_A_logit_slots_zero" if variant in MASKED else "seven_raw_A_logits"


class InputEvidenceTCN(AnchorReferenceTCN):
    """Reuse the exact plain native classifier, including singleton behavior.

    Seven slots stay allocated in every model. In visual cells they are zero
    at every forward, not merely during validation. Raw scores are accepted
    solely to verify sealed pairing, never used as a residual or gate input.
    Equal allocated parameters do not mean equal effective input capacity.
    """

    def __init__(self, config, variant, seed=42):
        input_policy(variant)
        super().__init__(config, "plain_aug" if variant in AUGMENTED else "plain", seed=seed)
        self.cell = variant

    def effective_inputs(self, features, a_logits):
        expected = self.config.global_dim + self.config.local_dim + 7
        if features.ndim != 3 or features.shape[-1] != expected or \
                a_logits.shape != (*features.shape[:2], 7):
            raise ValueError("frozen visual evidence and A logits have incorrect shapes")
        if not bool(torch.isfinite(features).all() and torch.isfinite(a_logits).all()):
            raise ValueError("nonfinite original evidence cannot be hidden by masking")
        # Collation produces zero padding; require exact paired raw evidence
        # before masking, so corrupt donor scores cannot silently disappear.
        if not torch.equal(features[..., -7:], a_logits):
            raise ValueError("original A logits must match all paired feature slots")
        if self.cell in MASKED:
            scores = torch.zeros_like(a_logits)
            return torch.cat((features[..., :-7], scores), dim=-1), scores
        return features, a_logits

    def forward(self, features, a_logits, valid, **kwargs):
        # No audit may inject non-unit coefficients into this ordinary TCN.
        if kwargs.get("overrides") is not None:
            raise ValueError("input diagnostic does not have coefficient overrides")
        features, a_logits = self.effective_inputs(features, a_logits)
        return super().forward(features, a_logits, valid, **kwargs)

    def parameter_counts(self):
        result = super().parameter_counts()
        result.update({"input_policy": input_policy(self.cell),
                       "A_logit_slots_allocated": 7,
                       "A_logit_slots_active": 0 if self.cell in MASKED else 7,
                       "unused_stem_weight_elements": 7 * self.config.hidden_dim if self.cell in MASKED else 0,
                       "effective_input_capacity_identical": False})
        return result
