"""Per-clip acceptance of a verified historical TCN's proposed correction.

This model preserves the old TCN's complete input chain, architecture, and
checkpoint.  It is a small follow-up to the old A+TCN baseline for the case
where its genuine full-chain prediction is sometimes worse than frozen A.
It does not infer continuity, learn R, introduce cuts, or treat source as an
input.  Each acceptance decision reads only that clip's own frozen A evidence
and the corresponding outputs of the old full-chain TCN.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from experiments.fsn_tcn.model import _validate_inputs


class _ClipAcceptanceGate(nn.Module):
    """Output one alpha in (0, 1) for each clip without reading its neighbors."""

    def __init__(self, feature_dim: int, hidden_dim: int, *, use_visual: bool):
        super().__init__()
        self.use_visual = use_visual
        visual_dim = min(hidden_dim, 16) if use_visual else 0
        self.visual = (nn.Sequential(nn.Linear(feature_dim, visual_dim),
                                     nn.LayerNorm(visual_dim), nn.GELU())
                       if use_visual else None)
        # Seven raw A logits, seven raw old-TCN logits, both seven-way
        # probabilities, seven absolute probability differences, A/TCN max
        # probability and entropy, and magnitude of the proposed correction.
        self.network = nn.Sequential(
            nn.Linear(visual_dim + 40, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.network[-1].weight, mean=0.0, std=0.005)
        nn.init.constant_(self.network[-1].bias, 2.0)

    def forward(self, features: Tensor, a_logits: Tensor, old_logits: Tensor) -> Tensor:
        if self.use_visual:
            visual = self.visual(features.detach().to(dtype=self.visual[0].weight.dtype))
            dtype = visual.dtype
        else:
            visual = None
            dtype = self.network[0].weight.dtype
        a_scores = a_logits.detach().to(dtype=dtype)
        old_scores = old_logits.detach().to(dtype=dtype)
        a_probability = a_scores.softmax(dim=-1)
        old_probability = old_scores.softmax(dim=-1)
        tiny = torch.finfo(a_probability.dtype).tiny
        entropy_a = -(a_probability * a_probability.clamp_min(tiny).log()).sum(-1, keepdim=True)
        entropy_old = -(old_probability * old_probability.clamp_min(tiny).log()).sum(-1, keepdim=True)
        confidence = torch.cat((a_probability.max(-1, keepdim=True).values,
                                old_probability.max(-1, keepdim=True).values), dim=-1)
        correction = (a_scores - old_scores).abs().mean(-1, keepdim=True)
        evidence = torch.cat((([visual] if visual is not None else []) +
                              [a_scores, old_scores, a_probability, old_probability,
                               (a_probability - old_probability).abs(),
                               entropy_a, entropy_old, confidence, correction]), dim=-1)
        return self.network(evidence).squeeze(-1).sigmoid()


class RevisionGateTCN(nn.Module):
    """Learn when to use an old TCN without changing its temporal messages.

    Load and verify ``base_tcn`` from its genuine checkpoint before wrapping.
    The unchanged old scorer is always run ONCE on exactly the input unit
    supplied by the historical chain-layout policy, including singleton
    units.  The caller must retain the historical ``full_chain`` versus
    ``eligible_segments`` layout; this module never changes boundaries.
    Inference uses
    ``A_logits + alpha * (old_logits - A_logits)``; ``gate_mode='unit'`` and
    ``gate_mode='zero'`` return the exact historical TCN and A tensors,
    respectively.  The gate receives no ground truth, source, edge status,
    identity, or information from other clips beyond that already contained
    in its own old-TCN output.

    ``variant='visual_logits'`` uses A features and both score distributions;
    ``logits_only`` removes visual features; ``scalar`` learns one global
    acceptance weight; ``class_conditioned`` learns seven weights selected by
    A's frozen predicted class (never by a true action label).  These four
    choices change G alone and keep the same historical TCN computation.

    ``freeze_base=True`` prevents old-TCN parameter updates and keeps it in
    eval mode even during gate training, ensuring a stable old checkpoint.
    Frozen A features and logits are supplied externally.  Returning details
    retains the alpha graph for training; callers must detach before logging.
    """

    def __init__(self, base_tcn: nn.Module, feature_dim: int,
                 gate_dim: int = 64, variant: str = "visual_logits",
                 freeze_base: bool = True):
        super().__init__()
        if not isinstance(base_tcn, nn.Module):
            raise ValueError("base_tcn must be an instantiated torch module")
        for name, value in (("feature_dim", feature_dim), ("gate_dim", gate_dim)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(freeze_base) is not bool:
            raise ValueError("freeze_base must be a boolean")
        if variant not in {"visual_logits", "logits_only", "scalar", "class_conditioned"}:
            raise ValueError("unknown revision gate variant")
        self.base_tcn = base_tcn
        self.feature_dim = feature_dim
        self.freeze_base = freeze_base
        self.variant = variant
        self.gate = (_ClipAcceptanceGate(feature_dim, gate_dim,
                                         use_visual=(variant == "visual_logits"))
                     if variant in {"visual_logits", "logits_only"} else None)
        self.register_parameter("scalar_logit", None)
        self.register_parameter("class_logits", None)
        if variant == "scalar":
            self.scalar_logit = nn.Parameter(torch.tensor(2.0))
        elif variant == "class_conditioned":
            # Condition on *A's frozen prediction*, not the action label.
            self.class_logits = nn.Parameter(torch.full((7,), 2.0))
        if freeze_base:
            self.base_tcn.requires_grad_(False)
            self.base_tcn.eval()

    def added_parameter_count(self) -> int:
        """Only newly trained parameters, excluding historical TCN weights."""
        return sum(value.numel() for name, value in self.named_parameters()
                   if not name.startswith("base_tcn."))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_base:
            self.base_tcn.eval()
        return self

    def forward(self, features: Tensor, a_logits: Tensor, *,
                gate_mode: str = "learned", return_details: bool = False
                ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        _validate_inputs(features, a_logits)
        if features.shape[1] != self.feature_dim or a_logits.shape[1] != 7:
            raise ValueError("expected features[T,feature_dim] and A logits[T,7]")
        if gate_mode not in {"learned", "unit", "zero"}:
            raise ValueError("gate_mode must be learned, unit or zero")
        old_logits = self.base_tcn(features, a_logits)
        if (not isinstance(old_logits, Tensor) or old_logits.shape != a_logits.shape
                or not old_logits.is_floating_point() or old_logits.device != a_logits.device
                or not bool(torch.isfinite(old_logits).all())):
            raise ValueError("historical TCN must return finite logits[T,7] on the A device")
        if gate_mode == "unit":
            alpha = a_logits.new_ones(len(features))
            final = old_logits  # Avoid rounding a bitwise-parity audit.
        elif gate_mode == "zero":
            alpha = a_logits.new_zeros(len(features))
            final = a_logits
        else:
            if self.variant == "scalar":
                alpha = self.scalar_logit.sigmoid().expand(len(features))
            elif self.variant == "class_conditioned":
                alpha = self.class_logits[a_logits.detach().argmax(-1)].sigmoid()
            else:
                alpha = self.gate(features, a_logits, old_logits)
            alpha = alpha.to(dtype=a_logits.dtype)
            final = a_logits + alpha.unsqueeze(-1) * (old_logits - a_logits)
        if not bool(torch.isfinite(final).all()):
            raise ValueError("revision gate returned nonfinite logits")
        if return_details:
            return final, {"alpha": alpha, "tcn_logits": old_logits}
        return final
