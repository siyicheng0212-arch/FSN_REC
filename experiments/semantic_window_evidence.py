"""Opt-in semantic evidence corroboration within one video clip.

This module reuses the per-frame seven-class heads that Original Uni-AdaFocus
already evaluates. Its streams are class evidence, not clinician/patient labels.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def time_bin_margins(
    frame_logits: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sweep-minus-reperfusion margins and sample counts in thirds."""
    if frame_logits.ndim != 3 or frame_logits.shape[-1] != 7:
        raise ValueError("frame_logits must have shape [B,T,7]")
    if positions.shape != frame_logits.shape[:2]:
        raise ValueError("positions must have shape [B,T]")
    margins = frame_logits[..., 3].float() - frame_logits[..., 4].float()
    values, counts = [], []
    for index in range(3):
        lower, upper = index / 3, (index + 1) / 3
        mask = (positions >= lower) & (
            positions <= upper if index == 2 else positions < upper
        )
        count = mask.sum(dim=1)
        values.append(
            (margins * mask.float()).sum(dim=1) / count.clamp_min(1).float()
        )
        counts.append(count)
    return torch.stack(values, dim=1), torch.stack(counts, dim=1)


def frame_quality(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Detect exact-ish repeats and very large jumps; not a clinical cut label."""
    if frames.ndim != 5 or frames.shape[2] != 3:
        raise ValueError("frames must have shape [B,T,3,H,W]")
    batch, time = frames.shape[:2]
    if time < 2:
        raise ValueError("at least two frames are required")
    small = F.adaptive_avg_pool2d(
        frames.reshape(batch * time, *frames.shape[2:]).float(), 16
    ).reshape(batch, time, 3, 16, 16)
    change = (small[:, 1:] - small[:, :-1]).abs().mean(dim=(2, 3, 4))
    distinct_fraction = (change > 1e-4).float().mean(dim=1)
    large_jump = (change > 0.45).any(dim=1)
    return distinct_fraction, large_jump


class SemanticWindowCorroborator(nn.Module):
    """Learn a bounded pair-margin correction from time-aligned class votes."""

    def __init__(self, *, corroboration: bool = True, correction_cap: float = 4.0):
        super().__init__()
        if correction_cap <= 0:
            raise ValueError("correction_cap must be positive")
        self.corroboration = corroboration
        self.correction_cap = correction_cap
        self.calibrator = nn.Sequential(
            nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1)
        )
        nn.init.zeros_(self.calibrator[-1].weight)
        nn.init.zeros_(self.calibrator[-1].bias)
        # Exact Original equivalence at initialization. Pair supervision
        # separately trains the calibrator, avoiding a double-zero bottleneck.
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        global_logits: torch.Tensor,
        local_logits: torch.Tensor,
        global_positions: torch.Tensor,
        local_positions: torch.Tensor,
        baseline_logits: torch.Tensor,
        frames: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if baseline_logits.ndim != 2 or baseline_logits.shape[1] != 7:
            raise ValueError("baseline_logits must have shape [B,7]")
        if global_logits.shape[0] != baseline_logits.shape[0]:
            raise ValueError("global batch size mismatch")
        if local_logits.shape[0] != baseline_logits.shape[0]:
            raise ValueError("local batch size mismatch")
        if frames.shape[0] != baseline_logits.shape[0]:
            raise ValueError("frame batch size mismatch")

        global_bins, global_counts = time_bin_margins(
            global_logits, global_positions
        )
        local_bins, local_counts = time_bin_margins(
            local_logits, local_positions
        )
        # A missing local view cannot be treated as zero evidence.
        bins = global_bins + torch.where(
            local_counts > 0, local_bins, torch.zeros_like(local_bins)
        )
        baseline_margin = (
            baseline_logits[:, 3].float() - baseline_logits[:, 4].float()
        )
        quality, large_jump = frame_quality(frames)
        features = torch.cat(
            (
                bins,
                local_counts.float() / local_logits.shape[1],
                baseline_margin[:, None],
                quality[:, None],
            ),
            dim=1,
        )
        robust_margin = (
            bins.median(dim=1).values
            if self.corroboration else bins.mean(dim=1)
        )
        proposal = robust_margin + self.calibrator(features).squeeze(1)
        valid = (quality >= 0.15) & ~large_jump
        if self.corroboration:
            positive = (bins > 0.5).sum(dim=1) >= 2
            negative = (bins < -0.5).sum(dim=1) >= 2
            valid = valid & (positive | negative)
        proposed_change = ((proposal - baseline_margin) / 2).clamp(
            -self.correction_cap, self.correction_cap
        )
        margin_correction = self.alpha * proposed_change * valid.float()
        correction = F.pad(
            torch.stack((margin_correction, -margin_correction), dim=1),
            (3, 2),
        )
        return {
            "correction": correction,
            "proposal_margin": proposal,
            "bin_margins": bins,
            "global_bin_counts": global_counts,
            "local_bin_counts": local_counts,
            "distinct_fraction": quality,
            "large_jump": large_jump,
            "corroboration_gate": valid,
            "alpha": self.alpha,
        }


class FSNSemanticWindowEvidence(nn.Module):
    """Frozen trained Original plus an opt-in, same-clip semantic correction."""

    model_name = "fsn_semantic_window_evidence"

    def __init__(
        self, baseline: nn.Module, *, corroboration: bool = True,
        pair_aux_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if pair_aux_weight < 0:
            raise ValueError("pair_aux_weight must be non-negative")
        self.baseline = baseline
        self.evidence_module = SemanticWindowCorroborator(
            corroboration=corroboration
        )
        self.pair_aux_weight = pair_aux_weight
        self.enabled = True
        self.register_buffer("class_weights", None)
        self.register_buffer("pair_weights", None)
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        self.baseline.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()
        return self

    def set_class_weights(
        self, class_weights: torch.Tensor, pair_weights: torch.Tensor
    ) -> None:
        if class_weights.shape != (7,) or pair_weights.shape != (2,):
            raise ValueError("expected seven class weights and two pair weights")
        self.class_weights = class_weights.detach().clone()
        self.pair_weights = pair_weights.detach().clone()

    def forward(self, frames: torch.Tensor) -> dict[str, Any]:
        if frames.ndim != 5 or frames.shape[1:3] != (36, 3):
            raise ValueError("expected RGB frames [B,36,3,H,W]")
        captured: dict[str, list[torch.Tensor]] = {"global": [], "local": []}

        def capture(name: str):
            def hook(_module, _inputs, output):
                captured[name].append(output)
            return hook

        handles = [
            self.baseline.core.global_CNN.new_fc.register_forward_hook(
                capture("global")
            ),
            self.baseline.core.local_CNN.new_fc.register_forward_hook(
                capture("local")
            ),
        ]
        try:
            with torch.no_grad():
                original = self.baseline(frames)
        finally:
            for handle in handles:
                handle.remove()
        if len(captured["global"]) != 1 or len(captured["local"]) != 1:
            raise RuntimeError("Original did not produce exactly one global/local vote tensor")

        batch = frames.shape[0]
        glance_count = self.baseline.num_glance_segments
        focus_count = self.baseline.num_focus_segments
        global_votes = captured["global"][0].reshape(batch, glance_count, 7)
        local_votes = captured["local"][0].reshape(batch, focus_count, 7)
        input_count = self.baseline.num_input_focus_segments
        frame_count = frames.shape[1]
        global_positions = torch.linspace(
            0, frame_count - 1, glance_count, device=frames.device
        ).round() / (frame_count - 1)
        global_positions = global_positions[None].expand(batch, -1)
        input_positions = torch.linspace(
            0, frame_count - 1, input_count, device=frames.device
        ).round() / (frame_count - 1)
        focus_indices = original["eval_outputs"][8].long().reshape(-1)
        if focus_indices.numel() != batch * focus_count:
            raise RuntimeError("unexpected focus_indices shape")
        local_positions = input_positions[None].expand(
            batch, -1
        ).reshape(-1).index_select(0, focus_indices).reshape(batch, focus_count)

        evidence = self.evidence_module(
            global_votes,
            local_votes,
            global_positions,
            local_positions,
            original["logits"],
            frames,
        )
        baseline_logits = original["logits"].float()
        correction = (
            evidence["correction"]
            if self.enabled else torch.zeros_like(evidence["correction"])
        )
        return {
            "logits": baseline_logits + correction,
            "baseline_logits": baseline_logits,
            "semantic_correction": correction,
            "semantic_evidence": evidence,
        }

    def compute_loss(
        self, output: dict[str, Any], target: torch.Tensor
    ) -> torch.Tensor:
        loss = F.cross_entropy(
            output["logits"], target, weight=self.class_weights
        )
        pair = (target == 3) | (target == 4)
        if self.pair_aux_weight and bool(pair.any()):
            proposal = output["semantic_evidence"]["proposal_margin"][pair]
            pair_logits = torch.stack((proposal / 2, -proposal / 2), dim=1)
            loss = loss + self.pair_aux_weight * F.cross_entropy(
                pair_logits, target[pair] - 3, weight=self.pair_weights
            )
        return loss
