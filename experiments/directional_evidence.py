"""Opt-in, within-clip directional evidence for the sweep/reperfusion pair.

Both evidence streams are *latent*. The seven-way clip labels do not identify
clinician-hand or patient-body activity, so the streams must not be presented
as anatomical/clinical role detectors without separately audited labels.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from experiments.model_wrappers import AdaFocusFSN


def summarize_window_activity(activity: torch.Tensor) -> torch.Tensor:
    """Keep both overlap and directional evidence; never force exclusivity."""
    if activity.ndim != 3 or activity.shape[1:] != (2, 9):
        raise ValueError("expected two activity streams over nine windows")
    first, second = activity[:, 0], activity[:, 1]
    forward_relation = (first[:, :-1] * second[:, 1:]).mean(dim=1)
    reverse_relation = (second[:, :-1] * first[:, 1:]).mean(dim=1)
    return torch.stack(
        [first.mean(1), second.mean(1), first.amax(1), second.amax(1),
         (first * second).mean(1), forward_relation, reverse_relation,
         first[:, -1] - first[:, 0], second[:, -1] - second[:, 0]],
        dim=1,
    )


class DirectionalClipEvidence(nn.Module):
    """Two non-exclusive spatial streams and directed within-clip interactions."""

    def __init__(
        self, relation_mode: str = "directed", quality_gate: bool = True,
        ambiguity_gate: bool = True,
    ) -> None:
        super().__init__()
        if relation_mode not in ("directed", "unordered"):
            raise ValueError("relation_mode must be directed or unordered")
        self.relation_mode = relation_mode
        self.quality_gate = quality_gate
        self.use_ambiguity_gate = ambiguity_gate
        self.encoder = nn.Sequential(
            nn.Conv3d(6, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.spatial_queries = nn.Parameter(torch.randn(2, 32) * 0.02)
        self.activity_weights = nn.Parameter(torch.randn(2, 32) * 0.02)
        self.activity_bias = nn.Parameter(torch.zeros(2))
        self.margin_head = nn.Sequential(nn.Linear(9, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.margin_head[-1].weight)
        nn.init.zeros_(self.margin_head[-1].bias)

    def forward(self, frames: torch.Tensor, baseline_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        if frames.ndim != 5 or frames.shape[1:] != (36, 3, 224, 224):
            raise ValueError("expected [B,36,3,224,224] RGB frames")
        if baseline_logits.shape != (frames.shape[0], 7):
            raise ValueError("expected baseline logits [B,7]")
        batch = frames.shape[0]
        small = F.interpolate(
            frames.reshape(batch * 36, 3, 224, 224), size=(56, 56), mode="area"
        ).reshape(batch, 36, 3, 56, 56)
        difference = small[:, 1:] - small[:, :-1]
        pair_change = difference.abs().mean(dim=(2, 3, 4))
        # Repeated decoded frames are not movement; exceptionally large jumps
        # are not trusted as continuous within-clip motion either.
        valid_pairs = (pair_change > 1.0 / 255.0) & (pair_change < 0.35)
        reliability = valid_pairs.float().mean(dim=1)
        signed_difference = torch.cat([torch.zeros_like(difference[:, :1]), difference], dim=1)
        encoded = self.encoder(
            torch.cat([small, signed_difference], dim=2).permute(0, 2, 1, 3, 4)
        )
        _, channels, frames_count, height, width = encoded.shape
        spatial = encoded.permute(0, 2, 3, 4, 1).reshape(batch, frames_count, height * width, channels)
        attention_logits = torch.einsum("btpd,rd->brtp", spatial, self.spatial_queries)
        attention = attention_logits.softmax(dim=-1)
        tokens = torch.einsum("brtp,btpd->brtd", attention, spatial)
        windows = tokens.reshape(batch, 2, 9, 4, channels).mean(dim=3)
        scores = torch.einsum("brwd,rd->brw", windows, self.activity_weights)
        scores = scores + self.activity_bias[None, :, None]
        activity = scores.sigmoid()  # Independent probabilities; co-occurrence is allowed.
        summary = summarize_window_activity(activity)
        if self.relation_mode == "unordered":
            relation = (summary[:, 5] + summary[:, 6]) / 2
            summary = torch.cat((summary[:, :5], relation[:, None], relation[:, None], summary[:, 7:]), dim=1)
        # A confident visual separation needs less correction. Detaching the
        # gate prevents the new module from gaming the Original's uncertainty.
        ambiguity = torch.exp(-((baseline_logits[:, 3] - baseline_logits[:, 4]).abs() / 2.0))
        gate = reliability if self.quality_gate else torch.ones_like(reliability)
        if self.use_ambiguity_gate:
            gate = gate * ambiguity.detach()
        margin = self.margin_head(summary).squeeze(1) * gate
        correction = F.pad(torch.stack((margin, -margin), dim=1), (3, 2))
        return {
            "correction": correction,
            "temporal_reliability": reliability,
            "ambiguity_gate": ambiguity.detach(),
            "activity_window_scores": scores,
            "activity_window_probabilities": activity,
            "spatial_attention": attention,
            "forward_relation": summary[:, 5],
            "reverse_relation": summary[:, 6],
        }


class FSNDirectionalEvidence(nn.Module):
    """Original AdaFocus plus an initially zero, pair-specific correction."""

    model_name = "fsn_directional_evidence"

    def __init__(
        self, baseline: AdaFocusFSN, *, relation_mode: str = "directed",
        quality_gate: bool = True, ambiguity_gate: bool = True,
    ) -> None:
        super().__init__()
        self.baseline = baseline
        self.evidence_module = DirectionalClipEvidence(
            relation_mode, quality_gate, ambiguity_gate
        )

    @property
    def core(self):
        return self.baseline.core

    def set_class_weights(self, weights: torch.Tensor | None) -> None:
        self.baseline.set_class_weights(weights)

    def forward(self, frames: torch.Tensor) -> dict[str, Any]:
        original = self.baseline(frames)
        evidence = self.evidence_module(frames, original["logits"])
        correction = evidence["correction"]
        output = {
            **original,
            "baseline_logits": original["logits"],
            "logits": original["logits"] + correction,
            "directional_correction": correction,
            "directional_evidence": evidence,
        }
        if "policy_branch" in original:
            for name in ("random_branch", "policy_branch"):
                branch = list(original[name])
                branch[0] = branch[0] + correction
                output[name] = tuple(branch)
        return output

    def compute_loss(self, output: dict[str, Any], target: torch.Tensor) -> torch.Tensor:
        return self.baseline.compute_loss(output, target)
