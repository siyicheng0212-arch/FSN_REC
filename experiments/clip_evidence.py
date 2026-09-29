"""Opt-in, within-clip evidence aggregation on the frozen 36-frame protocol.

The original AdaFocus path is untouched. A small temporal-difference encoder
reads adjacent candidate frames that the focus policy may skip, and a
class-wise multiple-instance pool aggregates evidence from nine four-frame
windows. This is a candidate, not a claim that clinical roles are observed.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from experiments.model_wrappers import AdaFocusFSN


class IntraClipEvidence(nn.Module):
    """A small signed-frame-difference encoder with class-wise window pooling."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.window_scores = nn.Linear(32, num_classes)
        self.delta_head = nn.Linear(num_classes, num_classes)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        if frames.ndim != 5 or frames.shape[1:] != (36, 3, 224, 224):
            raise ValueError("expected [B,36,3,224,224] RGB frames")
        batch = frames.shape[0]
        small = F.interpolate(
            frames.reshape(batch * 36, 3, 224, 224),
            size=(56, 56),
            mode="area",
        ).reshape(batch, 36, 3, 56, 56)
        difference = small[:, 1:] - small[:, :-1]
        pair_change = difference.abs().mean(dim=(2, 3, 4))
        # Very short clips can have fewer than 36 distinct decoded frames.
        # Never interpret repeated frames as observed motion evidence.
        quality = (pair_change > (1.0 / 255.0)).float().mean(dim=1)
        differences = torch.cat([torch.zeros_like(difference[:, :1]), difference], dim=1)
        encoded = self.encoder(differences.permute(0, 2, 1, 3, 4))
        tokens = encoded.mean(dim=(3, 4)).transpose(1, 2)
        windows = tokens.reshape(batch, 9, 4, 32).mean(dim=2)
        scores = self.window_scores(windows)
        weights = scores.softmax(dim=1)
        pooled = (weights * scores).sum(dim=1)
        correction = self.delta_head(pooled) * quality[:, None]
        return {
            "correction": correction,
            "motion_quality": quality,
            "window_scores": scores,
            "window_weights": weights,
        }


class FSNClipEvidence(nn.Module):
    """Original Uni-AdaFocus plus an initially zero within-clip correction."""

    model_name = "fsn_clip_evidence"

    def __init__(self, baseline: AdaFocusFSN) -> None:
        super().__init__()
        self.baseline = baseline
        self.evidence_module = IntraClipEvidence(num_classes=7)

    @property
    def core(self):
        return self.baseline.core

    def set_class_weights(self, weights: torch.Tensor | None) -> None:
        self.baseline.set_class_weights(weights)

    def forward(self, frames: torch.Tensor) -> dict[str, Any]:
        original = self.baseline(frames)
        evidence = self.evidence_module(frames)
        correction = evidence["correction"]
        output = {
            **original,
            "baseline_logits": original["logits"],
            "logits": original["logits"] + correction,
            "iea_correction": correction,
            "iea_motion_quality": evidence["motion_quality"],
            "iea_window_scores": evidence["window_scores"],
            "iea_window_weights": evidence["window_weights"],
        }
        if "policy_branch" in original:
            # Preserve the original five-term training objective: replace
            # its final-path logits instead of adding another CE term.
            for name in ("random_branch", "policy_branch"):
                branch = list(original[name])
                branch[0] = branch[0] + correction
                output[name] = tuple(branch)
        return output

    def compute_loss(self, output: dict[str, Any], target: torch.Tensor) -> torch.Tensor:
        return self.baseline.compute_loss(output, target)
