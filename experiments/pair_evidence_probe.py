"""Train-internal sweep/reperfusion evidence probe, not a clinical role model.

The only supervision is the existing primary seven-class clip label. The
binary probe asks whether RGB frame differences add information beyond an
equal-capacity RGB-only branch. It never infers clinician/patient activity
labels, and must not be advertised as a full seven-class FSN recognizer.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PairEvidenceProbe(nn.Module):
    """Equal-capacity RGB and RGB+signed-difference alternatives."""

    MODES = ("rgb_motion", "rgb_only", "rgb_motion_shuffled")

    def __init__(self, mode: str = "rgb_motion") -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unsupported pair-probe mode: {mode}")
        self.mode = mode
        order = torch.randperm(36, generator=torch.Generator().manual_seed(1729))
        self.register_buffer("shuffle_order", order, persistent=False)
        self.encoder = nn.Sequential(
            nn.Conv3d(6, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.spatial_score = nn.Conv3d(32, 1, kernel_size=1)
        self.temporal = nn.Sequential(nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.SiLU())
        self.temporal_score = nn.Linear(32, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(64), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 2)
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[1:] != (36, 3, 224, 224):
            raise ValueError("expected RGB clips [B,36,3,224,224]")
        batch = frames.shape[0]
        if self.mode == "rgb_motion_shuffled":
            frames = frames.index_select(1, self.shuffle_order)
        small = F.interpolate(
            frames.reshape(batch * 36, 3, 224, 224), size=(56, 56), mode="area"
        ).reshape(batch, 36, 3, 56, 56)
        difference = small[:, 1:] - small[:, :-1]
        difference = torch.cat((torch.zeros_like(difference[:, :1]), difference), dim=1)
        if self.mode == "rgb_only":
            difference = torch.zeros_like(difference)
        features = self.encoder(torch.cat((small, difference), dim=2).permute(0, 2, 1, 3, 4))
        spatial_weight = self.spatial_score(features).reshape(batch, 36, -1).softmax(dim=-1)
        tokens = features.reshape(batch, 32, 36, -1)
        pooled = (tokens * spatial_weight[:, None]).sum(dim=-1)
        temporal = self.temporal(pooled).transpose(1, 2)
        time_weight = self.temporal_score(temporal).softmax(dim=1)
        summary = torch.cat(((temporal * time_weight).sum(dim=1), temporal.amax(dim=1)), dim=1)
        return self.head(summary)
