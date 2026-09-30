"""Deterministic within-clip evidence probes; never join unrelated clips."""

from __future__ import annotations

import hashlib

import torch


def apply_probe(frames: torch.Tensor, clip_ids: list[str], probe: str = "none", seed: int = 42) -> torch.Tensor:
    """Input/output [B,T,3,H,W]. These are diagnostics, not clinical ROIs.

    ``static`` repeats the middle cached frame, so it also changes temporal
    coverage. ``shuffle`` retains the frame multiset and changes its order.
    A performance drop does not establish motion localization or causality.
    """
    if frames.ndim != 5 or frames.shape[2] != 3:
        raise ValueError("expected [B,T,3,H,W]")
    if len(clip_ids) != frames.shape[0] or frames.shape[1] < 1:
        raise ValueError("clip IDs/batch mismatch or empty temporal dimension")
    if probe == "none":
        return frames
    if probe == "static":
        return frames[:, frames.shape[1] // 2: frames.shape[1] // 2 + 1].expand_as(frames).contiguous()
    if probe == "shuffle":
        outputs = []
        for clip_id, clip in zip(clip_ids, frames):
            token = hashlib.sha256(f"fsn-probe-v1|{seed}|{clip_id}".encode()).digest()
            generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(token[:8], "big") % (2**63 - 1))
            order = torch.randperm(frames.shape[1], generator=generator).to(frames.device)
            outputs.append(clip.index_select(0, order))
        return torch.stack(outputs)
    raise ValueError(f"unknown evidence probe: {probe}")
