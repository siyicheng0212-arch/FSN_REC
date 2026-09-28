"""Calibration and selective-risk diagnostics for seven-class predictions."""

from __future__ import annotations

import math
from typing import Any

import torch


def probability_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor | None = None,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    probabilities = torch.as_tensor(probabilities).detach().cpu().double()
    targets = torch.as_tensor(targets).detach().cpu().long()
    if probabilities.ndim != 2 or probabilities.shape[1] != 7:
        raise ValueError("probabilities must have shape [N, 7]")
    if probabilities.shape[0] == 0 or targets.shape != (len(probabilities),):
        raise ValueError("targets must have matching nonempty shape [N]")
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("probabilities must be in [0, 1]")
    if not torch.allclose(
        probabilities.sum(dim=1), torch.ones(len(probabilities), dtype=torch.double),
        atol=1e-6,
    ):
        raise ValueError("probability rows must sum to one")
    if ((targets < 0) | (targets > 6)).any():
        raise ValueError("targets must be class ids in [0, 6]")
    if bins < 2:
        raise ValueError("at least two calibration bins are required")
    if predictions is None:
        predictions = probabilities.argmax(dim=1)
    else:
        predictions = torch.as_tensor(predictions).detach().cpu().long()
    if predictions.shape != targets.shape or ((predictions < 0) | (predictions > 6)).any():
        raise ValueError("predictions must be matching class ids in [0, 6]")

    row = torch.arange(len(targets))
    correctness = (predictions == targets).double()
    selected_confidence = probabilities[row, predictions]
    true_probability = probabilities[row, targets]
    one_hot = torch.nn.functional.one_hot(targets, num_classes=7).double()
    nll = -torch.log(true_probability.clamp_min(1e-15)).mean()
    brier = ((probabilities - one_hot) ** 2).sum(dim=1).mean()
    entropy = -(probabilities * probabilities.clamp_min(1e-15).log()).sum(dim=1)

    bin_index = torch.clamp((selected_confidence * bins).long(), max=bins - 1)
    reliability = []
    ece = 0.0
    for index in range(bins):
        mask = bin_index == index
        count = int(mask.sum())
        accuracy = float(correctness[mask].mean()) if count else None
        confidence = float(selected_confidence[mask].mean()) if count else None
        reliability.append({
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "count": count,
            "accuracy": accuracy,
            "mean_selected_probability": confidence,
        })
        if count:
            ece += count / len(targets) * abs(accuracy - confidence)

    high_confidence = selected_confidence >= 0.9
    high_count = int(high_confidence.sum())
    order = torch.argsort(selected_confidence, descending=True, stable=True)
    ordered_errors = 1.0 - correctness[order]
    cumulative_errors = ordered_errors.cumsum(dim=0)
    risk_by_coverage = {}
    for coverage in (0.5, 0.8, 0.9, 1.0):
        keep = max(1, math.ceil(len(targets) * coverage))
        risk_by_coverage[str(coverage)] = float(cumulative_errors[keep - 1] / keep)
    aurc = float((cumulative_errors / torch.arange(1, len(targets) + 1)).mean())
    return {
        "num_samples": len(targets),
        "decision_accuracy": float(correctness.mean()),
        "mean_selected_probability": float(selected_confidence.mean()),
        "mean_predictive_entropy": float(entropy.mean()),
        "nll": float(nll),
        "multiclass_brier": float(brier),
        "top_label_ece": ece,
        "reliability_bins": reliability,
        "high_confidence_threshold": 0.9,
        "high_confidence_count": high_count,
        "high_confidence_error_count": int((1 - correctness[high_confidence]).sum()),
        "high_confidence_error_rate": (
            float((1 - correctness[high_confidence]).mean()) if high_count else None
        ),
        "selective_error_rate_by_coverage": risk_by_coverage,
        "area_under_risk_coverage_curve": aurc,
    }
