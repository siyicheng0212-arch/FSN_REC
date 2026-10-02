"""Paired evaluation of a gate over the unmodified historical TCN chain."""

from __future__ import annotations

from collections import Counter
import math

import numpy as np
import torch
from torch.nn import functional as F

from experiments.fsn_tcn.io import load_chain
from experiments.fsn_tcn.io import validate_chains
from experiments.relation.evaluate_suite import classification_metrics


def _paired(labels, before, after):
    labels, before, after = [np.asarray(values, dtype=int) for values in (labels, before, after)]
    altered = before != after
    return {"changed": int(altered.sum()),
            "improved": int(((before != labels) & (after == labels)).sum()),
            "worsened": int(((before == labels) & (after != labels)).sum()),
            "wrong_to_different_wrong": int(((before != labels) & (after != labels) & altered).sum())}


def _directions(labels, predicted):
    labels, predicted = np.asarray(labels), np.asarray(predicted)
    return {"sweep_to_reperfusion": int(((labels == 3) & (predicted == 4)).sum()),
            "reperfusion_to_sweep": int(((labels == 4) & (predicted == 3)).sum())}


def _compare(labels, visual, old, current):
    values = {"A": classification_metrics(labels, visual),
              "old_TCN": classification_metrics(labels, old),
              "new": classification_metrics(labels, current),
              "old_TCN_vs_A": _paired(labels, visual, old),
              "new_vs_A": _paired(labels, visual, current),
              "new_vs_old_TCN": _paired(labels, old, current),
              "directional_errors": {"A": _directions(labels, visual),
                                     "old_TCN": _directions(labels, old),
                                     "new": _directions(labels, current)}}
    labels, visual, old, current = [np.asarray(x) for x in (labels, visual, old, current)]
    values.update({
        "old_rescues_preserved": int(((visual != labels) & (old == labels) & (current == labels)).sum()),
        "old_rescues_lost": int(((visual != labels) & (old == labels) & (current != labels)).sum()),
        "old_spoils_repaired": int(((visual == labels) & (old != labels) & (current == labels)).sum()),
        "old_spoils_remaining": int(((visual == labels) & (old != labels) & (current != labels)).sum()),
    })
    return values


def _stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "std": None, "quantiles": None}
    return {"count": len(values), "mean": float(values.mean()), "std": float(values.std()),
            "quantiles": {str(percent): float(np.quantile(values, percent))
                          for percent in (0.05, 0.5, 0.95)}}


def _legacy_scores(base, features, a_logits, eligible, layout):
    if layout == "full_chain":
        return base(features, a_logits)
    if layout != "eligible_segments":
        raise ValueError("declare the exact historical chain layout")
    start, outputs = 0, []
    for i, flag in enumerate(eligible):
        if not flag:
            outputs.append(base(features[start:i + 1], a_logits[start:i + 1]))
            start = i + 1
    outputs.append(base(features[start:], a_logits[start:]))
    return torch.cat(outputs)


def _historical_segments(chain, layout):
    ids, eligible = chain["ordered_clip_ids"], chain["eligible"]
    if layout == "full_chain":
        return [(0, len(ids))]
    if layout != "eligible_segments":
        raise ValueError("declare the exact historical chain layout")
    start, runs = 0, []
    for i, connected in enumerate(eligible):
        if not connected:
            runs.append((start, i + 1))
            start = i + 1
    runs.append((start, len(ids)))
    return runs


@torch.inference_mode()
def evaluate_revision(bundle, chains, model, *, legacy_base, chain_layout,
                      variant, device="cpu", split="val"):
    """Return safe aggregate and private rows. C/D labels never enter inference."""
    if split not in {"train", "val"}:
        raise ValueError("revision evaluation never reads test")
    validate_chains(chains, bundle.metadata, split)
    if variant not in {"scalar", "class_conditioned", "logits_only", "visual_logits"}:
        raise ValueError("unknown gate variant")
    model.eval()
    legacy_base.eval()
    if any(p.requires_grad for p in model.base_tcn.parameters()):
        raise ValueError("the historical TCN must remain frozen during gate evaluation")
    rows, loss_sum = [], 0.0
    alpha_values, a_confidence, old_confidence, score_delta = [], [], [], []
    counts = Counter(chains=0, clips=0, singleton_calls=0)
    for chain in chains:
        ids = chain["ordered_clip_ids"]
        features, a_logits, labels = load_chain(bundle, chain, device)
        starts = _historical_segments(chain, chain_layout)
        current_parts, old_parts, alpha_parts = [], [], []
        for start, stop in starts:
            feat, a_score = features[start:stop], a_logits[start:stop]
            current, detail = model(feat, a_score, gate_mode="learned", return_details=True)
            old = legacy_base(feat, a_score)
            if (current.shape != a_score.shape or old.shape != a_score.shape
                    or not bool(torch.isfinite(current).all()) or not bool(torch.isfinite(old).all())
                    or not torch.equal(detail["tcn_logits"], old)):
                raise ValueError("new wrapper changed historical TCN score/chain semantics")
            current_parts.append(current)
            old_parts.append(old)
            alpha_parts.append(detail["alpha"])
            counts["singleton_calls"] += int(stop - start == 1)
        output = torch.cat(current_parts, 0)
        old_output = torch.cat(old_parts, 0)
        alpha = torch.cat(alpha_parts, 0)
        if alpha.shape != (len(ids),) or not bool(torch.isfinite(alpha).all()) or bool(((alpha < 0) | (alpha > 1)).any()):
            raise ValueError("gate must return one finite coefficient in [0,1] per clip")
        loss_sum += float(F.cross_entropy(output.float(), labels, reduction="sum"))
        a_prob = a_logits.float().softmax(-1)
        old_prob = old_output.float().softmax(-1)
        confidence = a_prob.max(-1).values.detach().cpu().tolist()
        context_confidence = old_prob.max(-1).values.detach().cpu().tolist()
        delta = (old_prob - a_prob).abs().sum(-1).detach().cpu().tolist()
        alpha_values.extend(alpha.detach().float().cpu().tolist())
        a_confidence.extend(confidence)
        old_confidence.extend(context_confidence)
        score_delta.extend(delta)
        for i, clip in enumerate(ids):
            meta = bundle.metadata[clip]
            duration = meta.get("duration", meta.get("clip_duration_sec"))
            if (type(duration) not in (int, float) or not math.isfinite(duration)
                    or duration <= 0 or meta["label_id"] != int(labels[i])):
                raise ValueError("invalid sealed duration or action label")
            rows.append({"clip_id": clip, "chain_id": chain["chain_id"], "split": split,
                         "label_id": int(labels[i]), "A_prediction": int(a_logits[i].argmax()),
                         "old_TCN_prediction": int(old_output[i].argmax()),
                         "prediction": int(output[i].argmax()), "source_kind": meta["source_kind"],
                         "duration": float(duration), "alpha": float(alpha[i]),
                         "A_confidence": float(confidence[i]), "old_TCN_confidence": float(context_confidence[i]),
                         "probability_difference_L1": float(delta[i])})
        counts.update(chains=1, clips=len(ids))
    expected = {clip for clip, row in bundle.metadata.items() if row["split"] == split}
    if len(rows) != len(expected) or {row["clip_id"] for row in rows} != expected:
        raise ValueError("revision evaluation must cover exactly the requested sealed split")
    labels = np.asarray([row["label_id"] for row in rows])
    a = np.asarray([row["A_prediction"] for row in rows])
    old = np.asarray([row["old_TCN_prediction"] for row in rows])
    new = np.asarray([row["prediction"] for row in rows])
    sources = np.asarray([row["source_kind"] for row in rows])
    durations = np.asarray([row["duration"] for row in rows])
    mask = {"clinical": sources == "clinical", "network": sources == "network",
            "duration_le_1s": durations <= 1,
            "duration_gt_1_le_5s": (durations > 1) & (durations <= 5),
            "duration_gt_5s": durations > 5}
    confidence = {"A": _stats(a_confidence), "old_TCN": _stats(old_confidence),
                  "correction_probability_L1": _stats(score_delta), "acceptance_alpha": _stats(alpha_values)}
    return {"schema": "fsn-tcn-revision-evaluation-v1", "role": f"{split}_development_not_independent_test",
            "variant": variant, "old_TCN_chain_layout": chain_layout,
            "counts": dict(counts), "metrics": _compare(labels, a, old, new),
            "cross_entropy": loss_sum / len(rows),
            "confidence_distributions": confidence,
            "slices": {key: _compare(labels[values], a[values], old[values], new[values])
                       for key, values in mask.items()},
            "A_frozen": True, "old_TCN_frozen": True,
            "label_use": "loss/retrospective metrics only; gate input excludes action labels",
            "test_split_read": False}, rows
