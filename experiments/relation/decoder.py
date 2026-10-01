"""Fit trusted training transitions and selectively decode caller-ordered clips.

This decoder is a conditional sequence scorer, not an order reconstruction tool.
It neither sorts clips nor consumes inference-time clinical/action annotations.
Only explicitly supplied candidate adjacency is eligible for connection; a
missing eligibility mask safely falls back to independent visual predictions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import numpy as np


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _real(value, name: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not np.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def _array(value, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must be real-valued")
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def fit_transition(
    labels_by_id: Mapping[str, int],
    edges: Sequence[dict],
    num_classes: int = 7,
    smoothing: float = 1.0,
) -> dict:
    """Count directed transitions exclusively on trusted *training* edges.

    Each edge must declare ``split='train'`` and a ``status`` in ``C/D/U``.
    ``C`` means a verified valid clinical relationship; only those edges count.
    C edges require distinct ``left_clip_id`` / ``right_clip_id`` entries in
    ``labels_by_id``. D/U edges do not enter the count matrix. Every non-training
    edge is rejected even if D/U, so validation annotations cannot accidentally
    enter fitting. Source identity alone never creates a trusted edge here.

    Laplace smoothing is strictly positive. This cleaned matrix intentionally
    differs from old v3, which counted annotation adjacency without this audit.
    Returns NumPy transition/count arrays plus a JSON-compatible audit dict.
    """
    num_classes = _integer(num_classes, "num_classes", minimum=1)
    smoothing = _real(smoothing, "smoothing", minimum=0.0)
    if smoothing == 0:
        raise ValueError("smoothing must be > 0 to keep every transition positive")
    if not isinstance(labels_by_id, Mapping):
        raise ValueError("labels_by_id must be a mapping")
    labels = {}
    for clip_id, label in labels_by_id.items():
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise ValueError("label keys must be nonempty clip ID strings")
        label = _integer(label, f"label for {clip_id}")
        if label >= num_classes:
            raise ValueError(f"label for {clip_id} is outside the class range")
        labels[clip_id] = label
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
        raise ValueError("edges must be a sequence of mappings")

    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    status_counts = {"C": 0, "D": 0, "U": 0}
    for index, edge in enumerate(edges):
        if not isinstance(edge, Mapping):
            raise ValueError(f"edge {index} must be a mapping")
        if edge.get("split") != "train":
            raise ValueError(f"edge {index} is not explicitly split='train'")
        status = edge.get("status")
        if not isinstance(status, str) or status not in status_counts:
            raise ValueError(f"edge {index} must have status C, D, or U")
        status_counts[status] += 1
        if status != "C":
            continue
        left, right = edge.get("left_clip_id"), edge.get("right_clip_id")
        if not all(isinstance(x, str) and x.strip() for x in (left, right)):
            raise ValueError(f"C edge {index} needs two nonempty clip IDs")
        if left == right:
            raise ValueError(f"C edge {index} cannot connect a clip to itself")
        if left not in labels or right not in labels:
            raise ValueError(f"C edge {index} has missing/unknown endpoint labels")
        counts[labels[left], labels[right]] += 1

    smoothed = counts.astype(np.float64) + smoothing
    # Dividing before the row sum avoids overflow for very large finite smoothing.
    scaled = smoothed / smoothed.max(axis=1, keepdims=True)
    transition = scaled / scaled.sum(axis=1, keepdims=True)
    return {
        "transition": transition,
        "counts": counts,
        "audit": {
            "num_classes": num_classes,
            "smoothing": smoothing,
            "edge_count": len(edges),
            "accepted_C": status_counts["C"],
            "ignored_D": status_counts["D"],
            "ignored_U": status_counts["U"],
            "fit_split": "train",
            "trusted_edge_status": "C",
            "uniform_initial_prior": True,
        },
    }


def decode(
    logits,
    transition,
    reliability,
    threshold: float = 0.9,
    mode: str = "hard",
    strength: float = 1.0,
    eligible=None,
    potential_bound: float | None = None,
) -> dict:
    """Viterbi with selective directed transition potentials.

    Inputs are caller-ordered ``logits[T,K]`` and ``reliability[T-1]``. The latter
    must be predictions of the relation model, not human C/D/U test annotations.
    ``eligible[T-1]`` is a boolean structural candidate mask (for example valid
    playback adjacency in one declared recording), not a clinical GT mask. None
    makes every connection ineligible; arbitrary lists must not be auto-chained.

    Hard mode sets r=1 only where q>=threshold, q>0, and eligible=True. Soft mode
    uses r=q on eligible connections and explicitly ignores threshold. Weighted
    pair potentials are strength * log((1-r) + K*r*A[i,j]); optional
    ``potential_bound`` clips the final weighted potential to [-bound, bound].
    There is a uniform initial prior, including after each cut. Zero connections
    split independent chains. Singleton/all-off/strength-zero/bound-zero cases
    reproduce visual argmax exactly, including its first-index tie convention.

    Subtracting a per-clip constant from logits is equivalent to using visual
    log-softmax scores in this conditional argmax. No action labels are read.
    """
    logits = _array(logits, "logits")
    if logits.ndim != 2 or min(logits.shape, default=0) < 1:
        raise ValueError("logits must have shape [T,K] with T,K >= 1")
    time_steps, num_classes = logits.shape
    transition = _array(transition, "transition")
    if transition.shape != (num_classes, num_classes):
        raise ValueError("transition shape must be [K,K]")
    if (transition <= 0).any():
        raise ValueError("transition entries must be strictly positive")
    if not np.allclose(transition.sum(axis=1), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("transition rows must already sum to one")
    reliability = _array(reliability, "reliability")
    if reliability.shape != (time_steps - 1,):
        raise ValueError("reliability shape must be [T-1]")
    if (reliability < 0).any() or (reliability > 1).any():
        raise ValueError("reliability values must lie in [0,1]")
    if mode not in ("hard", "soft"):
        raise ValueError("mode must be 'hard' or 'soft'")
    threshold = _real(threshold, "threshold", minimum=0.0)
    if threshold > 1:
        raise ValueError("threshold must lie in [0,1]")
    strength = _real(strength, "strength", minimum=0.0)
    if potential_bound is not None:
        potential_bound = _real(potential_bound, "potential_bound", minimum=0.0)
    if eligible is None:
        eligible = np.zeros(time_steps - 1, dtype=bool)
    else:
        eligible = np.asarray(eligible)
        if time_steps == 1 and eligible.shape == (0,):
            eligible = eligible.astype(bool)
        if eligible.shape != (time_steps - 1,) or eligible.dtype.kind != "b":
            raise ValueError("eligible must be a boolean array of shape [T-1]")

    if mode == "hard":
        effective = (eligible & (reliability > 0) & (reliability >= threshold)).astype(float)
    else:
        effective = np.where(eligible, reliability, 0.0)
    visual = np.argmax(logits, axis=1).astype(np.int64)
    predictions = visual.copy()
    if time_steps > 1 and np.any(effective > 0) and strength > 0 and potential_bound != 0:
        # End an independent subchain wherever an edge is exactly neutral.
        boundaries = [0] + (np.flatnonzero(effective == 0) + 1).tolist() + [time_steps]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            if stop - start == 1:
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                unary = logits[start:stop] - logits[start:stop].max(axis=1, keepdims=True)
            if not np.isfinite(unary).all():
                raise ValueError("logits have an unsupported numerical dynamic range")
            scores = unary[0].copy()
            backpointers = []
            for time in range(start + 1, stop):
                r = effective[time - 1]
                with np.errstate(over="ignore", invalid="ignore"):
                    potential = strength * np.log((1 - r) + num_classes * r * transition)
                if potential_bound is not None:
                    potential = np.clip(potential, -potential_bound, potential_bound)
                if not np.isfinite(potential).all():
                    raise ValueError("weighted potentials exceed the supported numerical range")
                candidates = scores[:, None] + potential
                pointers = np.argmax(candidates, axis=0)
                scores = candidates[pointers, np.arange(num_classes)] + unary[time - start]
                if not np.isfinite(scores).all():
                    raise ValueError("sequence scores exceed the supported numerical range")
                scores -= scores.max()
                backpointers.append(pointers)
            state = int(np.argmax(scores))
            predictions[stop - 1] = state
            for time in range(stop - 2, start - 1, -1):
                state = int(backpointers[time - start][state])
                predictions[time] = state

    return {
        "predictions": predictions.tolist(),
        "visual_predictions": visual.tolist(),
        "effective_reliability": effective.tolist(),
        "changed_indices": np.flatnonzero(predictions != visual).tolist(),
    }
