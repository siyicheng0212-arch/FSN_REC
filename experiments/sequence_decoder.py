"""Train-only procedural transition prior for FSN clip sequences.

The visual model continues to classify every annotated clip independently.
This module adds no visual parameters: it estimates a seven-state transition
matrix from ordered *training* clips and combines it with visual log
probabilities using Viterbi decoding.  Validation labels are never used to fit
or tune the transition prior.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from experiments.pilot_data import ClipRecord, EXPECTED_LABELS


NUM_CLASSES = len(EXPECTED_LABELS)


def _ordered_groups(records: Sequence[ClipRecord]) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record.record_id or record.clip_id].append(index)
    return [
        sorted(
            indices,
            key=lambda index: (
                records[index].clip_start_sec,
                records[index].clip_end_sec,
                records[index].clip_id,
            ),
        )
        for _, indices in sorted(groups.items())
    ]


@dataclass(frozen=True)
class TransitionPrior:
    initial_probability: torch.Tensor
    transition_probability: torch.Tensor
    smoothing: float
    training_sequences: int
    training_clips: int

    def __post_init__(self) -> None:
        if self.initial_probability.shape != (NUM_CLASSES,):
            raise ValueError("initial_probability must have shape [7]")
        if self.transition_probability.shape != (NUM_CLASSES, NUM_CLASSES):
            raise ValueError("transition_probability must have shape [7, 7]")
        if not torch.isfinite(self.initial_probability).all():
            raise ValueError("initial_probability must be finite")
        if not torch.isfinite(self.transition_probability).all():
            raise ValueError("transition_probability must be finite")
        if (self.initial_probability <= 0).any() or (
            self.transition_probability <= 0
        ).any():
            raise ValueError("smoothed probabilities must be strictly positive")

    def to_dict(self) -> dict:
        return {
            "smoothing": self.smoothing,
            "training_sequences": self.training_sequences,
            "training_clips": self.training_clips,
            "initial_probability": self.initial_probability.tolist(),
            "transition_probability": self.transition_probability.tolist(),
        }


def fit_transition_prior(
    records: Sequence[ClipRecord], smoothing: float = 1.0
) -> TransitionPrior:
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")
    if not records:
        raise ValueError("at least one training clip is required")
    initial = torch.full((NUM_CLASSES,), float(smoothing), dtype=torch.float64)
    transition = torch.full(
        (NUM_CLASSES, NUM_CLASSES), float(smoothing), dtype=torch.float64
    )
    groups = _ordered_groups(records)
    for indices in groups:
        labels = [records[index].label_id for index in indices]
        initial[labels[0]] += 1
        for left, right in zip(labels, labels[1:]):
            transition[left, right] += 1
    initial /= initial.sum()
    transition /= transition.sum(dim=1, keepdim=True)
    return TransitionPrior(
        initial_probability=initial,
        transition_probability=transition,
        smoothing=float(smoothing),
        training_sequences=len(groups),
        training_clips=len(records),
    )


def _viterbi(emission: torch.Tensor, prior: TransitionPrior, weight: float) -> torch.Tensor:
    if emission.ndim != 2 or emission.shape[1] != NUM_CLASSES:
        raise ValueError("emission must have shape [T, 7]")
    if not torch.isfinite(emission).all():
        raise ValueError("emission must be finite")
    if weight < 0:
        raise ValueError("transition weight must be non-negative")
    if emission.shape[0] == 1:
        # No transition evidence exists for a singleton record.  Keeping the
        # visual prediction avoids turning class frequency into a shortcut.
        return emission.argmax(dim=1)
    log_initial = prior.initial_probability.log().to(emission)
    log_transition = prior.transition_probability.log().to(emission)
    length = emission.shape[0]
    score = emission[0] + weight * log_initial
    backpointers: list[torch.Tensor] = []
    for step in range(1, length):
        candidates = score[:, None] + weight * log_transition
        best_score, best_state = candidates.max(dim=0)
        score = emission[step] + best_score
        backpointers.append(best_state)
    path = torch.empty(length, dtype=torch.long, device=emission.device)
    path[-1] = score.argmax()
    for step in range(length - 2, -1, -1):
        path[step] = backpointers[step][path[step + 1]]
    return path


def decode_sequences(
    logits: torch.Tensor,
    records: Sequence[ClipRecord],
    prior: TransitionPrior,
    weight: float = 1.0,
) -> torch.Tensor:
    logits = torch.as_tensor(logits).detach().cpu().float()
    if logits.shape != (len(records), NUM_CLASSES):
        raise ValueError(f"logits must have shape [{len(records)}, 7]")
    emission = torch.log_softmax(logits, dim=1)
    prediction = logits.argmax(dim=1)
    for indices in _ordered_groups(records):
        index = torch.tensor(indices, dtype=torch.long)
        prediction[index] = _viterbi(emission[index], prior, weight)
    return prediction


def sequence_marginal_probabilities(
    logits: torch.Tensor,
    records: Sequence[ClipRecord],
    prior: TransitionPrior,
    weight: float = 1.0,
) -> torch.Tensor:
    """Return per-clip probabilities under the first-order sequence model.

    Viterbi yields the most likely *whole path*.  These forward-backward
    marginals instead sum over every path and are appropriate for NLL, Brier,
    and calibration diagnostics.  The final Viterbi labels are unchanged.
    """
    if weight < 0:
        raise ValueError("transition weight must be non-negative")
    logits = torch.as_tensor(logits).detach().cpu().double()
    if logits.shape != (len(records), NUM_CLASSES):
        raise ValueError(f"logits must have shape [{len(records)}, 7]")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    emission = torch.log_softmax(logits, dim=1)
    log_initial = prior.initial_probability.log().double() * weight
    log_transition = prior.transition_probability.log().double() * weight
    probabilities = torch.empty_like(emission)
    for indices in _ordered_groups(records):
        index = torch.tensor(indices, dtype=torch.long)
        local = emission[index]
        if len(indices) == 1:
            probabilities[index] = local.exp()
            continue
        forward = torch.empty_like(local)
        backward = torch.empty_like(local)
        forward[0] = local[0] + log_initial
        for step in range(1, len(indices)):
            forward[step] = local[step] + torch.logsumexp(
                forward[step - 1, :, None] + log_transition, dim=0
            )
        backward[-1] = 0
        for step in range(len(indices) - 2, -1, -1):
            backward[step] = torch.logsumexp(
                log_transition
                + local[step + 1, None, :]
                + backward[step + 1, None, :],
                dim=1,
            )
        probabilities[index] = torch.softmax(forward + backward, dim=1)
    return probabilities


def predictions_to_logits(predictions: Iterable[int]) -> torch.Tensor:
    predictions = torch.as_tensor(list(predictions), dtype=torch.long)
    if predictions.ndim != 1 or ((predictions < 0) | (predictions >= NUM_CLASSES)).any():
        raise ValueError("predictions must be one-dimensional class ids in [0, 6]")
    logits = torch.full((len(predictions), NUM_CLASSES), -20.0)
    logits[torch.arange(len(predictions)), predictions] = 20.0
    return logits


__all__ = [
    "TransitionPrior",
    "decode_sequences",
    "fit_transition_prior",
    "predictions_to_logits",
    "sequence_marginal_probabilities",
]
