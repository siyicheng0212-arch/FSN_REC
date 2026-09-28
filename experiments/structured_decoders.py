"""Higher-order train-only structured decoders for FSN sequence ablations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import torch

from experiments.pilot_data import ClipRecord, EXPECTED_LABELS
from experiments.sequence_decoder import _ordered_groups


NUM_CLASSES = len(EXPECTED_LABELS)


@dataclass(frozen=True)
class SecondOrderPrior:
    initial_probability: torch.Tensor
    first_transition_probability: torch.Tensor
    second_transition_probability: torch.Tensor
    smoothing: float


@dataclass(frozen=True)
class SegmentalPrior:
    initial_probability: torch.Tensor
    segment_transition_probability: torch.Tensor
    duration_probability: torch.Tensor
    max_duration: int
    smoothing: float


def fit_second_order_prior(
    records: Sequence[ClipRecord], smoothing: float = 1.0
) -> SecondOrderPrior:
    if not records or smoothing <= 0:
        raise ValueError("records must be non-empty and smoothing positive")
    initial = torch.full((NUM_CLASSES,), smoothing, dtype=torch.float64)
    first = torch.full(
        (NUM_CLASSES, NUM_CLASSES), smoothing, dtype=torch.float64
    )
    second = torch.full(
        (NUM_CLASSES, NUM_CLASSES, NUM_CLASSES),
        smoothing,
        dtype=torch.float64,
    )
    for indices in _ordered_groups(records):
        labels = [records[index].label_id for index in indices]
        initial[labels[0]] += 1
        for left, right in zip(labels, labels[1:]):
            first[left, right] += 1
        for left, middle, right in zip(labels, labels[1:], labels[2:]):
            second[left, middle, right] += 1
    initial /= initial.sum()
    first /= first.sum(dim=1, keepdim=True)
    second /= second.sum(dim=2, keepdim=True)
    return SecondOrderPrior(initial, first, second, float(smoothing))


def fit_segmental_prior(
    records: Sequence[ClipRecord], smoothing: float = 1.0
) -> SegmentalPrior:
    if not records or smoothing <= 0:
        raise ValueError("records must be non-empty and smoothing positive")
    groups = _ordered_groups(records)
    runs_by_sequence: list[list[tuple[int, int]]] = []
    maximum = 1
    for indices in groups:
        labels = [records[index].label_id for index in indices]
        runs: list[tuple[int, int]] = []
        current, length = labels[0], 1
        for label in labels[1:]:
            if label == current:
                length += 1
            else:
                runs.append((current, length))
                current, length = label, 1
        runs.append((current, length))
        runs_by_sequence.append(runs)
        maximum = max(maximum, *(length for _, length in runs))

    initial = torch.full((NUM_CLASSES,), smoothing, dtype=torch.float64)
    transition = torch.full(
        (NUM_CLASSES, NUM_CLASSES), smoothing, dtype=torch.float64
    )
    duration = torch.full(
        (NUM_CLASSES, maximum + 1), smoothing, dtype=torch.float64
    )
    duration[:, 0] = 0
    for runs in runs_by_sequence:
        initial[runs[0][0]] += 1
        for label, length in runs:
            duration[label, length] += 1
        for (left, _), (right, _) in zip(runs, runs[1:]):
            transition[left, right] += 1
    initial /= initial.sum()
    transition /= transition.sum(dim=1, keepdim=True)
    duration /= duration.sum(dim=1, keepdim=True)
    return SegmentalPrior(
        initial, transition, duration, maximum, float(smoothing)
    )


def _second_order_viterbi(
    emission: torch.Tensor, prior: SecondOrderPrior, weight: float
) -> torch.Tensor:
    length = emission.shape[0]
    if length == 1:
        return emission.argmax(dim=1)
    log_initial = prior.initial_probability.log().to(emission)
    log_first = prior.first_transition_probability.log().to(emission)
    log_second = prior.second_transition_probability.log().to(emission)
    pair = (
        emission[0, :, None]
        + emission[1, None, :]
        + weight * (log_initial[:, None] + log_first)
    )
    backpointers: list[torch.Tensor] = []
    for step in range(2, length):
        scores = pair[:, :, None] + weight * log_second
        best, previous = scores.max(dim=0)
        pair = emission[step, None, :] + best
        backpointers.append(previous)
    path = torch.empty(length, dtype=torch.long, device=emission.device)
    flat = pair.argmax()
    path[-2] = flat // NUM_CLASSES
    path[-1] = flat % NUM_CLASSES
    for step in range(length - 3, -1, -1):
        path[step] = backpointers[step][path[step + 1], path[step + 2]]
    return path


def _segmental_viterbi(
    emission: torch.Tensor, prior: SegmentalPrior, weight: float
) -> torch.Tensor:
    length = emission.shape[0]
    if length == 1:
        return emission.argmax(dim=1)
    log_initial = prior.initial_probability.log().to(emission)
    log_transition = prior.segment_transition_probability.log().to(emission)
    log_duration = prior.duration_probability.clamp_min(1e-300).log().to(emission)
    prefix = torch.cat(
        [emission.new_zeros((1, NUM_CLASSES)), emission.cumsum(dim=0)], dim=0
    )
    score = emission.new_full((length + 1, NUM_CLASSES), -torch.inf)
    previous_class = torch.full(
        (length + 1, NUM_CLASSES), -1, dtype=torch.long, device=emission.device
    )
    previous_duration = torch.ones(
        (length + 1, NUM_CLASSES), dtype=torch.long, device=emission.device
    )
    for end in range(1, length + 1):
        for label in range(NUM_CLASSES):
            candidates = []
            back_classes = []
            durations = range(1, min(end, prior.max_duration) + 1)
            for duration in durations:
                segment_emission = prefix[end, label] - prefix[end - duration, label]
                duration_score = weight * log_duration[label, duration]
                if end == duration:
                    candidates.append(
                        log_initial[label] * weight
                        + duration_score
                        + segment_emission
                    )
                    back_classes.append(-1)
                else:
                    values = score[end - duration] + weight * log_transition[:, label]
                    values = values.clone()
                    values[label] = -torch.inf
                    best_value, best_class = values.max(dim=0)
                    candidates.append(best_value + duration_score + segment_emission)
                    back_classes.append(int(best_class))
            stacked = torch.stack(candidates)
            best_index = int(stacked.argmax())
            score[end, label] = stacked[best_index]
            previous_duration[end, label] = best_index + 1
            previous_class[end, label] = back_classes[best_index]
    path = torch.empty(length, dtype=torch.long, device=emission.device)
    end, label = length, int(score[length].argmax())
    while end:
        duration = int(previous_duration[end, label])
        path[end - duration : end] = label
        label = int(previous_class[end, label])
        end -= duration
    return path


def _decode(
    logits: torch.Tensor,
    records: Sequence[ClipRecord],
    weight: float,
    decode_one,
) -> torch.Tensor:
    if weight < 0:
        raise ValueError("weight must be non-negative")
    logits = torch.as_tensor(logits).detach().cpu().float()
    if logits.shape != (len(records), NUM_CLASSES):
        raise ValueError(f"logits must have shape [{len(records)}, 7]")
    emission = torch.log_softmax(logits, dim=1)
    prediction = logits.argmax(dim=1)
    for indices in _ordered_groups(records):
        index = torch.tensor(indices, dtype=torch.long)
        prediction[index] = decode_one(emission[index])
    return prediction


def decode_second_order_sequences(
    logits: torch.Tensor,
    records: Sequence[ClipRecord],
    prior: SecondOrderPrior,
    weight: float = 1.0,
) -> torch.Tensor:
    return _decode(
        logits,
        records,
        weight,
        lambda emission: _second_order_viterbi(emission, prior, weight),
    )


def decode_segmental_sequences(
    logits: torch.Tensor,
    records: Sequence[ClipRecord],
    prior: SegmentalPrior,
    weight: float = 1.0,
) -> torch.Tensor:
    return _decode(
        logits,
        records,
        weight,
        lambda emission: _segmental_viterbi(emission, prior, weight),
    )


__all__ = [
    "SecondOrderPrior",
    "SegmentalPrior",
    "decode_second_order_sequences",
    "decode_segmental_sequences",
    "fit_second_order_prior",
    "fit_segmental_prior",
]
