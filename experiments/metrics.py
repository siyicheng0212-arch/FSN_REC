"""Dependency-light evaluation metrics for the seven-class FSN task.

The public entry point accepts model logits, integer targets, and the clip
metadata rows that correspond to them.  Metrics are returned as ordinary
Python containers/scalars so the result can be passed directly to
``json.dump``.

Conventions
-----------
* Scores are fractions in ``[0, 1]`` rather than percentages.
* The confusion matrix uses target classes as rows and predictions as columns.
* Undefined precision/recall/F1 values are reported as ``0.0``.
* Macro F1 always averages all seven classes, including classes with no
  support in a slice.  Weighted F1 weights classes by target support.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch


DEFAULT_CLASS_NAMES = (
    "消毒",
    "进针",
    "运针",
    "扫散",
    "再灌注",
    "拔针",
    "固定",
)
NUM_CLASSES = len(DEFAULT_CLASS_NAMES)
UNKNOWN_SOURCE = "__unknown__"


def _as_cpu_tensor(value: Any, name: str) -> torch.Tensor:
    try:
        return torch.as_tensor(value).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name} must be convertible to a torch tensor") from exc


def _validate_inputs(
    logits: Any,
    targets: Any,
    class_names: Sequence[str] | None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    logits_tensor = _as_cpu_tensor(logits, "logits")
    targets_tensor = _as_cpu_tensor(targets, "targets")

    if logits_tensor.ndim != 2 or logits_tensor.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"logits must have shape [N, {NUM_CLASSES}], got "
            f"{tuple(logits_tensor.shape)}"
        )
    if targets_tensor.ndim != 1:
        raise ValueError(f"targets must have shape [N], got {tuple(targets_tensor.shape)}")
    if logits_tensor.shape[0] != targets_tensor.shape[0]:
        raise ValueError(
            "logits and targets must contain the same number of samples: "
            f"{logits_tensor.shape[0]} != {targets_tensor.shape[0]}"
        )
    if logits_tensor.shape[0] == 0:
        raise ValueError("at least one sample is required")
    if not torch.isfinite(logits_tensor).all().item():
        raise ValueError("logits must contain only finite values")

    if targets_tensor.is_floating_point():
        if not torch.isfinite(targets_tensor).all().item():
            raise ValueError("targets must contain only finite values")
        if not torch.equal(targets_tensor, targets_tensor.round()):
            raise ValueError("targets must contain integer class ids")
    targets_tensor = targets_tensor.to(torch.long)
    invalid = (targets_tensor < 0) | (targets_tensor >= NUM_CLASSES)
    if invalid.any().item():
        values = sorted(set(targets_tensor[invalid].tolist()))
        raise ValueError(f"target class ids must be in [0, {NUM_CLASSES - 1}], got {values}")

    names = list(DEFAULT_CLASS_NAMES if class_names is None else class_names)
    if len(names) != NUM_CLASSES:
        raise ValueError(f"class_names must contain exactly {NUM_CLASSES} names")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("class_names must be non-empty strings")

    predictions = logits_tensor.argmax(dim=1).to(torch.long)
    return predictions, targets_tensor, names


def _column_value_at(value: Any, index: int, sample_count: int, key: str) -> Any:
    """Read one value from DataLoader-style columnar metadata."""

    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            if sample_count != 1:
                raise ValueError(f"metadata column {key!r} is scalar for {sample_count} samples")
            return value.item()
        if len(value) != sample_count:
            raise ValueError(
                f"metadata column {key!r} has {len(value)} rows; expected {sample_count}"
            )
        item = value[index]
        return item.item() if item.ndim == 0 else item.tolist()

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != sample_count:
            raise ValueError(
                f"metadata column {key!r} has {len(value)} rows; expected {sample_count}"
            )
        return value[index]

    if sample_count != 1:
        raise ValueError(f"metadata column {key!r} is scalar for {sample_count} samples")
    return value


def _normalise_metadata(metadata: Any, sample_count: int) -> list[Mapping[str, Any]]:
    """Accept either a list of rows or a mapping of collated columns."""

    if isinstance(metadata, Mapping):
        rows: list[Mapping[str, Any]] = []
        for index in range(sample_count):
            rows.append(
                {
                    str(key): _column_value_at(value, index, sample_count, str(key))
                    for key, value in metadata.items()
                }
            )
        return rows

    if isinstance(metadata, (str, bytes, bytearray)) or not isinstance(metadata, Sequence):
        raise TypeError("clip_metadata must be a sequence of mappings or a mapping of columns")
    if len(metadata) != sample_count:
        raise ValueError(
            f"clip_metadata has {len(metadata)} rows; expected {sample_count}"
        )
    rows = list(metadata)
    if any(not isinstance(row, Mapping) for row in rows):
        raise TypeError("every clip_metadata row must be a mapping")
    return rows


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(numerator, dtype=torch.float64)
    valid = denominator != 0
    result[valid] = numerator[valid].to(torch.float64) / denominator[valid].to(torch.float64)
    return result


def _metric_block(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: Sequence[str],
) -> dict[str, Any]:
    encoded = targets * NUM_CLASSES + predictions
    confusion = torch.bincount(encoded, minlength=NUM_CLASSES**2).reshape(
        NUM_CLASSES, NUM_CLASSES
    )

    true_positive = confusion.diag()
    support = confusion.sum(dim=1)
    predicted_count = confusion.sum(dim=0)
    false_positive = predicted_count - true_positive
    false_negative = support - true_positive

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    sample_count = int(targets.numel())
    correct = int(true_positive.sum().item())
    accuracy = float(correct / sample_count) if sample_count else 0.0

    total_true_positive = true_positive.sum().reshape(1)
    total_false_positive = false_positive.sum().reshape(1)
    total_false_negative = false_negative.sum().reshape(1)
    micro_precision = _safe_divide(
        total_true_positive, total_true_positive + total_false_positive
    )
    micro_recall = _safe_divide(
        total_true_positive, total_true_positive + total_false_negative
    )
    micro_f1 = _safe_divide(
        2.0 * micro_precision * micro_recall, micro_precision + micro_recall
    )[0]

    if sample_count:
        weighted_f1 = float((f1 * support.to(torch.float64)).sum().item() / sample_count)
    else:
        weighted_f1 = 0.0

    per_class = [
        {
            "class_id": class_id,
            "class_name": class_names[class_id],
            "precision": float(precision[class_id].item()),
            "recall": float(recall[class_id].item()),
            "f1": float(f1[class_id].item()),
            "support": int(support[class_id].item()),
        }
        for class_id in range(NUM_CLASSES)
    ]

    return {
        "num_samples": sample_count,
        "accuracy": accuracy,
        "macro_f1": float(f1.mean().item()),
        "micro_f1": float(micro_f1.item()),
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.to(torch.long).tolist(),
    }


def _source_from_row(row: Mapping[str, Any]) -> str:
    value = row.get("source_collection", row.get("source"))
    if value is None or not str(value).strip():
        return UNKNOWN_SOURCE
    return str(value)


def _duration_from_row(row: Mapping[str, Any], row_index: int) -> float | None:
    value = None
    for key in ("clip_duration_sec", "duration_sec", "duration"):
        if key in row and row[key] is not None:
            value = row[key]
            break

    if value is None and row.get("clip_start_sec") is not None and row.get("clip_end_sec") is not None:
        value = float(row["clip_end_sec"]) - float(row["clip_start_sec"])
    if value is None:
        return None

    try:
        duration = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"clip_metadata row {row_index} has a non-numeric duration") from exc
    if not math.isfinite(duration) or duration < 0:
        raise ValueError(
            f"clip_metadata row {row_index} has invalid duration {duration!r}"
        )
    return duration


def _block_for_indices(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    indices: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, Any]:
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    return _metric_block(predictions[index_tensor], targets[index_tensor], class_names)


def compute_classification_metrics(
    logits: Any,
    targets: Any,
    clip_metadata: Any,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute overall and prespecified slice metrics for FSN classification.

    Args:
        logits: Array-like scores with shape ``[N, 7]``. Softmax is neither
            needed nor applied because only ``argmax`` predictions are used.
        targets: Array-like integer class ids with shape ``[N]``.
        clip_metadata: Either one mapping per sample or a mapping of collated
            columns. Source is read from ``source_collection`` (falling back to
            ``source``). Duration is read from ``clip_duration_sec``,
            ``duration_sec``, or ``duration``; start/end times are a final
            fallback.
        class_names: Optional names in class-id order. Exactly seven are
            required; the canonical FSN labels are used by default.

    Returns:
        A JSON-safe dictionary with an ``all`` metric block and the required
        ``duration_le_10s``, ``duration_gt_10s``, and ``source_collection``
        slices. Samples without duration remain in ``all`` and source slices,
        and their count is made explicit in ``slice_coverage``.
    """

    predictions, target_tensor, names = _validate_inputs(logits, targets, class_names)
    metadata_rows = _normalise_metadata(clip_metadata, len(target_tensor))

    source_indices: dict[str, list[int]] = defaultdict(list)
    duration_le_10s: list[int] = []
    duration_gt_10s: list[int] = []
    duration_lt_01s: list[int] = []
    duration_ge_01s: list[int] = []
    duration_unknown = 0
    source_unknown = 0

    for index, row in enumerate(metadata_rows):
        source = _source_from_row(row)
        source_indices[source].append(index)
        if source == UNKNOWN_SOURCE:
            source_unknown += 1

        duration = _duration_from_row(row, index)
        if duration is None:
            duration_unknown += 1
        elif duration <= 10.0:
            duration_le_10s.append(index)
            if duration < 0.1:
                duration_lt_01s.append(index)
            else:
                duration_ge_01s.append(index)
        else:
            duration_gt_10s.append(index)
            duration_ge_01s.append(index)

    source_metrics = {
        source: _block_for_indices(predictions, target_tensor, indices, names)
        for source, indices in sorted(source_indices.items())
    }

    return {
        "schema_version": "fsn-classification-metrics-1.0",
        "num_classes": NUM_CLASSES,
        "class_names": names,
        "metric_scale": "fraction_0_to_1",
        "confusion_matrix_axes": {"rows": "target", "columns": "prediction"},
        "all": _metric_block(predictions, target_tensor, names),
        "slices": {
            "duration_le_10s": _block_for_indices(
                predictions, target_tensor, duration_le_10s, names
            ),
            "duration_gt_10s": _block_for_indices(
                predictions, target_tensor, duration_gt_10s, names
            ),
            "duration_lt_0.1s": _block_for_indices(
                predictions, target_tensor, duration_lt_01s, names
            ),
            "duration_ge_0.1s": _block_for_indices(
                predictions, target_tensor, duration_ge_01s, names
            ),
            "source_collection": source_metrics,
        },
        "slice_coverage": {
            "total": len(metadata_rows),
            "duration_known": len(metadata_rows) - duration_unknown,
            "duration_unknown": duration_unknown,
            "source_unknown": source_unknown,
        },
    }


# Concise alias for experiment runners while keeping the descriptive public API.
compute_metrics = compute_classification_metrics


__all__ = [
    "DEFAULT_CLASS_NAMES",
    "NUM_CLASSES",
    "compute_classification_metrics",
    "compute_metrics",
]
