"""Dependency-light, seven-class paired metrics for the historical diagnostic."""
from __future__ import annotations

import numpy as np


KNOWN_HISTORY = {"improved": 50, "worsened": 15, "wrong_to_different_wrong": 5,
                 "A_3_to_4": 50, "TCN_3_to_4": 21, "A_4_to_3": 22,
                 "TCN_4_to_3": 33}


def classification_metrics(labels, predictions):
    labels, predictions = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)
    if labels.ndim != 1 or predictions.shape != labels.shape:
        raise ValueError("labels and predictions must be equally sized vectors")
    if np.any((labels < 0) | (labels > 6) | (predictions < 0) | (predictions > 6)):
        raise ValueError("classification labels must be in [0,6]")
    confusion = np.zeros((7, 7), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    support, predicted, correct = confusion.sum(1), confusion.sum(0), confusion.diagonal()
    precision = np.divide(correct, predicted, out=np.zeros(7), where=predicted != 0)
    recall = np.divide(correct, support, out=np.zeros(7), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(7),
                   where=precision + recall != 0)
    return {"count": len(labels),
            "accuracy": float(correct.sum() / len(labels)) if len(labels) else 0.,
            "macro_f1": float(f1.mean()), "confusion": confusion.tolist(),
            "per_class": [{"label_id": i, "precision": float(precision[i]),
                           "recall": float(recall[i]), "f1": float(f1[i]),
                           "support": int(support[i])} for i in range(7)]}


def paired(labels, original, revised):
    labels, original, revised = map(np.asarray, (labels, original, revised))
    if labels.shape != original.shape or labels.shape != revised.shape:
        raise ValueError("paired vectors must share clip coverage")
    changed = original != revised
    return {"changed": int(changed.sum()),
            "improved": int(((original != labels) & (revised == labels)).sum()),
            "worsened": int(((original == labels) & (revised != labels)).sum()),
            "wrong_to_different_wrong": int(((original != labels) &
                                              (revised != labels) & changed).sum())}


def confusions(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    return {f"{left}_to_{right}": int(((labels == left) & (predictions == right)).sum())
            for left, right in ((3, 4), (4, 3))}


def summary(rows):
    labels = np.array([row["label_id"] for row in rows], dtype=int)
    a = np.array([row["A_prediction"] for row in rows], dtype=int)
    tcn = np.array([row["prediction"] for row in rows], dtype=int)
    return {"clip_count": len(rows), "A": classification_metrics(labels, a),
            "old_TCN": classification_metrics(labels, tcn),
            "old_TCN_vs_A": paired(labels, a, tcn),
            "A_directional_errors": confusions(labels, a),
            "old_TCN_directional_errors": confusions(labels, tcn)}


def assert_known_history(aggregate):
    actual = aggregate["old_TCN_vs_A"] | {
        "A_3_to_4": aggregate["A_directional_errors"]["3_to_4"],
        "TCN_3_to_4": aggregate["old_TCN_directional_errors"]["3_to_4"],
        "A_4_to_3": aggregate["A_directional_errors"]["4_to_3"],
        "TCN_4_to_3": aggregate["old_TCN_directional_errors"]["4_to_3"]}
    mismatch = {key: {"expected": expected, "observed": actual[key]}
                for key, expected in KNOWN_HISTORY.items() if actual[key] != expected}
    if mismatch:
        raise ValueError(f"old val823 predictions do not reproduce the historical report: {mismatch}")
