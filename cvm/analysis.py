"""Safe, aggregate-only diagnostics for matched seven-class FSN predictions.

This module reads predictions; it never runs a model, fits thresholds, or chooses
checkpoints. Oracle decoding uses ground truth solely as an explanatory upper
bound. Paired uncertainty resamples original recording groups, not individual
clips. The JSONL input and its metadata sidecar are private artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLASS_NAMES = ("消毒", "进针", "运针", "扫散", "再灌注", "拔针", "固定")
NUM_CLASSES = 7
CLINICAL_GROUPS = ((0,), (6,), (1, 2, 5), (3, 4))
AMBIGUOUS_CLASSES = (1, 2, 3, 4, 5)
SCHEMA_VERSION = "fsn-cvm-safe-analysis-1.0"


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _vector(value: Any, size: int, name: str, *, probabilities: bool = False) -> tuple[float, ...]:
    if not _sequence(value) or len(value) != size or any(not _number(x) for x in value):
        raise ValueError(f"{name} must be a finite vector of the required length")
    result = tuple(float(x) for x in value)
    if probabilities and (any(x < 0 or x > 1 for x in result) or not math.isclose(sum(result), 1.0, abs_tol=1e-6)):
        raise ValueError(f"{name} must be normalized probabilities")
    return result


def _softmax(scores: Sequence[float]) -> tuple[float, ...]:
    maximum = max(scores)
    values = [math.exp(x - maximum) for x in scores]
    total = sum(values)
    return tuple(x / total for x in values)


def _argmax(values: Sequence[float], classes: Sequence[int] | None = None) -> int:
    indices = range(len(values)) if classes is None else classes
    # Canonical class ID resolves ties, independent of taxonomy order.
    return max(indices, key=lambda i: (values[i], -i))


def validate_groups(groups: Any) -> tuple[tuple[int, ...], ...]:
    if not _sequence(groups) or not groups:
        raise ValueError("taxonomy must contain nonempty groups")
    if any(not _sequence(group) or not group for group in groups):
        raise ValueError("taxonomy must contain nonempty groups")
    result = tuple(tuple(group) for group in groups)
    values = [x for group in result for x in group]
    if any(not isinstance(x, int) or isinstance(x, bool) for x in values) or sorted(values) != list(range(NUM_CLASSES)):
        raise ValueError("taxonomy must partition the seven canonical class IDs exactly once")
    return result


def _head(row: Mapping[str, Any], name: str, size: int) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    logits = row.get(name + "_logits")
    probs = row.get(name + "_probs")
    scores = _vector(logits, size, name + "_logits") if logits is not None else None
    probabilities = _vector(probs, size, name + "_probs", probabilities=True) if probs is not None else None
    if scores is not None:
        inferred = _softmax(scores)
        if probabilities is not None and any(not math.isclose(a, b, abs_tol=1e-6) for a, b in zip(inferred, probabilities)):
            raise ValueError(f"{name} logits and probabilities disagree")
        return scores, inferred
    return None if probabilities is None else (probabilities, probabilities)


def _conditional(row: Mapping[str, Any], groups: tuple[tuple[int, ...], ...]) -> tuple[float, ...] | None:
    logits = row.get("conditional_logits")
    probabilities = row.get("conditional_probs", row.get("fine_probs"))
    if logits is None and probabilities is None:
        return None

    def convert(values: Any, is_probability: bool) -> tuple[float, ...]:
        nested = _sequence(values) and len(values) == len(groups) and all(_sequence(v) for v in values)
        flat = None if nested else _vector(values, NUM_CLASSES, "conditional values")
        canonical = [0.0] * NUM_CLASSES
        for index, group in enumerate(groups):
            scores = _vector(values[index], len(group), "conditional group values", probabilities=is_probability) if nested else tuple(flat[c] for c in group)
            if is_probability:
                scores = _vector(scores, len(group), "conditional group probabilities", probabilities=True)
            else:
                scores = _softmax(scores)
            for class_id, probability in zip(group, scores):
                canonical[class_id] = probability
        return tuple(canonical)

    inferred = convert(logits, False) if logits is not None else None
    supplied = convert(probabilities, True) if probabilities is not None else None
    if inferred is not None and supplied is not None and any(not math.isclose(a, b, abs_tol=1e-6) for a, b in zip(inferred, supplied)):
        raise ValueError("conditional logits and probabilities disagree")
    return inferred if inferred is not None else supplied


@dataclass(frozen=True)
class PredictionSet:
    """Validated private data. Do not serialize this object into a public report."""

    rows: tuple[dict[str, Any], ...]
    groups: tuple[tuple[int, ...], ...]
    metadata: dict[str, Any]
    predictions: dict[str, tuple[int, ...]]
    class_to_group: tuple[int, ...]


def validate_predictions(rows: Sequence[Mapping[str, Any]], *, metadata: Mapping[str, Any] | None = None, groups: Any = None) -> PredictionSet:
    if not _sequence(rows) or not rows or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("predictions must contain nonempty mapping rows")
    meta = dict(metadata or {})
    # Per-row metadata is accepted for trainer exports; consistency is required.
    for key in ("mode", "model_mode", "trained_heads", "taxonomy", "groups", "protocol_sha256", "split", "selection_exposure", "eval_metadata"):
        present = [row[key] for row in rows if key in row]
        if present and (len(present) != len(rows) or any(x != present[0] for x in present)):
            raise ValueError("prediction metadata must be constant across rows")
        if present:
            if key in meta and meta[key] != present[0]:
                raise ValueError("row and sidecar metadata disagree")
            meta[key] = present[0]
    taxonomy = meta.get("taxonomy", meta.get("groups", CLINICAL_GROUPS))
    if isinstance(taxonomy, Mapping):
        taxonomy = taxonomy.get("groups")
    partition = validate_groups(taxonomy if groups is None else groups)
    if groups is not None and "taxonomy" in meta and validate_groups(taxonomy) != partition:
        raise ValueError("explicit and metadata taxonomies disagree")
    heads = meta.get("trained_heads")
    if not _sequence(heads) or not heads or any(x not in ("flat", "capacity", "group", "conditional") for x in heads) or len(set(heads)) != len(heads):
        raise ValueError("trained_heads metadata is required and must list known heads")
    mode = meta.get("mode", meta.get("model_mode", "unspecified"))
    if not isinstance(mode, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", mode):
        raise ValueError("model mode must be a safe category label")
    meta["mode"] = mode
    if meta.get("split") not in ("val", "validation", "test"):
        raise ValueError("evaluation split metadata must explicitly identify validation or test")
    meta["split"] = "val" if meta["split"] == "validation" else meta["split"]
    if not isinstance(meta.get("protocol_sha256"), str) or not re.fullmatch(r"[0-9a-fA-F]{64}", meta["protocol_sha256"]):
        raise ValueError("protocol_sha256 metadata must be a SHA-256 hex digest")
    group_map = [0] * NUM_CLASSES
    for index, group in enumerate(partition):
        for class_id in group:
            group_map[class_id] = index
    output_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    method_predictions: dict[str, list[int]] = {"main": []}
    if "flat" in heads:
        method_predictions["flat"] = []
        method_predictions["oracle_flat"] = []
    if "group" in heads and "conditional" in heads:
        for method in ("hard", "soft", "oracle_hierarchy"):
            method_predictions[method] = []
    for raw in rows:
        clip_id = raw.get("clip_id")
        group_id = raw.get("group_id")
        if not isinstance(clip_id, str) or not clip_id or not isinstance(group_id, str) or not group_id:
            raise ValueError("private clip and original recording group identifiers are required")
        if clip_id in seen:
            raise ValueError("duplicate clip identifiers are not allowed")
        seen.add(clip_id)
        target = raw.get("target", raw.get("label", raw.get("label_id")))
        if any(raw[key] != target for key in ("target", "label", "label_id") if key in raw):
            raise ValueError("target and label aliases disagree")
        if not isinstance(target, int) or isinstance(target, bool) or not 0 <= target < NUM_CLASSES:
            raise ValueError("targets must be integer canonical class IDs")
        source = raw.get("source", raw.get("source_collection"))
        if not isinstance(source, str) or not source:
            raise ValueError("private source category metadata is required")
        duration = raw.get("duration")
        if not _number(duration) or duration < 0:
            raise ValueError("clip duration must be finite and nonnegative")
        repeated = raw.get("repeated_frame_fraction")
        if repeated is not None and (not _number(repeated) or not 0 <= repeated <= 1):
            raise ValueError("repeated frame fraction must lie in [0, 1]")
        flat = _head(raw, "flat", NUM_CLASSES)
        router = _head(raw, "group", len(partition))
        conditional = _conditional(raw, partition)
        # Validate supplied outputs even when a head is untrained; never report
        # performance for an untrained head.
        if "flat" in heads and flat is None:
            raise ValueError("trained flat head outputs are missing")
        if ("group" in heads) != ("conditional" in heads):
            raise ValueError("hierarchical diagnostics require both trained group and conditional heads")
        if "group" in heads and (router is None or conditional is None):
            raise ValueError("trained hierarchy head outputs are missing")
        leaf = _head(raw, "leaf", NUM_CLASSES)
        if leaf is None:
            if flat is not None and "flat" in heads:
                leaf = flat
            elif router is not None and conditional is not None and "group" in heads:
                values = tuple(router[1][group_map[c]] * conditional[c] for c in range(NUM_CLASSES))
                leaf = (values, values)
            else:
                raise ValueError("main leaf probabilities or logits are required")
        main = _argmax(leaf[0])
        given_prediction = raw.get("prediction")
        if given_prediction is not None and (not isinstance(given_prediction, int) or isinstance(given_prediction, bool) or given_prediction != main):
            raise ValueError("saved prediction disagrees with main leaf scores")
        method_predictions["main"].append(main)
        if "flat" in heads:
            method_predictions["flat"].append(_argmax(flat[0]))
            method_predictions["oracle_flat"].append(_argmax(flat[0], partition[group_map[target]]))
        if "group" in heads:
            chosen_group = _argmax(router[0])
            method_predictions["hard"].append(_argmax(conditional, partition[chosen_group]))
            joint = tuple(router[1][group_map[c]] * conditional[c] for c in range(NUM_CLASSES))
            method_predictions["soft"].append(_argmax(joint))
            method_predictions["oracle_hierarchy"].append(_argmax(conditional, partition[group_map[target]]))
        output_rows.append({"clip_id": clip_id, "group_id": group_id, "target": target, "source": source,
                            "duration": float(duration), "repeated_frame_fraction": repeated,
                            "main_scores": leaf[0], "flat_scores": None if flat is None else flat[0],
                            "flat_probabilities": None if flat is None else flat[1],
                            "router_prediction": None if router is None else _argmax(router[0])})
    return PredictionSet(tuple(output_rows), partition, meta, {key: tuple(value) for key, value in method_predictions.items()}, tuple(group_map))


def _confusion(targets: Sequence[int], predictions: Sequence[int], *, size: int = NUM_CLASSES) -> list[list[int]]:
    matrix = [[0] * size for _ in range(size)]
    for target, prediction in zip(targets, predictions):
        matrix[target][prediction] += 1
    return matrix


def _metrics_from_confusion(matrix: Sequence[Sequence[int]], classes: Sequence[int] = tuple(range(NUM_CLASSES))) -> dict[str, Any]:
    size = len(matrix)
    support = [sum(row) for row in matrix]
    predicted = [sum(row[c] for row in matrix) for c in range(size)]
    count = sum(support)
    per_class = []
    for c in classes:
        tp = matrix[c][c]
        precision = tp / predicted[c] if predicted[c] else 0.0
        recall = tp / support[c] if support[c] else 0.0
        f1 = 2 * tp / (support[c] + predicted[c]) if support[c] + predicted[c] else 0.0
        per_class.append({"class_id": c, "class_name": CLASS_NAMES[c] if size == NUM_CLASSES else None,
                          "support": support[c], "precision": precision, "recall": recall, "f1": f1})
    selected_support = sum(support[c] for c in classes)
    selected_predicted = sum(predicted[c] for c in classes)
    selected_tp = sum(matrix[c][c] for c in classes)
    return {"num_samples": count, "accuracy": sum(matrix[c][c] for c in range(size)) / count if count else 0.0,
            "macro_f1": sum(c["f1"] for c in per_class) / len(classes),
            "macro_precision": sum(c["precision"] for c in per_class) / len(classes),
            "macro_recall": sum(c["recall"] for c in per_class) / len(classes),
            "balanced_accuracy": sum(c["recall"] for c in per_class) / len(classes),
            "present_class_macro_f1": sum(c["f1"] for c in per_class if c["support"]) / sum(bool(c["support"]) for c in per_class) if selected_support else 0.0,
            "weighted_f1": sum(c["support"] * c["f1"] for c in per_class) / selected_support if selected_support else 0.0,
            "weighted_precision": sum(c["support"] * c["precision"] for c in per_class) / selected_support if selected_support else 0.0,
            "weighted_recall": selected_tp / selected_support if selected_support else 0.0,
            "micro_precision": selected_tp / selected_predicted if selected_predicted else 0.0,
            "micro_recall": selected_tp / selected_support if selected_support else 0.0,
            "micro_f1": 2 * selected_tp / (selected_support + selected_predicted) if selected_support + selected_predicted else 0.0,
            "per_class": per_class, "confusion_matrix": [list(row) for row in matrix],
            "averaged_class_ids": list(classes), "confusion_axes": {"rows": "target", "columns": "prediction"}}


def _block(data: PredictionSet, predictions: Sequence[int], indices: Sequence[int] | None = None, classes: Sequence[int] = tuple(range(NUM_CLASSES))) -> dict[str, Any]:
    positions = list(range(len(data.rows))) if indices is None else indices
    return _metrics_from_confusion(_confusion([data.rows[i]["target"] for i in positions], [predictions[i] for i in positions]), classes)


def _safe_metadata(data: PredictionSet) -> dict[str, Any]:
    return {"mode": data.metadata["mode"], "trained_heads": list(data.metadata["trained_heads"]),
            "taxonomy_groups": [list(group) for group in data.groups], "protocol_sha256": data.metadata["protocol_sha256"],
            "split": data.metadata["split"], "independent_test_claim": False,
            "selection_exposure_reported": "selection_exposure" in data.metadata}


def _router_report(data: PredictionSet) -> dict[str, Any]:
    targets = [row["target"] for row in data.rows]
    actual = None
    if "group" in data.metadata["trained_heads"]:
        predicted_groups = [row["router_prediction"] for row in data.rows]
        true_groups = [data.class_to_group[target] for target in targets]
        matrix = _confusion(true_groups, predicted_groups, size=len(data.groups))
        correct_route = [i for i, (target, prediction) in enumerate(zip(true_groups, predicted_groups)) if target == prediction]
        hard = data.predictions["hard"]
        correct_final = sum(hard[i] == targets[i] for i in correct_route)
        actual = _metrics_from_confusion(matrix, tuple(range(len(data.groups))))
        actual.update({"accuracy": len(correct_route) / len(targets), "confusion_matrix": matrix,
                  "num_route_correct": len(correct_route), "num_route_errors": len(targets) - len(correct_route),
                  "conditional_fine_recall_given_correct_route": correct_final / len(correct_route) if correct_route else None,
                  "hard_final_accuracy": sum(a == b for a, b in zip(hard, targets)) / len(targets),
                  "per_class_conditional_recall": []})
        for class_id in range(NUM_CLASSES):
            eligible = [i for i in correct_route if targets[i] == class_id]
            actual["per_class_conditional_recall"].append({"class_id": class_id, "route_correct_support": len(eligible),
                                                         "recall": sum(hard[i] == class_id for i in eligible) / len(eligible) if eligible else None})
        five_positions = [i for i, target in enumerate(targets) if target in AMBIGUOUS_CLASSES]
        five_correct_route = [i for i in correct_route if targets[i] in AMBIGUOUS_CLASSES]
        actual["ground_truth_ambiguous_five"] = {"num_samples": len(five_positions), "route_correct_support": len(five_correct_route),
                                                "route_accuracy": len(five_correct_route) / len(five_positions) if five_positions else None,
                                                "conditional_fine_recall_given_correct_route": sum(hard[i] == targets[i] for i in five_correct_route) / len(five_correct_route) if five_correct_route else None}
    mapped = None
    aggregated_probability_route = None
    if "flat" in data.predictions:
        predictions = [data.class_to_group[c] for c in data.predictions["flat"]]
        true_groups = [data.class_to_group[c] for c in targets]
        mapped = _metrics_from_confusion(_confusion(true_groups, predictions, size=len(data.groups)), tuple(range(len(data.groups))))
        mapped["definition"] = "Group of the flat seven-class argmax; not a trained router."
        probability_predictions = [_argmax([sum(row["flat_probabilities"][c] for c in group) for group in data.groups]) for row in data.rows]
        aggregated_probability_route = _metrics_from_confusion(_confusion(true_groups, probability_predictions, size=len(data.groups)), tuple(range(len(data.groups))))
        aggregated_probability_route["definition"] = "Argmax of summed trained flat leaf probabilities in each group; distinct from mapping the flat leaf argmax."
    return {"actual_router": actual, "actual_router_unavailable_reason": None if actual else "group_head_not_trained",
            "flat_mapped_group": mapped, "flat_mapped_group_unavailable_reason": None if mapped else "flat_head_not_trained",
            "flat_aggregated_probability_group": aggregated_probability_route}


def _prediction_changes(targets: Sequence[int], baseline: Sequence[int], candidate: Sequence[int]) -> dict[str, int]:
    return {"num_samples": len(targets), "changed": sum(a != b for a, b in zip(baseline, candidate)),
            "corrected": sum(a != target and b == target for a, b, target in zip(baseline, candidate, targets)),
            "harmed": sum(a == target and b != target for a, b, target in zip(baseline, candidate, targets))}


def _error_decomposition(data: PredictionSet, indices: Sequence[int]) -> dict[str, Any] | None:
    if "hard" not in data.predictions:
        return None
    routing_errors = within_group_errors = correct = 0
    for i in indices:
        target = data.rows[i]["target"]
        route_wrong = data.rows[i]["router_prediction"] != data.class_to_group[target]
        if route_wrong:
            routing_errors += 1
        elif data.predictions["hard"][i] != target:
            within_group_errors += 1
        else:
            correct += 1
    return {"num_samples": len(indices), "routing_errors": routing_errors, "within_group_errors_given_correct_route": within_group_errors,
            "correct": correct, "total_hard_errors": routing_errors + within_group_errors,
            "definition": "Disjoint hard-routing error counts. Soft decoding may select a different group than router argmax and does not share this identity."}


def _slices(data: PredictionSet, *, min_samples: int, min_groups: int) -> dict[str, Any]:
    indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(data.rows):
        # Source strings can be paths or identifiers in private manifests. Public
        # aggregates never echo them and never publish an alias-to-source map.
        source_alias = "source-" + hashlib.sha256(row["source"].encode()).hexdigest()[:12]
        indices["source:" + source_alias].append(index)
        indices["duration:le_3s" if row["duration"] <= 3 else "duration:gt_3s_le_10s" if row["duration"] <= 10 else "duration:gt_10s"].append(index)
        repeated = row["repeated_frame_fraction"]
        if repeated is not None:
            indices["repeated_frames:none" if repeated == 0 else "repeated_frames:gt_0_le_0_25" if repeated <= 0.25 else "repeated_frames:gt_0_25"].append(index)
    output = {}
    for name, positions in sorted(indices.items()):
        group_count = len({data.rows[i]["group_id"] for i in positions})
        suppressed = len(positions) < min_samples or group_count < min_groups
        output[name] = {"num_samples": len(positions), "num_recording_groups": group_count, "suppressed": suppressed}
        if suppressed:
            output[name]["reason"] = "below_prespecified_minimum_samples_or_groups"
        else:
            output[name]["methods"] = {method: _block(data, predictions, positions) for method, predictions in data.predictions.items() if not method.startswith("oracle_")}
    return {"minimum_samples": min_samples, "minimum_recording_groups": min_groups,
            "repeated_frame_metadata_coverage": sum(row["repeated_frame_fraction"] is not None for row in data.rows),
            "results": output, "interpretation": "Descriptive slices, not evidence of held-out-source or patient generalization. Missing-class F1 is zero in fixed-seven macro."}


def _metric_report(data: PredictionSet, min_slice_samples: int, min_slice_groups: int) -> dict[str, Any]:
    if min_slice_samples < 1 or min_slice_groups < 2:
        raise ValueError("slice thresholds require at least one sample and two recording groups")
    ambiguous_positions = [i for i, row in enumerate(data.rows) if row["target"] in AMBIGUOUS_CLASSES]
    singleton_classes = tuple(c for group in data.groups if len(group) == 1 for c in group)
    singleton_positions = [i for i, row in enumerate(data.rows) if row["target"] in singleton_classes]
    methods = {method: _block(data, predictions) for method, predictions in data.predictions.items()}
    ambiguous = {method: _block(data, predictions, classes=AMBIGUOUS_CLASSES) for method, predictions in data.predictions.items() if not method.startswith("oracle_")}
    gt_restricted = {method: _block(data, predictions, ambiguous_positions, AMBIGUOUS_CLASSES) for method, predictions in data.predictions.items() if not method.startswith("oracle_")}
    renormalized = []
    for row in data.rows:
        renormalized.append(_argmax(row["main_scores"], AMBIGUOUS_CLASSES))
    main = methods["main"]
    singleton_correct = sum(data.predictions["main"][i] == data.rows[i]["target"] for i in singleton_positions)
    singleton_f1_sum = sum(main["per_class"][c]["f1"] for c in singleton_classes)
    return {"schema_version": SCHEMA_VERSION, "metadata": _safe_metadata(data),
            "num_samples": len(data.rows), "num_recording_groups": len({row["group_id"] for row in data.rows}),
            "methods": methods, "routing": _router_report(data),
            "ambiguous_five": {"class_ids": list(AMBIGUOUS_CLASSES), "num_ground_truth_five": len(ambiguous_positions), "methods": ambiguous,
                               "definition": "Average the five per-class F1 values from the full seven-class confusion matrix. Retain singleton-to-five false positives and five-to-singleton false negatives.",
                               "ground_truth_restricted_sensitivity": gt_restricted,
                               "ground_truth_restricted_warning": "Restricting ground truth to five classes removes singleton-to-five false positives. This is a sensitivity diagnostic, not the primary five-class F1.",
                               "subset_renormalized_main_diagnostic": _block(data, renormalized, ambiguous_positions, AMBIGUOUS_CLASSES),
                               "subset_renormalized_warning": "Truth-subset diagnostic changes the decision task; never substitute it for actual seven-class prediction performance."},
            "singleton_contribution": {"class_ids": list(singleton_classes), "num_samples": len(singleton_positions),
                                       "num_correct_main": singleton_correct, "accuracy_on_singletons": singleton_correct / len(singleton_positions) if singleton_positions else None,
                                       "contribution_to_seven_class_macro_f1": singleton_f1_sum / NUM_CLASSES,
                                       "main_seven_class_macro_f1_minus_singleton_terms": main["macro_f1"] - singleton_f1_sum / NUM_CLASSES},
            "slices": _slices(data, min_samples=min_slice_samples, min_groups=min_slice_groups),
            "hard_error_decomposition": {"all": _error_decomposition(data, list(range(len(data.rows)))),
                                         "ground_truth_ambiguous_five": _error_decomposition(data, ambiguous_positions)},
            "same_checkpoint_hard_soft": _prediction_changes([row["target"] for row in data.rows], data.predictions["hard"], data.predictions["soft"]) if "hard" in data.predictions else None,
            "oracle_warning": "Oracle hierarchy and oracle flat use the true group: explanatory bounds, not deployable predictions. A hierarchy with an untrained flat head needs a separately trained matched flat comparator.",
            "privacy": {"aggregate_only": True, "clip_ids_emitted": False, "recording_ids_emitted": False, "paths_emitted": False},
            "inference_or_fitting_performed": False, "p_values": None}


def metric_report(rows: Sequence[Mapping[str, Any]], *, metadata: Mapping[str, Any] | None = None, groups: Any = None,
                  min_slice_samples: int = 10, min_slice_groups: int = 2) -> dict[str, Any]:
    """Build aggregate diagnostics; never serialize private rows or identifiers."""
    data = validate_predictions(rows, metadata=metadata, groups=groups)
    return _metric_report(data, min_slice_samples, min_slice_groups)


def _matched(baseline: PredictionSet, candidate: PredictionSet) -> tuple[PredictionSet, PredictionSet]:
    for key in ("protocol_sha256", "split", "eval_metadata", "selection_exposure"):
        if baseline.metadata.get(key) != candidate.metadata.get(key):
            raise ValueError("paired evaluations must share protocol, split, exposure, and evaluation treatment metadata")
    left = {row["clip_id"]: index for index, row in enumerate(baseline.rows)}
    right = {row["clip_id"]: index for index, row in enumerate(candidate.rows)}
    if left.keys() != right.keys():
        raise ValueError("paired evaluations must have exactly the same clip identifiers; intersection matching is forbidden")
    for clip_id, index in left.items():
        for key in ("target", "group_id", "source", "duration", "repeated_frame_fraction"):
            if baseline.rows[index][key] != candidate.rows[right[clip_id]][key]:
                raise ValueError("paired clip targets and grouping/source/duration/frame metadata must agree exactly")
    order = [right[row["clip_id"]] for row in baseline.rows]
    aligned = PredictionSet(tuple(candidate.rows[i] for i in order), candidate.groups, candidate.metadata,
                            {method: tuple(values[i] for i in order) for method, values in candidate.predictions.items()}, candidate.class_to_group)
    return baseline, aligned


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * q
    low = math.floor(location)
    high = math.ceil(location)
    return ordered[low] * (high - location) + ordered[high] * (location - low) if low != high else ordered[low]


def paired_cluster_bootstrap(baseline: PredictionSet, candidate: PredictionSet, *, baseline_method: str = "main", candidate_method: str = "main",
                            replicates: int = 2000, seed: int = 42) -> dict[str, Any]:
    """Paired percentile intervals resampling complete original recording groups."""
    baseline, candidate = _matched(baseline, candidate)
    if baseline_method not in baseline.predictions or candidate_method not in candidate.predictions:
        raise ValueError("requested comparison uses an unavailable or untrained prediction head")
    oracle_taxonomy = None
    if baseline_method == "oracle_flat" and candidate_method in ("oracle_flat", "oracle_hierarchy"):
        oracle_taxonomy = [list(group) for group in candidate.groups]
        predictions = dict(baseline.predictions)
        predictions["oracle_flat"] = tuple(_argmax(row["flat_scores"], candidate.groups[candidate.class_to_group[row["target"]]]) for row in baseline.rows)
        baseline = PredictionSet(baseline.rows, baseline.groups, baseline.metadata, predictions, baseline.class_to_group)
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 2:
        raise ValueError("bootstrap requires at least two replicates")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("bootstrap seed must be an integer")
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(baseline.rows):
        clusters[row["group_id"]].append(index)
    if len(clusters) < 2:
        raise ValueError("paired cluster uncertainty requires at least two original recording groups")
    # Group order is independent of input row order and private identifier text.
    positions = sorted(clusters.values(), key=lambda values: min(values))
    targets = [row["target"] for row in baseline.rows]
    matrices = []
    for indices in positions:
        matrices.append((_confusion([targets[i] for i in indices], [baseline.predictions[baseline_method][i] for i in indices]),
                         _confusion([targets[i] for i in indices], [candidate.predictions[candidate_method][i] for i in indices])))
    metrics = ("macro_f1", "five_macro_f1", "weighted_f1", "accuracy", "micro_f1")
    observed_baseline = _block(baseline, baseline.predictions[baseline_method])
    observed_candidate = _block(candidate, candidate.predictions[candidate_method])
    observed_baseline["five_macro_f1"] = _block(baseline, baseline.predictions[baseline_method], classes=AMBIGUOUS_CLASSES)["macro_f1"]
    observed_candidate["five_macro_f1"] = _block(candidate, candidate.predictions[candidate_method], classes=AMBIGUOUS_CLASSES)["macro_f1"]
    draws = {metric: [] for metric in metrics}
    missing_class_replicates = 0
    rng = random.Random(seed)
    for _ in range(replicates):
        totals = [[[0] * NUM_CLASSES for _ in range(NUM_CLASSES)] for _ in range(2)]
        for _ in positions:
            chosen = matrices[rng.randrange(len(matrices))]
            for arm in range(2):
                for row in range(NUM_CLASSES):
                    for column in range(NUM_CLASSES):
                        totals[arm][row][column] += chosen[arm][row][column]
        if any(sum(row) == 0 for row in totals[0]):
            missing_class_replicates += 1
        reports = [_metrics_from_confusion(matrix) for matrix in totals]
        for report, matrix in zip(reports, totals):
            report["five_macro_f1"] = _metrics_from_confusion(matrix, AMBIGUOUS_CLASSES)["macro_f1"]
        for metric in metrics:
            draws[metric].append(reports[1][metric] - reports[0][metric])
    return {"method": "paired_original_recording_cluster_percentile_bootstrap", "num_recording_groups": len(positions),
            "num_clips": len(targets), "replicates": replicates, "seed": seed,
            "baseline_method": baseline_method, "candidate_method": candidate_method,
            "oracle_shared_truth_group_taxonomy": oracle_taxonomy,
            "metrics": {metric: {"baseline": observed_baseline[metric], "candidate": observed_candidate[metric],
                                 "paired_delta": observed_candidate[metric] - observed_baseline[metric],
                                 "percentile_95_interval": [_quantile(draws[metric], 0.025), _quantile(draws[metric], 0.975)]} for metric in metrics},
            "replicates_missing_at_least_one_class": missing_class_replicates,
            "missing_class_rule": "Always average the same seven classes in both arms; undefined F1 is zero. No replicates are selectively dropped.",
            "interpretation": "Uncertainty conditional on these checkpoints and recording groups; distinct from variation across training seeds. Reused development validation is exploratory.",
            "p_values": None}


def compare_predictions(baseline_rows: Sequence[Mapping[str, Any]], candidate_rows: Sequence[Mapping[str, Any]], *,
                        baseline_metadata: Mapping[str, Any] | None = None, candidate_metadata: Mapping[str, Any] | None = None,
                        baseline_groups: Any = None, candidate_groups: Any = None, baseline_method: str = "main", candidate_method: str = "main",
                        bootstrap_replicates: int = 2000, seed: int = 42, min_slice_samples: int = 10, min_slice_groups: int = 2) -> dict[str, Any]:
    baseline, candidate = _matched(validate_predictions(baseline_rows, metadata=baseline_metadata, groups=baseline_groups),
                                   validate_predictions(candidate_rows, metadata=candidate_metadata, groups=candidate_groups))
    fair_flat_predictions = None
    if "flat" in baseline.metadata["trained_heads"]:
        fair_flat_predictions = tuple(_argmax(row["flat_scores"], candidate.groups[candidate.class_to_group[row["target"]]]) for row in baseline.rows)
    bootstrap_baseline = baseline
    if baseline_method == "oracle_flat" and candidate_method in ("oracle_hierarchy", "oracle_flat") and fair_flat_predictions is not None:
        predictions = dict(baseline.predictions)
        predictions["oracle_flat"] = fair_flat_predictions
        bootstrap_baseline = PredictionSet(baseline.rows, baseline.groups, baseline.metadata, predictions, baseline.class_to_group)
    interval = paired_cluster_bootstrap(bootstrap_baseline, candidate, baseline_method=baseline_method, candidate_method=candidate_method, replicates=bootstrap_replicates, seed=seed)
    if bootstrap_baseline is not baseline:
        interval["oracle_shared_truth_group_taxonomy"] = [list(group) for group in candidate.groups]
    targets = [row["target"] for row in baseline.rows]
    left = bootstrap_baseline.predictions[baseline_method]
    right = candidate.predictions[candidate_method]
    fair_flat = None
    if fair_flat_predictions is not None:
        fair_flat = _block(baseline, fair_flat_predictions)
    return {"schema_version": SCHEMA_VERSION, "baseline": _metric_report(baseline, min_slice_samples, min_slice_groups),
            "candidate": _metric_report(candidate, min_slice_samples, min_slice_groups), "paired_uncertainty": interval,
            "paired_prediction_changes": _prediction_changes(targets, left, right),
            "fair_oracle_comparison": {"baseline_trained_flat": fair_flat,
                                       "candidate_trained_hierarchy": _block(candidate, candidate.predictions["oracle_hierarchy"]) if "oracle_hierarchy" in candidate.predictions else None,
                                       "shared_truth_group_taxonomy": [list(group) for group in candidate.groups],
                                       "baseline_grouping_recomputed": baseline.groups != candidate.groups,
                                       "warning": "Both use true taxonomy groups. Compare only matched information budgets; these are not deployed accuracies."},
            "privacy": {"aggregate_only": True, "clip_ids_emitted": False, "recording_ids_emitted": False, "paths_emitted": False}}


def summarize_seed_differences(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report training-seed variation separately; no n=3 significance claim."""
    if not pairs or any(not isinstance(row.get("seed"), int) or isinstance(row.get("seed"), bool) or not _number(row.get("baseline")) or not _number(row.get("candidate")) or not 0 <= row["baseline"] <= 1 or not 0 <= row["candidate"] <= 1 for row in pairs):
        raise ValueError("seed summaries require finite matched seed metrics")
    if len({row["seed"] for row in pairs}) != len(pairs):
        raise ValueError("seed summaries must not repeat training seeds")
    deltas = [row["candidate"] - row["baseline"] for row in pairs]
    mean = sum(deltas) / len(deltas)
    sd = math.sqrt(sum((delta - mean) ** 2 for delta in deltas) / (len(deltas) - 1)) if len(deltas) > 1 else None
    return {"num_training_seeds": len(pairs), "seeds": [row["seed"] for row in pairs],
            "baseline_mean": sum(row["baseline"] for row in pairs) / len(pairs),
            "candidate_mean": sum(row["candidate"] for row in pairs) / len(pairs), "paired_delta_mean": mean,
            "paired_delta_sample_sd": sd, "p_values": None,
            "interpretation": "Training-seed summary, not clip/bootstrap sample replication or a significance claim."}


def derive_visual_groups(train_confusion: Sequence[Sequence[int]], *, provenance: Mapping[str, Any], sizes: Sequence[int] = (1, 1, 3, 2)) -> dict[str, Any]:
    """Choose a constrained grouping from TRAIN-ONLY out-of-fold confusion.

    The caller must establish OOF provenance. No evaluation predictions, held-out
    labels, or validation/test confusion may be supplied to this helper.
    """
    if provenance.get("split") != "train" or provenance.get("out_of_fold") is not True or provenance.get("held_out_evaluation_used") is not False:
        raise ValueError("visual grouping requires explicit train-only out-of-fold provenance")
    if not _sequence(train_confusion) or len(train_confusion) != NUM_CLASSES or any(not _sequence(row) or len(row) != NUM_CLASSES for row in train_confusion):
        raise ValueError("training confusion must be a seven-by-seven matrix")
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for row in train_confusion for x in row):
        raise ValueError("training confusion counts must be nonnegative integers")
    if not sizes or any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in sizes) or sum(sizes) != NUM_CLASSES:
        raise ValueError("visual grouping sizes must sum to seven")
    support = [sum(row) for row in train_confusion]
    if any(value == 0 for value in support):
        raise ValueError("training OOF confusion must include support for every class")
    similarity = [[train_confusion[a][b] / support[a] + train_confusion[b][a] / support[b] for b in range(NUM_CLASSES)] for a in range(NUM_CLASSES)]

    def partitions(remaining: tuple[int, ...], remaining_sizes: tuple[int, ...]):
        if not remaining_sizes:
            yield ()
            return
        for selected in itertools.combinations(remaining, remaining_sizes[0]):
            rest = tuple(c for c in remaining if c not in selected)
            for tail in partitions(rest, remaining_sizes[1:]):
                yield (selected,) + tail

    candidates = partitions(tuple(range(NUM_CLASSES)), tuple(sizes))
    best = None
    best_score = -1.0
    for grouping in candidates:
        score = sum(similarity[a][b] for group in grouping for a, b in itertools.combinations(group, 2))
        if score > best_score or (score == best_score and (best is None or grouping < best)):
            best, best_score = grouping, score
    digest = hashlib.sha256(json.dumps([list(row) for row in train_confusion], separators=(",", ":")).encode()).hexdigest()
    return {"groups": [list(group) for group in best], "selection_source": "train_only_out_of_fold_confusion", "group_sizes": list(sizes),
            "objective": "sum of symmetrized row-normalized within-group confusions", "objective_value": best_score,
            "training_confusion_sha256": digest, "held_out_evaluation_used": False}


def read_predictions(path: str | Path, metadata_path: str | Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read private JSONL and explicit metadata; parsing errors reveal no raw text."""
    try:
        with Path(path).open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if metadata_path is None:
            sibling = Path(path).with_name("prediction_metadata.json")
            metadata_path = sibling if sibling.exists() else None
        metadata = {} if metadata_path is None else json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        raise ValueError("unable to parse private prediction input or metadata") from None
    if not isinstance(metadata, dict):
        raise ValueError("prediction metadata must be a mapping")
    return rows, metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    aggregate = commands.add_parser("aggregate", help="Private predictions to safe aggregate diagnostics; never runs inference")
    aggregate.add_argument("--predictions", required=True)
    aggregate.add_argument("--metadata")
    compare = commands.add_parser("compare", help="Strict paired recording-group bootstrap; never fits or chooses models")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--baseline-metadata")
    compare.add_argument("--candidate-metadata")
    compare.add_argument("--baseline-method", default="main", choices=("main", "flat", "hard", "soft", "oracle_flat", "oracle_hierarchy"))
    compare.add_argument("--candidate-method", default="main", choices=("main", "flat", "hard", "soft", "oracle_flat", "oracle_hierarchy"))
    compare.add_argument("--bootstrap-replicates", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=42)
    for command in (aggregate, compare):
        command.add_argument("--output", required=True)
        command.add_argument("--min-slice-samples", type=int, default=10)
        command.add_argument("--min-slice-groups", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        if args.command == "aggregate":
            rows, metadata = read_predictions(args.predictions, args.metadata)
            report = metric_report(rows, metadata=metadata, min_slice_samples=args.min_slice_samples, min_slice_groups=args.min_slice_groups)
        else:
            baseline_rows, baseline_meta = read_predictions(args.baseline, args.baseline_metadata)
            candidate_rows, candidate_meta = read_predictions(args.candidate, args.candidate_metadata)
            report = compare_predictions(baseline_rows, candidate_rows, baseline_metadata=baseline_meta, candidate_metadata=candidate_meta,
                                         baseline_method=args.baseline_method, candidate_method=args.candidate_method,
                                         bootstrap_replicates=args.bootstrap_replicates, seed=args.seed,
                                         min_slice_samples=args.min_slice_samples, min_slice_groups=args.min_slice_groups)
        serialized = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        output = Path(args.output)
        inputs = [Path(value).resolve() for key, value in vars(args).items() if key in ("predictions", "metadata", "baseline", "candidate", "baseline_metadata", "candidate_metadata") and value]
        inputs.extend(Path(value).with_name("prediction_metadata.json").resolve() for key, value in vars(args).items() if key in ("predictions", "baseline", "candidate") and value)
        if output.resolve() in inputs:
            raise ValueError("safe aggregate output must not overwrite private input")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(output)
    except (ValueError, TypeError, OSError) as exc:
        # OSError filenames and JSON snippets can include private source paths.
        message = str(exc) if isinstance(exc, ValueError) else "prediction analysis failed validation or output writing"
        parser.exit(2, "error: " + message + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
