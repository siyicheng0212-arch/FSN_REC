"""Paired development evaluation for legacy/reference context utility models.

Structural cuts are applied before model calls, but a singleton is still passed
to the chosen backbone. This intentionally preserves legacy singleton semantics
instead of imposing the new reference model's exact frozen-A fallback. Reports
contain aggregates only; returned prediction rows remain private.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from collections import Counter, defaultdict

import numpy as np
import torch

from experiments.fsn_tcn.evaluate import validate_evaluation_chains
from experiments.fsn_tcn.io import load_chain
from experiments.relation.evaluate_suite import classification_metrics


_MODES = {"learned", "unit", "permuted"}


def _paired(labels, prediction, baseline):
    changed = prediction != baseline
    improved = int(((baseline != labels) & (prediction == labels)).sum())
    worsened = int(((baseline == labels) & (prediction != labels)).sum())
    return {"changed_count": int(changed.sum()),
            "changed_fraction": float(changed.mean()) if len(labels) else 0.,
            "improved": improved, "worsened": worsened,
            "wrong_to_different_wrong": int(((baseline != labels) & (prediction != labels) & changed).sum()),
            "net_correct_gain": improved - worsened}


def _directions(labels, prediction):
    return {name: {"count": int(((labels == left) & (prediction == right)).sum()),
                   "true_class_support": int((labels == left).sum()),
                   "rate": float(((labels == left) & (prediction == right)).sum() / (labels == left).sum())
                   if (labels == left).any() else 0.}
            for left, right, name in ((3, 4, "sweep_to_reperfusion"), (4, 3, "reperfusion_to_sweep"))}


def _f1_delta(labels, current, baseline):
    now = classification_metrics(labels, current)["per_class"]
    before = classification_metrics(labels, baseline)["per_class"]
    return [{"label_id": i, "support": now[i]["support"],
             "f1_delta": now[i]["f1"] - before[i]["f1"],
             "macro_f1_contribution": (now[i]["f1"] - before[i]["f1"]) / 7}
            for i in range(7)]


def _metrics(labels, prediction, visual):
    return {**classification_metrics(labels, prediction),
            "paired_vs_A": _paired(labels, prediction, visual),
            "sweep_reperfusion": _directions(labels, prediction),
            "per_class_delta_vs_A": _f1_delta(labels, prediction, visual)}


def _segments(eligible, size):
    start = 0
    for i, active in enumerate(eligible):
        if not active:
            yield start, i + 1
            start = i + 1
    yield start, size


def _segment_seed(seed, chain_id, start):
    token = f"{seed}:{chain_id}:{start}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % (2**31)


def _score(model, features, a_logits, gate_mode, permutation_seed):
    signature = inspect.signature(model.forward)
    parameters = signature.parameters
    arbitrary = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    options = {}
    if arbitrary or "gate_mode" in parameters:
        options["gate_mode"] = gate_mode
    if arbitrary or "permutation_seed" in parameters:
        options["permutation_seed"] = permutation_seed
    output = model(features, a_logits, **options)
    if output.shape != (len(features), 7) or not bool(torch.isfinite(output).all()):
        raise ValueError("context model must output finite logits with shape [clips,7]")
    return output


def _statistics(values):
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None,
                "quantiles": None, "fraction_below_one": None, "fraction_above_one": None}
    return {"count": len(array), "mean": float(array.mean()), "std": float(array.std()),
            "min": float(array.min()), "max": float(array.max()),
            "quantiles": {str(q): float(np.quantile(array, q)) for q in (.05, .25, .5, .75, .95)},
            "fraction_below_one": float((array < 1).mean()),
            "fraction_above_one": float((array > 1).mean())}


def _audit_entries(value):
    """Accept the documented wrapper audit; never copy arbitrary audit fields."""
    if value is None:
        return [], {}
    if isinstance(value, list):
        return value, {}
    if isinstance(value, dict):
        entries = value.get("entries", value.get("gates", []))
        if not isinstance(entries, list):
            raise ValueError("last_gate_audit entries must be a JSON list")
        return entries, value.get("permutation", {})
    raise ValueError("last_gate_audit must be a JSON list or mapping")


def _model_audit_entries(model, path_indices):
    coefficients = getattr(model, "last_coefficients", None)
    if callable(coefficients):
        entries, shuffled = [], 0
        for block in coefficients():
            if not isinstance(block, dict):
                raise ValueError("last_coefficients must contain mappings")
            path = block.get("path")
            if not isinstance(path, str):
                raise ValueError("coefficient module path must be a string")
            if path not in path_indices:
                path_indices[path] = len(path_indices)
            values = torch.as_tensor(block["values"]).detach().flatten().cpu().tolist()
            targets = torch.as_tensor(block["target_positions"]).detach().flatten().cpu().tolist()
            originals = torch.as_tensor(block.get("original_values", block["values"])).detach().flatten().cpu().tolist()
            if len(values) != len(targets) or len(originals) != len(values):
                raise ValueError("coefficient values, target positions and original values must align")
            moved = block.get("shuffled_positions", 0)
            if type(moved) is not int or not 0 <= moved <= len(values):
                raise ValueError("shuffled position count must lie within its coefficient group")
            shuffled += moved
            offset = block["offset"]
            for value, target, original in zip(values, targets, originals):
                entries.append({"layer": path_indices[path], "offset": offset,
                                "target": target, "neighbor": target + offset,
                                "value": value, "original_value": original})
        return entries, {"shuffled_index_positions": shuffled}
    value = getattr(model, "last_gate_audit", None)
    return _audit_entries(value() if callable(value) else value)


def _collect_audit(model, ordered, metadata, start, stop, breaks, values, groups, permutation, path_indices):
    entries, permutation_summary = _model_audit_entries(model, path_indices)
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("gate audit entries must be mappings")
        raw = entry.get("value", entry.get("coefficient"))
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            raise ValueError("gate audit coefficients must be finite JSON numbers")
        value = float(raw)
        values.append(value)
        offset = entry.get("offset", 0)
        if type(offset) is not int:
            raise ValueError("gate audit offset must be an integer")
        span = entry.get("span", abs(offset))
        if type(span) is not int or span < 0:
            raise ValueError("gate audit span must be a nonnegative integer")
        side = "left" if offset < 0 else "right" if offset > 0 else "unspecified"
        groups[("side", side)].append(value)
        groups[("span", str(span))].append(value)
        layer = entry.get("layer", entry.get("layer_index", "unspecified"))
        if type(layer) not in {int, str}:
            raise ValueError("gate audit layer must be an integer or string")
        # Only numeric layer indices are emitted as grouping labels. Arbitrary
        # strings could otherwise smuggle source paths into public reports.
        layer = str(layer) if type(layer) is int else "unspecified"
        groups[("layer", layer)].append(value)
        target, neighbor = entry.get("target"), entry.get("neighbor")
        if target is not None or neighbor is not None:
            if (type(target) is not int or type(neighbor) is not int
                    or not 0 <= target < stop - start or not 0 <= neighbor < stop - start):
                raise ValueError("gate audit endpoint indices must belong to the current segment")
            target_meta, neighbor_meta = (metadata[ordered[start + index]] for index in (target, neighbor))
            source = target_meta["source_kind"] if target_meta["source_kind"] == neighbor_meta["source_kind"] else "mixed"
            groups[("source", source)].append(value)
            left, right = sorted((start + target, start + neighbor))
            groups[("connection", "crosses_synthetic_break" if any(breaks[left:right]) else "no_synthetic_break")].append(value)
        if "original_value" in entry:
            original = entry["original_value"]
            if isinstance(original, bool) or not isinstance(original, (int, float)) or not math.isfinite(original):
                raise ValueError("original gate audit values must be finite JSON numbers")
            permutation["compared_positions"] += 1
            permutation["changed_positions"] += int(value != float(original))
    # Summaries are optional when the wrapper does not expose per-position
    # original values; only known integer count fields are admitted.
    if not any("original_value" in row for row in entries) and isinstance(permutation_summary, dict):
        for key in ("compared_positions", "changed_positions"):
            count = permutation_summary.get(key, 0)
            if type(count) is not int or count < 0:
                raise ValueError("gate permutation counts must be nonnegative integers")
            permutation[key] += count
    if isinstance(permutation_summary, dict):
        moved = permutation_summary.get("shuffled_index_positions", 0)
        if type(moved) is not int or moved < 0:
            raise ValueError("shuffled index position count must be a nonnegative integer")
        permutation["shuffled_index_positions"] += moved


def _rows_map(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("paired comparison requires nonempty private prediction rows")
    mapped = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("prediction row must be a mapping")
        clip = row.get("clip_id")
        if not isinstance(clip, str) or not clip or clip in mapped:
            raise ValueError("paired prediction clip identifiers must be unique")
        if row.get("split") != "val":
            raise ValueError("paired comparison accepts val rows only")
        for field in ("label_id", "prediction", "A_prediction"):
            if type(row.get(field)) is not int or row[field] not in range(7):
                raise ValueError("paired row classes must be seven-class integer identifiers")
        mapped[clip] = row
    return mapped


def summarize_comparison(current_rows, baseline_rows):
    """Strict clip-paired aggregate comparison; neither row list is published.

    Preserved baseline rescues mean clips on which the baseline corrected A and
    the current model remains correct. Wrong-to-different-wrong changes are
    counted explicitly, so correction/spoil counts never imply total changes.
    """
    current, baseline = _rows_map(current_rows), _rows_map(baseline_rows)
    if current.keys() != baseline.keys():
        raise ValueError("paired comparisons require identical complete clip coverage")
    ids = sorted(current)
    for clip in ids:
        if any(current[clip].get(field) != baseline[clip].get(field)
               for field in ("label_id", "A_prediction", "source_kind", "duration")):
            raise ValueError("paired labels, frozen-A predictions and metadata must align")
    labels = np.asarray([current[clip]["label_id"] for clip in ids], dtype=int)
    visual = np.asarray([current[clip]["A_prediction"] for clip in ids], dtype=int)
    prediction = np.asarray([current[clip]["prediction"] for clip in ids], dtype=int)
    before = np.asarray([baseline[clip]["prediction"] for clip in ids], dtype=int)
    baseline_rescues = (visual != labels) & (before == labels)
    baseline_spoils = (visual == labels) & (before != labels)
    now_metrics, old_metrics = classification_metrics(labels, prediction), classification_metrics(labels, before)
    return {"count": len(ids), "paired": _paired(labels, prediction, before),
            "accuracy_delta": now_metrics["accuracy"] - old_metrics["accuracy"],
            "macro_f1_delta": now_metrics["macro_f1"] - old_metrics["macro_f1"],
            "per_class_delta": _f1_delta(labels, prediction, before),
            "baseline_rescues_vs_A": int(baseline_rescues.sum()),
            "baseline_rescues_preserved": int((baseline_rescues & (prediction == labels)).sum()),
            "baseline_rescues_lost": int((baseline_rescues & (prediction != labels)).sum()),
            "baseline_spoils_vs_A": int(baseline_spoils.sum()),
            "baseline_spoils_repaired": int((baseline_spoils & (prediction == labels)).sum()),
            "current_sweep_reperfusion": _directions(labels, prediction),
            "baseline_sweep_reperfusion": _directions(labels, before)}


@torch.no_grad()
def evaluate_chains(bundle, chains, model, *, baseline=None, device="cpu",
                    gate_mode="learned", permutation_seed=20261002):
    """Return safe aggregates and private rows for full development validation."""
    if gate_mode not in _MODES:
        raise ValueError("gate_mode must be learned, unit or permuted")
    if type(permutation_seed) is not int:
        raise ValueError("permutation_seed must be an integer")
    prepared = validate_evaluation_chains(bundle, chains)
    model = model.to(device).eval()
    if baseline is not None:
        baseline = baseline.to(device).eval()
    rows, baseline_rows = [], []
    counts = Counter(chains=0, clips=0, structural_cuts=0, candidate_edges=0,
                     synthetic_breaks=0, segments=0, singleton_segments=0)
    gate_values, gate_groups, permutation, path_indices = [], defaultdict(list), Counter(), {}
    for chain, ordered, eligible, breaks, sources in prepared:
        features, a_logits, label_tensor = load_chain(bundle, chain, device)
        if a_logits.shape != (len(ordered), 7) or not bool(torch.isfinite(a_logits).all()):
            raise ValueError("frozen A must provide finite seven-class logits")
        if features.ndim != 2 or not bool(torch.isfinite(features).all()):
            raise ValueError("frozen visual features must be a finite [clips,features] tensor")
        outputs, baseline_outputs = [], []
        segment_lengths = []
        for start, stop in _segments(eligible, len(ordered)):
            seed = _segment_seed(permutation_seed, chain["chain_id"], start)
            outputs.append(_score(model, features[start:stop], a_logits[start:stop], gate_mode, seed))
            _collect_audit(model, ordered, bundle.metadata, start, stop, breaks,
                           gate_values, gate_groups, permutation, path_indices)
            if baseline is not None:
                baseline_outputs.append(_score(baseline, features[start:stop], a_logits[start:stop], "unit", seed))
            segment_lengths.extend([stop - start] * (stop - start))
            counts["segments"] += 1
            counts["singleton_segments"] += int(stop - start == 1)
        output = torch.cat(outputs)
        losses = torch.nn.functional.cross_entropy(output.float(), label_tensor, reduction="none")
        a_losses = torch.nn.functional.cross_entropy(a_logits.float(), label_tensor, reduction="none")
        if not bool(torch.isfinite(losses).all()) or not bool(torch.isfinite(a_losses).all()):
            raise ValueError("development cross entropy must remain finite")
        prediction = output.argmax(-1).cpu().tolist()
        visual = a_logits.argmax(-1).cpu().tolist()
        labels = label_tensor.cpu().tolist()
        initial_prediction = torch.cat(baseline_outputs).argmax(-1).cpu().tolist() if baseline is not None else None
        for i, clip in enumerate(ordered):
            metadata = bundle.metadata[clip]
            if type(labels[i]) is not int or labels[i] != metadata["label_id"]:
                raise ValueError("feature labels must match sealed metadata")
            duration = metadata.get("duration", metadata.get("clip_duration_sec"))
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
                raise ValueError("duration must be finite and positive")
            row = {"clip_id": clip, "chain_id": chain["chain_id"], "split": "val",
                   "label_id": labels[i], "prediction": int(prediction[i]), "A_prediction": int(visual[i]),
                   "source_kind": metadata["source_kind"], "duration": float(duration),
                   "recording_key": metadata["recording_key"], "segment_length": segment_lengths[i],
                   "cross_entropy": float(losses[i]), "A_cross_entropy": float(a_losses[i])}
            rows.append(row)
            if initial_prediction is not None:
                baseline_rows.append({**row, "prediction": int(initial_prediction[i])})
        counts["chains"] += 1
        counts["clips"] += len(ordered)
        counts["structural_cuts"] += sum(not flag for flag in eligible)
        counts["candidate_edges"] += sum(eligible)
        counts["synthetic_breaks"] += sum(breaks)
    labels = np.asarray([row["label_id"] for row in rows], dtype=int)
    prediction = np.asarray([row["prediction"] for row in rows], dtype=int)
    visual = np.asarray([row["A_prediction"] for row in rows], dtype=int)
    sources = np.asarray([row["source_kind"] for row in rows])
    durations = np.asarray([row["duration"] for row in rows])
    masks = {"clinical": sources == "clinical", "network": sources == "network",
             "duration_le_1s": durations <= 1., "duration_gt_1_le_5s": (durations > 1.) & (durations <= 5.),
             "duration_gt_5s": durations > 5.}
    by_record = defaultdict(list)
    for i, row in enumerate(rows):
        by_record[row["recording_key"]].append(i)
    # Record identity and even original traversal order stay private.
    record_metrics = [_metrics(labels[idx], prediction[idx], visual[idx]) for idx in by_record.values()]
    record_metrics.sort(key=lambda row: (row["count"], row["accuracy"], row["macro_f1"]))
    result = {"schema_version": "context_utility_evaluation_v1",
              "evaluation_role": "development_validation_not_independent_test",
              "condition": "synthetic_disruptions" if prepared[0][0].get("synthetic_chain", False) else "sealed_original_chains",
              "synthetic_evidence_scope": "constructed disruptions; not independently verified real editing boundaries",
              "gate_mode": gate_mode, "permutation_seed": permutation_seed,
              "A_frozen": True, "all_val_clips_evaluated_once": True,
              "singleton_semantics": "chosen backbone called on singleton; no evaluator-imposed A fallback",
              "counts": dict(counts), "metrics": _metrics(labels, prediction, visual),
              "A_metrics": classification_metrics(labels, visual),
              "cross_entropy": float(np.mean([row["cross_entropy"] for row in rows])),
              "A_cross_entropy": float(np.mean([row["A_cross_entropy"] for row in rows])),
              "cross_entropy_definition": "unweighted seven-class CE averaged over all val clips",
              "slices": {name: _metrics(labels[mask], prediction[mask], visual[mask]) for name, mask in masks.items()},
              "recording_aggregate": {"count": len(record_metrics), "identity_removed_metrics": record_metrics,
                  "mean_accuracy": float(np.mean([row["accuracy"] for row in record_metrics])),
                  "mean_macro_f1": float(np.mean([row["macro_f1"] for row in record_metrics])),
                  "macro_f1_definition": "seven-class macro per recording, absent classes assigned zero"},
              "gate_audit": {"overall": _statistics(gate_values),
                  "groups": {kind: {key: _statistics(values) for (group, key), values in sorted(gate_groups.items()) if group == kind}
                             for kind in ("layer", "side", "span", "source", "connection")},
                  "denominator": "valid non-center convolution contributions, not clips or adjacent candidate edges",
                  "permutation": {"compared_positions": permutation["compared_positions"],
                                  "changed_positions": permutation["changed_positions"],
                                  "shuffled_index_positions": permutation["shuffled_index_positions"]}}}
    if baseline is not None:
        result["paired_vs_initial_TCN"] = summarize_comparison(rows, baseline_rows)
    return result, rows


def evaluate_same_checkpoint_gate_modes(bundle, chains, model, *, device="cpu", permutation_seed=20261002):
    """Learned/unit/permuted evaluations of one model, without retraining."""
    reports, private = {}, {}
    for mode in ("learned", "unit", "permuted"):
        reports[mode], private[mode] = evaluate_chains(bundle, chains, model, device=device,
                                                     gate_mode=mode, permutation_seed=permutation_seed)
    reports["paired_learned_vs_unit"] = summarize_comparison(private["learned"], private["unit"])
    reports["paired_learned_vs_permuted"] = summarize_comparison(private["learned"], private["permuted"])
    return reports, private
