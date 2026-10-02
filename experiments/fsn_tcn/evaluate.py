"""Paired FSN-TCN development evaluation without exporting identifiers.

All methods consume the same frozen-A evidence. Ordinary chains are sealed
source candidates; synthetic chains are separate, explicitly declared controls.
The latter establish robustness to constructed disruptions, not detection of
real editing boundaries. Private prediction rows are returned separately and
must never be included in a public aggregate report.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import torch

from experiments.relation.evaluate_suite import classification_metrics
from .io import load_chain
from .model import SegmentedTCN
from .gates import select_gates


_GROUP_FIELDS = ("group_id", "recording_key", "source_video_key", "timebase_key")
_SYNTHETIC_ORIGINS = {
    "fsn_tcn_clinical_shuffle_v1",
    "fsn_tcn_clinical_cross_record_v1",
}


def _boolean_vector(value, count, name):
    if not isinstance(value, list) or len(value) != count or any(type(x) is not bool for x in value):
        raise ValueError(f"{name} must contain exactly one boolean per adjacency")
    return value


def _pair(left, right):
    return left, right


def _canonical(bundle):
    """Return the sealed development chains and eligible original pairs."""
    sealed, legal, all_pairs = {}, set(), {}
    for chain in bundle.chains["val"]:
        chain_id, ordered = chain.get("chain_id"), chain.get("ordered_clip_ids")
        if not isinstance(chain_id, str) or not chain_id or chain_id in sealed:
            raise ValueError("sealed val chain identifiers must be unique")
        if not isinstance(ordered, list) or not ordered or len(set(ordered)) != len(ordered):
            raise ValueError("sealed val chains require unique ordered clip identifiers")
        eligible = _boolean_vector(chain.get("eligible"), len(ordered) - 1, "sealed eligible")
        sealed[chain_id] = (ordered, eligible)
        for left, right, flag in zip(ordered, ordered[1:], eligible):
            if (left, right) in all_pairs:
                raise ValueError("sealed val adjacency must not be duplicated")
            all_pairs[left, right] = flag
            if flag:
                legal.add((left, right))
    return sealed, legal, all_pairs


def validate_evaluation_chains(bundle, chains):
    """Validate coverage and admissible adjacencies before reading features.

    Ordinary input chains must exactly match the sealed development chains.
    Synthetic controls may rearrange their clips only with declared break
    provenance. Labels do not determine which connections are eligible.
    """
    if any(row.get("split") not in {"train", "val"} for row in bundle.metadata.values()):
        raise ValueError("test evidence must not enter FSN-TCN evaluation")
    expected = {key for key, row in bundle.metadata.items() if row["split"] == "val"}
    if not expected:
        raise ValueError("development validation must contain clips")
    for clip_id in expected:
        row = bundle.metadata[clip_id]
        if row.get("source_kind") not in {"clinical", "network"}:
            raise ValueError("source_kind must be clinical or network")
        if type(row.get("label_id")) is not int or row["label_id"] not in range(7):
            raise ValueError("sealed action label must be an integer in [0,6]")
        if any(not isinstance(row.get(key), str) or not row[key] for key in _GROUP_FIELDS):
            raise ValueError("candidate metadata needs group, record, video and timebase identity")
    sealed, original_pairs, all_original_pairs = _canonical(bundle)
    if not isinstance(chains, list) or not chains:
        raise ValueError("evaluation requires a nonempty chain list")
    seen_clips, seen_chains = set(), set()
    modes = set()
    prepared = []
    for chain in chains:
        chain_id, ordered = chain.get("chain_id"), chain.get("ordered_clip_ids")
        if not isinstance(chain_id, str) or not chain_id or chain_id in seen_chains:
            raise ValueError("evaluation chain_id must be unique and nonempty")
        if chain.get("split") != "val":
            raise ValueError("evaluation chains must declare split=val; test is not read")
        if not isinstance(ordered, list) or not ordered or any(
            not isinstance(clip_id, str) or clip_id not in expected for clip_id in ordered
        ):
            raise ValueError("development chains must contain known val clips only")
        if len(set(ordered)) != len(ordered) or seen_clips.intersection(ordered):
            raise ValueError("each val clip must appear exactly once")
        seen_clips.update(ordered)
        seen_chains.add(chain_id)
        eligible = _boolean_vector(chain.get("eligible"), len(ordered) - 1, "eligible")
        synthetic = chain.get("synthetic_chain", False)
        if type(synthetic) is not bool:
            raise ValueError("synthetic_chain must be a boolean")
        modes.add(synthetic)
        breaks = _boolean_vector(
            chain.get("synthetic_breaks", [False] * len(eligible)), len(eligible), "synthetic_breaks"
        )
        origin = chain.get("synthetic_origin")
        if synthetic:
            if origin not in _SYNTHETIC_ORIGINS:
                raise ValueError("synthetic chains require recognized construction provenance")
        else:
            if any(breaks) or origin is not None:
                raise ValueError("ordinary chains cannot declare synthetic breaks")
            if chain_id not in sealed or (ordered, eligible) != sealed[chain_id]:
                raise ValueError("ordinary chains must match sealed val candidates")
        sources = []
        for i, (left_id, right_id) in enumerate(zip(ordered, ordered[1:])):
            left, right = bundle.metadata[left_id], bundle.metadata[right_id]
            for row in (left, right):
                if row.get("source_kind") not in {"clinical", "network"}:
                    raise ValueError("source_kind must be clinical or network")
                if any(not isinstance(row.get(key), str) or not row[key] for key in _GROUP_FIELDS):
                    raise ValueError("candidate metadata needs group, record, video and timebase identity")
            pair = _pair(left_id, right_id)
            sources.append(left["source_kind"] if left["source_kind"] == right["source_kind"] else "mixed")
            if breaks[i]:
                if not synthetic or not eligible[i] or pair in original_pairs:
                    raise ValueError("synthetic breaks must be eligible non-original directed pairs")
                if left["source_kind"] != "clinical" or right["source_kind"] != "clinical":
                    raise ValueError("minimal synthetic breaks must have clinical endpoints")
                if origin == "fsn_tcn_clinical_shuffle_v1":
                    if any(left[key] != right[key] for key in _GROUP_FIELDS):
                        raise ValueError("clinical shuffle breaks must remain within one recording")
                else:
                    if any(left[key] == right[key] for key in ("group_id", "recording_key", "source_video_key")):
                        raise ValueError("cross-record breaks require distinct group, recording and video")
                    if not left.get("source_collection") or left.get("source_collection") != right.get("source_collection"):
                        raise ValueError("cross-record breaks must match clinical source collection")
            else:
                if synthetic:
                    if eligible[i] and pair not in original_pairs:
                        raise ValueError("unmarked synthetic-chain candidates must be original legal pairs")
                    if not eligible[i] and (pair not in all_original_pairs or all_original_pairs[pair]):
                        raise ValueError("unmarked ineligible adjacency must be an original structural cut")
                if any(left[key] != right[key] for key in _GROUP_FIELDS) or sources[-1] == "mixed":
                    raise ValueError("ordinary adjacencies cannot cross group, record, video, timebase or source")
            if eligible[i] and not breaks[i] and pair not in original_pairs:
                raise ValueError("ordinary eligible adjacency differs from sealed candidate pairs")
        prepared.append((chain, ordered, eligible, breaks, sources))
    if seen_clips != expected:
        raise ValueError("evaluation chains must cover all val clips exactly once")
    if len(modes) != 1:
        raise ValueError("formal and synthetic conditions must be evaluated separately")
    return prepared


def _paired(labels, prediction, visual):
    changed = prediction != visual
    improved = int(((visual != labels) & (prediction == labels)).sum())
    worsened = int(((visual == labels) & (prediction != labels)).sum())
    return {
        "changed_count": int(changed.sum()),
        "changed_fraction": float(changed.mean()) if len(labels) else 0.,
        "improved": improved,
        "worsened": worsened,
        "wrong_to_different_wrong": int(((visual != labels) & (prediction != labels) & changed).sum()),
        "net_correct_gain": improved - worsened,
    }


def _directions(labels, prediction):
    result = {}
    for left, right, name in ((3, 4, "sweep_to_reperfusion"), (4, 3, "reperfusion_to_sweep")):
        support = int((labels == left).sum())
        count = int(((labels == left) & (prediction == right)).sum())
        result[name] = {"count": count, "true_class_support": support, "rate": count / support if support else 0.}
    return result


def _metrics(labels, prediction, visual):
    return {
        **classification_metrics(labels, prediction),
        "paired_vs_A": _paired(labels, prediction, visual),
        "sweep_reperfusion": _directions(labels, prediction),
    }


def _edge_count(flags):
    return {"count": len(flags), "open": sum(flags), "open_fraction": sum(flags) / len(flags) if flags else 0.}


@torch.no_grad()
def evaluate_chains(bundle, chains, base, relation_scores, *, strategy, threshold, seed, device="cpu"):
    """Return ``(safe_aggregate, private_prediction_rows)`` for all val clips.

    Relation scores map explicit directed clip-id pairs to uncalibrated [0,1]
    scores. No label or source identifier is supplied to the TCN. The ``random``
    strategy matches learned-open counts per source within each candidate chain.
    """
    prepared = validate_evaluation_chains(bundle, chains)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0,1]")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    relation_scores = relation_scores or {}
    if not isinstance(relation_scores, dict):
        raise ValueError("relation_scores must map directed pairs to scores")
    wrapper = SegmentedTCN(base).to(device).eval()
    labels_all, visual_all, prediction_all, source_all, duration_all = [], [], [], [], []
    predrows, edge_flags = [], {"original_clinical": [], "original_network": [], "synthetic_breaks": []}
    counts = Counter(chains=0, original_singleton_chains=0, isolated_clips_after_gating=0,
                     structural_cuts=0, candidate_edges=0)
    random_audit = Counter(different_edges=0, eligible_edges=0,
                           chains_with_changed_positions=0, eligible_chains=0)
    cross_entropy_sum = 0.
    for chain, ordered, eligible, breaks, sources in prepared:
        scores = []
        for left, right, flag in zip(ordered, ordered[1:], eligible):
            value = relation_scores.get((left, right))
            if strategy in {"learned", "random"} and flag and value is None:
                raise ValueError("every eligible learned/random edge requires a relation score")
            value = 0. if value is None else value
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("relation scores must be finite scalars in [0,1]")
            scores.append(float(value))
        opened = select_gates(strategy, eligible=eligible, source_kinds=sources, scores=scores,
                              threshold=threshold, seed=seed, chain_id=chain["chain_id"])
        opened = _boolean_vector(list(opened), len(eligible), "selected gates")
        if any(active and not flag for active, flag in zip(opened, eligible)):
            raise ValueError("a gate cannot enable a structural cut")
        if strategy == "random":
            learned = select_gates("learned", eligible=eligible, source_kinds=sources, scores=scores,
                                   threshold=threshold, seed=seed, chain_id=chain["chain_id"])
            changed_positions = sum(flag and selected != reference
                                    for flag, selected, reference in zip(eligible, opened, learned))
            random_audit["different_edges"] += changed_positions
            random_audit["eligible_edges"] += sum(eligible)
            random_audit["chains_with_changed_positions"] += int(changed_positions > 0)
            random_audit["eligible_chains"] += int(any(eligible))
        features, a_logits, labels = load_chain(bundle, chain, device)
        output = wrapper(features, a_logits, opened)
        if output.shape != (len(ordered), 7) or not bool(torch.isfinite(output).all()):
            raise ValueError("TCN must output finite logits with shape [clips,7]")
        if a_logits.shape != (len(ordered), 7) or not bool(torch.isfinite(a_logits).all()):
            raise ValueError("frozen A must provide finite seven-class logits")
        label_tensor = torch.as_tensor(labels, dtype=torch.long, device=output.device)
        chain_loss = float(torch.nn.functional.cross_entropy(output.float(), label_tensor, reduction="sum").item())
        if not math.isfinite(chain_loss):
            raise ValueError("TCN validation cross entropy must remain finite")
        cross_entropy_sum += chain_loss
        labels = label_tensor.cpu().numpy()
        if labels.shape != (len(ordered),) or any(int(labels[i]) != bundle.metadata[clip_id]["label_id"] for i, clip_id in enumerate(ordered)):
            raise ValueError("feature labels must match sealed metadata")
        prediction, visual = output.argmax(-1).cpu().numpy(), a_logits.argmax(-1).cpu().numpy()
        for i, clip_id in enumerate(ordered):
            meta = bundle.metadata[clip_id]
            duration = meta.get("duration", meta.get("clip_duration_sec"))
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
                raise ValueError("duration must be finite and positive")
            isolated = (i == 0 or not opened[i - 1]) and (i == len(ordered) - 1 or not opened[i])
            counts["isolated_clips_after_gating"] += int(isolated)
            if isolated and not torch.equal(output[i], a_logits[i]):
                raise ValueError("isolated clips must return A logits exactly")
            predrows.append({"clip_id": clip_id, "chain_id": chain["chain_id"], "split": "val",
                             "label_id": int(labels[i]), "A_prediction": int(visual[i]),
                             "prediction": int(prediction[i]), "source_kind": meta["source_kind"],
                             "duration": float(duration), "isolated_after_gating": isolated,
                             "left_open": bool(opened[i - 1]) if i else False,
                             "right_open": bool(opened[i]) if i < len(opened) else False})
            source_all.append(meta["source_kind"])
            duration_all.append(float(duration))
        labels_all.extend(labels.tolist()); visual_all.extend(visual.tolist()); prediction_all.extend(prediction.tolist())
        counts["chains"] += 1
        counts["original_singleton_chains"] += int(len(ordered) == 1)
        for active, flag, synthetic, source in zip(opened, eligible, breaks, sources):
            counts["structural_cuts"] += int(not flag)
            if flag:
                counts["candidate_edges"] += 1
                key = "synthetic_breaks" if synthetic else f"original_{source}"
                edge_flags[key].append(active)
    labels, visual, prediction = map(lambda x: np.asarray(x, dtype=int), (labels_all, visual_all, prediction_all))
    sources, durations = np.asarray(source_all), np.asarray(duration_all)
    slices = {"clinical": sources == "clinical", "network": sources == "network",
              "duration_le_1s": durations <= 1,
              "duration_gt_1_le_5s": (durations > 1) & (durations <= 5),
              "duration_gt_5s": durations > 5}
    synthetic = prepared[0][0].get("synthetic_chain", False)
    result = {
        "schema_version": "fsn_tcn_evaluation_v1",
        "evaluation_role": "development_validation_not_independent_test",
        "condition": "synthetic_disruptions" if synthetic else "sealed_original_chains",
        "synthetic_evidence_scope": "constructed breaks; not independently verified real editing boundaries",
        "strategy": strategy, "threshold": float(threshold), "threshold_is_calibrated": False,
        "seed": seed, "A_frozen": True, "all_val_clips_evaluated_once": True,
        "counts": {"clips": len(labels), **dict(counts)},
        "cross_entropy": cross_entropy_sum / len(labels),
        "cross_entropy_definition": "unweighted seven-class CE averaged over all val clips",
        "metrics": _metrics(labels, prediction, visual),
        "A_metrics": classification_metrics(labels, visual),
        "slices": {name: _metrics(labels[mask], prediction[mask], visual[mask]) for name, mask in slices.items()},
        "gate_audit": {name: _edge_count(flags) for name, flags in edge_flags.items()},
        "gate_audit_denominator": "candidate edges; classification denominator is clips",
        "original_clinical_interpretation": "retention of source-policy clinical candidates, not verified continuity recall",
        "random_control": "matches learned-open count per source within each chain" if strategy == "random" else None,
    }
    if strategy == "random":
        result["random_position_audit"] = dict(random_audit)
    return result, predrows
