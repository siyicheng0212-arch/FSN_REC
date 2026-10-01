"""Paired development evaluation with one frozen A and one fixed transition table.

Source-rule targets are weak policy labels, not independently verified relation
ground truth. Identifiers and predictions stay private; the public file contains
aggregate counts/metrics only. No action label enters R or decoding.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .data import EVIDENCE_KEYS, load_edges, load_feature_index, read_jsonl, sha256_file
from .decoder import decode
from .network import RelationConfig, RelationNet
from .pipeline import predict_chain
from .protocol import SOURCE_POLICY, validate_source_artifacts


def classification_metrics(labels, predictions):
    labels, predictions = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)
    if labels.shape != predictions.shape or labels.ndim != 1:
        raise ValueError("labels/predictions must be equally sized vectors")
    if np.any((labels < 0) | (labels > 6) | (predictions < 0) | (predictions > 6)):
        raise ValueError("seven-class labels must lie in [0,6]")
    confusion = np.zeros((7, 7), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    support, predicted = confusion.sum(1), confusion.sum(0)
    diagonal = confusion.diagonal()
    precision = np.divide(diagonal, predicted, out=np.zeros(7, dtype=float), where=predicted > 0)
    recall = np.divide(diagonal, support, out=np.zeros(7, dtype=float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(7),
                   where=precision + recall > 0)
    return {"count": len(labels), "accuracy": float(diagonal.sum() / len(labels)) if len(labels) else 0.,
            "macro_f1": float(f1.mean()), "confusion": confusion.tolist(),
            "per_class": [{"label_id": i, "precision": float(precision[i]),
                           "recall": float(recall[i]), "f1": float(f1[i]),
                           "support": int(support[i])} for i in range(7)]}


def _paired(labels, predictions, visual):
    labels, predictions, visual = map(np.asarray, (labels, predictions, visual))
    changed = predictions != visual
    return {"changed_count": int(changed.sum()),
            "changed_fraction": float(changed.mean()) if len(labels) else 0.,
            "improved": int(((visual != labels) & (predictions == labels)).sum()),
            "worsened": int(((visual == labels) & (predictions != labels)).sum()),
            "wrong_to_different_wrong": int(((visual != labels) & (predictions != labels) & changed).sum())}


def _directions(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    result = {}
    for left, right, name in ((3, 4, "sweep_to_reperfusion"), (4, 3, "reperfusion_to_sweep")):
        support, count = int((labels == left).sum()), int(((labels == left) & (predictions == right)).sum())
        result[name] = {"count": count, "true_class_support": support,
                        "rate": count / support if support else 0.}
    return result


def _metadata(index, rows):
    metadata = {}
    for row in rows:
        clip_id = row.get("clip_id")
        if clip_id not in index.clips or clip_id in metadata:
            raise ValueError("metadata must contain every indexed clip exactly once")
        indexed = index.clips[clip_id]
        if any(row.get(key) != indexed.get(key) for key in ("split", "group_id", "label_id")):
            raise ValueError("metadata split/group/action label differs from evidence index")
        label, duration = row.get("label_id"), row.get("duration")
        if not isinstance(label, int) or isinstance(label, bool) or not 0 <= label < 7:
            raise ValueError("metadata label_id must be an integer in [0,6]")
        if row.get("source_kind") not in ("clinical", "network"):
            raise ValueError("metadata source_kind must be clinical or network")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            raise ValueError("metadata duration must be finite and nonnegative")
        for key in ("source_kind", "duration"):
            if key in indexed and indexed[key] != row[key]:
                raise ValueError(f"metadata {key} differs from evidence index")
        metadata[clip_id] = row
    if metadata.keys() != index.clips.keys():
        raise ValueError("metadata is incomplete")
    return metadata


@torch.no_grad()
def _edge_probability(relation, index, left, right):
    relation.eval()
    device = next(relation.parameters()).device
    sides = [{key: torch.as_tensor(row[key], dtype=torch.float32, device=device).unsqueeze(0)
              for key in EVIDENCE_KEYS} for row in (index.read(left), index.read(right))]
    logit = relation(*sides)
    if logit.shape != (1,) or not bool(torch.isfinite(logit).all()):
        raise ValueError("invalid R edge prediction")
    return float(logit.sigmoid().item())


def evaluate_suite(index, metadata, chains, edges, transition, relations=None,
                   threshold=.9, strength=1., potential_bound=None):
    """All variants share A logits; C/D labels are used only after prediction.

    ``relations`` maps learned_mlp/learned_dual to trained modules. Every val clip
    must appear once, including singleton/structurally disconnected clips.
    """
    metadata = _metadata(index, metadata)
    relations = relations or {}
    if not set(relations) <= {"learned_mlp", "learned_dual"}:
        raise ValueError("unsupported learned variant name")
    names = ["A_only", "all_candidate", "source_rule", *relations]
    labels, visual, ids, sources, durations = [], [], [], [], []
    predictions = {name: [] for name in names}
    gates = {name: [] for name in names}
    gate_sources, candidate_pairs, seen_chains, seen_clips = [], {}, set(), set()
    options = {"threshold": threshold, "strength": strength, "potential_bound": potential_bound}
    for chain in chains:
        chain_id, ordered = chain.get("chain_id"), chain.get("ordered_clip_ids")
        if not isinstance(chain_id, str) or not chain_id or chain_id in seen_chains:
            raise ValueError("chain_id must be unique and nonempty")
        seen_chains.add(chain_id)
        if not isinstance(ordered, list) or not ordered or any(not isinstance(x, str) or x not in metadata for x in ordered):
            raise ValueError("chain needs known explicit ordered_clip_ids")
        if len(set(ordered)) != len(ordered) or seen_clips.intersection(ordered):
            raise ValueError("a val clip must not be evaluated more than once")
        if any(metadata[x]["split"] != "val" for x in ordered):
            raise ValueError("development chains must contain val clips only")
        seen_clips.update(ordered)
        eligible = chain.get("eligible", [False] * (len(ordered) - 1))
        if not isinstance(eligible, list) or len(eligible) != len(ordered) - 1 or any(type(x) is not bool for x in eligible):
            raise ValueError("eligible must explicitly contain adjacency booleans")
        pair_source = []
        for i, flag in enumerate(eligible):
            left, right = metadata[ordered[i]], metadata[ordered[i + 1]]
            if flag and any(left[key] != right[key] for key in ("group_id", "source_kind")):
                raise ValueError("eligible neighbors must share recording group and source kind")
            for key in ("source_video_key", "timebase_key", "recording_key"):
                if flag and (key in left or key in right) and left.get(key) != right.get(key):
                    raise ValueError(f"eligible neighbors must share {key}")
            pair_source.append(left["source_kind"] if left["source_kind"] == right["source_kind"] else "mixed")
            if flag:
                candidate_pairs[(ordered[i], ordered[i + 1])] = "C" if left["source_kind"] == "clinical" else "D"
        features = [index.read(x) for x in ordered]
        logits = np.stack([row["logits"] for row in features])
        if logits.shape != (len(ordered), 7) or not np.isfinite(logits).all():
            raise ValueError("A evidence must contain finite seven-class logits")
        results = {"A_only": decode(logits, transition, np.zeros(len(eligible)), eligible=eligible, **options),
                   "all_candidate": decode(logits, transition, np.ones(len(eligible)), eligible=eligible, **options),
                   "source_rule": decode(logits, transition, [float(x == "clinical") for x in pair_source],
                                         eligible=eligible, **options)}
        for name, relation in relations.items():
            results[name] = predict_chain(relation, features, transition, eligible=eligible, **options)
        ids.extend(ordered)
        labels.extend(metadata[x]["label_id"] for x in ordered)
        visual.extend(logits.argmax(1).tolist())
        sources.extend(metadata[x]["source_kind"] for x in ordered)
        durations.extend(metadata[x]["duration"] for x in ordered)
        gate_sources.extend(source for source, flag in zip(pair_source, eligible) if flag)
        for name, result in results.items():
            predictions[name].extend(result["predictions"])
            gates[name].extend(value > 0 for value, flag in zip(result["effective_reliability"], eligible) if flag)
    expected = {key for key, row in metadata.items() if row["split"] == "val"}
    if not expected or seen_clips != expected:
        raise ValueError("chains must cover all val clips exactly once")
    edge_pairs = set()
    for edge in edges:
        if edge["split"] != "val":
            continue
        pair = (edge["left_clip_id"], edge["right_clip_id"])
        if (pair in edge_pairs or candidate_pairs.get(pair) != edge["status"] or
                edge.get("label_origin", SOURCE_POLICY) != SOURCE_POLICY):
            raise ValueError("val source-policy edges differ from structural candidate chains")
        edge_pairs.add(pair)
    if edge_pairs != candidate_pairs.keys():
        raise ValueError("val source-policy edges must cover eligible candidates exactly")
    labels, visual, sources, durations = map(np.asarray, (labels, visual, sources, durations))
    summary = {"evaluation_role": "development_validation_not_independent_test",
               "label_policy": "source_rule_v1",
               "relation_target_interpretation": "source policy weak labels, not verified clinical continuity",
               "A_frozen_same_logits_all_variants": True, "threshold": threshold,
               "threshold_is_calibrated": False, "strength": strength, "potential_bound": potential_bound,
               "transition_control": "one train-only C table; uniform initial prior; not old full-v3 decoder",
               "variants": {}}
    for name in names:
        pred = np.asarray(predictions[name])
        metric = classification_metrics(labels, pred)
        metric.update(paired_vs_A=_paired(labels, pred, visual), sweep_reperfusion=_directions(labels, pred))
        slices = {"clinical": sources == "clinical", "network": sources == "network",
                  "duration_lt_1s": durations < 1, "duration_1_to_5s": (durations >= 1) & (durations < 5),
                  "duration_ge_5s": durations >= 5}
        metric["slices"] = {key: {**classification_metrics(labels[mask], pred[mask]),
                                  "paired_vs_A": _paired(labels[mask], pred[mask], visual[mask]),
                                  "sweep_reperfusion": _directions(labels[mask], pred[mask])}
                            for key, mask in slices.items()}
        network_mask = np.asarray(gate_sources) == "network"
        active = np.asarray(gates[name], dtype=bool)[network_mask]
        metric["network_candidate_gate"] = {"count": len(active), "active": int(active.sum()),
                                             "activation_fraction": float(active.mean()) if len(active) else 0.}
        if name in relations:
            metric["R_trainable_parameters"] = sum(x.numel() for x in relations[name].parameters() if x.requires_grad)
            tp = fp = fn = tn = 0
            for edge in edges:
                if edge["split"] != "val" or edge["status"] == "U":
                    continue
                positive = edge["status"] == "C"
                predicted = _edge_probability(relations[name], index, edge["left_clip_id"], edge["right_clip_id"]) >= threshold
                tp += int(positive and predicted); fp += int(not positive and predicted)
                fn += int(positive and not predicted); tn += int(not positive and not predicted)
            metric["source_policy_edge_metrics"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                                     "precision": tp / (tp + fp) if tp + fp else 0.,
                                                     "recall": tp / (tp + fn) if tp + fn else 0.,
                                                     "ground_truth_is_verified_continuity": False}
        summary["variants"][name] = metric
    private = [{"clip_id": clip_id, "label_id": int(labels[i]),
                "predictions": {name: int(predictions[name][i]) for name in names}}
               for i, clip_id in enumerate(ids)]
    return summary, private


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "protocol-dir", "transition", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--mlp-checkpoint")
    parser.add_argument("--dual-checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=.9)
    parser.add_argument("--strength", type=float, default=1.)
    parser.add_argument("--potential-bound", type=float)
    args = parser.parse_args(argv)
    output, protocol_dir, transition_dir = map(Path, (args.output, args.protocol_dir, args.transition))
    if output.exists():
        raise FileExistsError("evaluation output must be a fresh directory")
    index = load_feature_index(args.index)
    files = {"edges": protocol_dir / "edges.jsonl", "metadata": protocol_dir / "manifest_metadata.jsonl",
             "chains": protocol_dir / "chains_val.jsonl"}
    validate_source_artifacts(protocol_dir, index, files["edges"])
    hashes = {"index_sha256": sha256_file(args.index), "edges_sha256": sha256_file(files["edges"]),
              "checkpoint_sha256": index.checkpoint_sha256, "label_policy": SOURCE_POLICY,
              "source_audit_sha256": sha256_file(protocol_dir / "audit.json")}
    audit = json.loads((transition_dir / "audit.json").read_text())
    if any(audit.get(key) != value for key, value in hashes.items()):
        raise ValueError("transition does not share the exact feature/edge/A protocol")
    with np.load(transition_dir / "transition.npz", allow_pickle=False) as stored:
        transition = stored["transition"].copy()
    relations, checkpoint_hashes = {}, {}
    for mode in ("mlp", "dual"):
        checkpoint_path = getattr(args, f"{mode}_checkpoint")
        if checkpoint_path:
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if any(saved["protocol"].get(key) != value for key, value in hashes.items()):
                raise ValueError("R does not share the exact feature/edge/A protocol")
            if saved["config"]["mode"] != mode:
                raise ValueError("R checkpoint mode does not match argument")
            relation = RelationNet(RelationConfig(**saved["config"])).to(args.device)
            relation.load_state_dict(saved["model"], strict=True)
            relations[f"learned_{mode}"] = relation
            checkpoint_hashes[mode] = sha256_file(checkpoint_path)
    summary, private = evaluate_suite(index, read_jsonl(files["metadata"]), read_jsonl(files["chains"]),
                                       load_edges(files["edges"], index), transition, relations,
                                       args.threshold, args.strength, args.potential_bound)
    output.mkdir(parents=True, exist_ok=False)
    (output / "public_safe_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    with (output / "private_predictions.jsonl").open("x") as handle:
        for row in private:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "private_protocol.json").write_text(json.dumps({**hashes, "relation_checkpoint_sha256": checkpoint_hashes,
        "metadata_sha256": sha256_file(files["metadata"]), "chains_sha256": sha256_file(files["chains"]),
        "transition_sha256": sha256_file(transition_dir / "transition.npz"),
        "arguments": vars(args), "contains_private_paths": True}, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    main()
