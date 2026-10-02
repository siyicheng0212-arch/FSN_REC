"""Audit legacy TCN errors before considering a new trainable module.

Inputs are sealed full train7372/val823 evidence, an independently preserved
historical val prediction file and, optionally, the *real* legacy checkpoint.
Nothing in this module trains, edits source artifacts, or reads a test split.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.context_utility.config import build_base, load_config
from experiments.fsn_tcn.io import load_bundle, load_chain
from experiments.relation.data import read_jsonl, sha256_file
from .metrics import (
    assert_known_history as _historical_assertion,
    classification_metrics, confusions as _confusions,
    paired as _paired, summary as _summary,
)


REQUIRED_COLUMNS = ("clip_id", "label_id", "A_prediction", "prediction")


def _prediction_class(value, name):
    if type(value) is not int or not 0 <= value < 7:
        raise ValueError(f"{name} must be an integer class in [0, 6]")
    return value


def _columns(path):
    if path is None:
        return {key: key for key in REQUIRED_COLUMNS}
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(mapping, dict) or set(mapping) != set(REQUIRED_COLUMNS)
            or any(not isinstance(value, str) or not value for value in mapping.values())
            or len(set(mapping.values())) != len(mapping)):
        raise ValueError("columns JSON must map the four required columns to distinct nonempty names")
    return mapping


def _chain_positions(bundle):
    """Map each val clip to its adjacent *eligible* edge statuses."""
    edges = {(edge["left_clip_id"], edge["right_clip_id"]): edge["status"]
             for edge in bundle.edges if edge["split"] == "val"}
    result = {}
    used = set()
    for chain in bundle.chains["val"]:
        ids, eligible = chain["ordered_clip_ids"], chain["eligible"]
        edge_status = []
        for left, right, valid in zip(ids, ids[1:], eligible):
            status = edges.get((left, right)) if valid else None
            if bool(status) != valid:
                raise ValueError("candidate chain and declared val edge differ")
            if valid:
                used.add((left, right))
            edge_status.append(status)
        for i, clip in enumerate(ids):
            adjacent = [edge_status[j] for j in (i - 1, i)
                        if 0 <= j < len(edge_status) and edge_status[j] is not None]
            status_set = set(adjacent)
            if not status_set:
                category = "isolated"
            else:
                category = "+".join(sorted(status_set))
            nearest_d = min((min(abs(i - j), abs(i - (j + 1)))
                             for j, status in enumerate(edge_status) if status == "D"),
                            default=None)
            result[clip] = {"source": bundle.metadata[clip]["source_kind"],
                            "adjacent_status": category, "distance_to_D": nearest_d,
                            "candidate_chain_length": len(ids)}
    if used != set(edges) or set(result) != {
        clip for clip, row in bundle.metadata.items() if row["split"] == "val"
    }:
        raise ValueError("val edge/clip coverage is incomplete")
    return result


def load_historical_predictions(bundle, prediction_path, *, column_map=None,
                                logits_key=None):
    """Require exact val coverage and verify every historical A against frozen evidence."""
    names = _columns(column_map)
    expected = {clip for clip, row in bundle.metadata.items() if row["split"] == "val"}
    observed = {}
    for raw in read_jsonl(prediction_path):
        if not set(names.values()).issubset(raw):
            raise ValueError("historical prediction row lacks an explicitly mapped column")
        clip = raw[names["clip_id"]]
        if not isinstance(clip, str) or clip not in expected or clip in observed:
            raise ValueError("historical predictions contain unknown, non-val or duplicate clip")
        label = _prediction_class(raw[names["label_id"]], "historical label")
        old_a = _prediction_class(raw[names["A_prediction"]], "historical A prediction")
        old_tcn = _prediction_class(raw[names["prediction"]], "historical TCN prediction")
        if label != bundle.metadata[clip]["label_id"]:
            raise ValueError("historical action label differs from sealed val label")
        frozen_a = int(bundle.index.read(clip)["logits"].argmax())
        if frozen_a != old_a:
            raise ValueError("historical A disagrees with sealed frozen-A evidence")
        row = {"clip_id": clip, "label_id": label, "A_prediction": old_a,
               "prediction": old_tcn}
        if logits_key is not None:
            scores = raw.get(logits_key)
            if (not isinstance(scores, list) or len(scores) != 7
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           for value in scores)):
                raise ValueError("requested historical TCN logits must be seven finite numbers")
            row["tcn_logits"] = scores
        observed[clip] = row
    if set(observed) != expected or len(observed) != 823:
        raise ValueError("historical prediction file must cover exactly the same val823 clips")
    return observed


def audit_predictions(bundle, historical):
    positions = _chain_positions(bundle)
    ordered = sorted(historical)
    total = _summary([historical[clip] for clip in ordered])
    _historical_assertion(total)
    slices = {}
    for name, predicate in {
        "clinical": lambda row: row["source"] == "clinical",
        "network": lambda row: row["source"] == "network",
        "adjacent_C": lambda row: "C" in row["adjacent_status"],
        "adjacent_D": lambda row: "D" in row["adjacent_status"],
        "isolated": lambda row: row["adjacent_status"] == "isolated",
        "within_1_position_of_D": lambda row: row["distance_to_D"] is not None and row["distance_to_D"] <= 1,
        "within_2_positions_of_D": lambda row: row["distance_to_D"] is not None and row["distance_to_D"] <= 2,
    }.items():
        selected = [historical[clip] for clip in ordered if predicate(positions[clip])]
        if selected:
            slices[name] = _summary(selected)
    return {"total": total, "slices": slices,
            "annotation_interpretation": "C/D status is the sealed supplied supervision; proximity is associative, not causal"}


def _intervals(count, cuts):
    if len(cuts) != max(0, count - 1):
        raise ValueError("cut mask must have T-1 entries")
    start = 0
    for i, cut in enumerate(cuts):
        if cut:
            yield start, i + 1
            start = i + 1
    yield start, count


@torch.inference_mode()
def _predict(model, features, a_logits, cuts):
    outputs = []
    for start, end in _intervals(len(features), cuts):
        output = model(features[start:end], a_logits[start:end])
        if (not isinstance(output, torch.Tensor) or output.shape != (end - start, 7)
                or not torch.isfinite(output).all()):
            raise ValueError("actual legacy model must return finite [T,7] scores")
        outputs.append(output)
    return torch.cat(outputs, dim=0)


def _run_counterfactual(bundle, historical, model, *, chain_layout, device, logits_atol):
    """Run the unchanged old model on full and D-cut chains, including singleton calls."""
    statuses = {(edge["left_clip_id"], edge["right_clip_id"]): edge["status"]
                for edge in bundle.edges if edge["split"] == "val"}
    after, before = {}, {}
    cut_edges = 0
    cut_singletons = 0
    preexisting_structural_crossings = 0
    for chain in bundle.chains["val"]:
        ids, eligible = chain["ordered_clip_ids"], chain["eligible"]
        features, a_logits, _ = load_chain(bundle, chain, device=device)
        structural = [not flag for flag in eligible]
        if chain_layout == "full_chain":
            original_cuts = [False] * len(eligible)
            preexisting_structural_crossings += sum(structural)
        elif chain_layout == "eligible_segments":
            original_cuts = structural
        else:
            raise ValueError("declare original chain_layout explicitly")
        d_flags = [bool(eligible[i] and statuses[(ids[i], ids[i + 1])] == "D")
                   for i in range(len(eligible))]
        cut_edges += sum(d_flags)
        intervention_cuts = [old or d for old, d in zip(original_cuts, d_flags)]
        original_scores = _predict(model, features, a_logits, original_cuts)
        changed_scores = _predict(model, features, a_logits, intervention_cuts)
        cut_singletons += sum(end - start == 1 for start, end in _intervals(len(ids), intervention_cuts))
        for i, clip in enumerate(ids):
            if clip in before:
                raise ValueError("duplicate val clip in counterfactual evaluation")
            historical_row = historical[clip]
            old_score = original_scores[i].float().cpu().numpy()
            observed = int(old_score.argmax())
            if observed != historical_row["prediction"]:
                raise ValueError("legacy checkpoint fails exact class parity with historical val823 predictions")
            if "tcn_logits" in historical_row and not np.allclose(
                old_score, historical_row["tcn_logits"], atol=logits_atol, rtol=0
            ):
                raise ValueError("legacy checkpoint fails numerical logit parity with historical val823")
            before[clip] = observed
            after[clip] = int(changed_scores[i].argmax())
    if set(before) != set(historical) or not cut_edges:
        raise ValueError("legacy counterfactual lacks complete val coverage or D edges")
    ordered = sorted(historical)
    labels = np.array([historical[clip]["label_id"] for clip in ordered])
    a = np.array([historical[clip]["A_prediction"] for clip in ordered])
    initial = np.array([before[clip] for clip in ordered])
    cut = np.array([after[clip] for clip in ordered])
    was_rescue = (a != labels) & (initial == labels)
    was_spoil = (a == labels) & (initial != labels)
    answer = {"legacy_exact_prediction_parity_clips": len(before),
              "numeric_logit_parity_checked": all("tcn_logits" in row for row in historical.values()),
              "original_chain_layout": chain_layout,
              "structural_edges_crossed_by_historical_layout": preexisting_structural_crossings,
              "removed_D_edges": cut_edges,
              "resulting_singleton_model_calls": cut_singletons,
              "singleton_policy": "delegate to unchanged historical model (never substitute A)",
              "all_original_metrics": classification_metrics(labels, initial),
              "all_cut_metrics": classification_metrics(labels, cut),
              "cut_vs_original": _paired(labels, initial, cut),
              "cut_vs_A": _paired(labels, a, cut),
              "old_TCN_rescues_preserved": int((was_rescue & (cut == labels)).sum()),
              "old_TCN_rescues_lost": int((was_rescue & (cut != labels)).sum()),
              "old_TCN_spoils_repaired": int((was_spoil & (cut == labels)).sum()),
              "old_TCN_spoils_remaining": int((was_spoil & (cut != labels)).sum()),
              "old_TCN_3_to_4": _confusions(labels, initial)["3_to_4"],
              "cut_3_to_4": _confusions(labels, cut)["3_to_4"],
              "old_TCN_4_to_3": _confusions(labels, initial)["4_to_3"],
              "cut_4_to_3": _confusions(labels, cut)["4_to_3"]}
    for source in ("clinical", "network"):
        selected = [i for i, clip in enumerate(ordered)
                    if bundle.metadata[clip]["source_kind"] == source]
        if selected:
            answer[source] = {"count": len(selected),
                              "original": classification_metrics(labels[selected], initial[selected]),
                              "cut": classification_metrics(labels[selected], cut[selected]),
                              "cut_vs_original": _paired(labels[selected], initial[selected], cut[selected])}
    return answer


def run(args):
    if (args.legacy_config is None) != (args.checkpoint is None):
        raise ValueError("legacy config and checkpoint must both be given for a counterfactual")
    if args.checkpoint and args.chain_layout is None:
        raise ValueError("counterfactual requires the verified historical chain layout")
    if args.device != "cpu" and not args.checkpoint:
        raise ValueError("a device is needed only for the checkpoint stage")
    bundle = load_bundle(args.feature_index, args.source_protocol_dir)
    historical = load_historical_predictions(bundle, args.historical_predictions,
                                             column_map=args.column_map,
                                             logits_key=args.logits_key)
    report = {"schema": "fsn-legacy-tcn-readonly-audit-v1",
              "role": "val823_development_not_independent_test",
              "fingerprint": bundle.fingerprint,
              "historical_predictions_sha256": sha256_file(args.historical_predictions),
              "known_historical_counts_verified": True,
              "paired_diagnostic": audit_predictions(bundle, historical),
              "counterfactual": None}
    if args.checkpoint:
        config = load_config(args.legacy_config)
        if config["baseline"]["kind"] != "legacy":
            raise ValueError("counterfactual requires the actual audited historical model, not a reference TCN")
        model, model_audit = build_base(
            config, bundle.index.global_dim + bundle.index.local_dim,
            args.checkpoint, args.device, fingerprint=bundle.fingerprint
        )
        model.eval()
        report["counterfactual"] = _run_counterfactual(
            bundle, historical, model, chain_layout=args.chain_layout,
            device=args.device, logits_atol=args.logits_atol
        )
        report["legacy_model_audit"] = {key: value for key, value in model_audit.items()
                                        if key != "checkpoint" and key != "legacy_audit"}
        report["legacy_model_audit"]["legacy_audit_sha256"] = model_audit["legacy_audit"]["sha256"]
    if args.output:
        destination = Path(args.output).resolve()
        project = Path(__file__).resolve().parents[2]
        if destination.is_relative_to(project):
            raise ValueError("write private aggregate outside the code worktree")
        # Never overwrite a historical artifact or emit per-clip rows.
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-index", required=True)
    parser.add_argument("--source-protocol-dir", required=True)
    parser.add_argument("--historical-predictions", required=True)
    parser.add_argument("--column-map", help="Private JSON mapping four canonical names to old JSONL keys")
    parser.add_argument("--logits-key", help="Optional JSONL key with actual old TCN seven logits")
    parser.add_argument("--legacy-config", help="Verified context_utility legacy config (optional)")
    parser.add_argument("--checkpoint", help="Actual old TCN checkpoint (optional)")
    parser.add_argument("--chain-layout", choices=("full_chain", "eligible_segments"),
                        help="Verified historical preprocessing; mandatory with checkpoint")
    parser.add_argument("--device", default="cpu", help="CPU default; optional cuda:0 for inference only")
    parser.add_argument("--logits-atol", type=float, default=1e-5)
    parser.add_argument("--output", help="Optional new private aggregate JSON file; never overwritten")
    args = parser.parse_args()
    if not math.isfinite(args.logits_atol) or args.logits_atol < 0:
        parser.error("--logits-atol must be nonnegative and finite")
    result = run(args)
    concise = {"role": result["role"],
               "paired": result["paired_diagnostic"]["total"]["old_TCN_vs_A"],
               "counterfactual": (None if result["counterfactual"] is None else {
                   key: result["counterfactual"][key]
                   for key in ("removed_D_edges", "cut_vs_original", "old_TCN_rescues_preserved",
                               "old_TCN_rescues_lost", "old_TCN_spoils_repaired")}),
               "output": args.output}
    print(json.dumps(concise, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
