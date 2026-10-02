"""Known-arithmetic and provenance tests for paired FSN-TCN evaluation."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.fsn_tcn.evaluate import evaluate_chains, validate_evaluation_chains


class FixedFeatureClassifier(nn.Module):
    """Use a synthetic visual feature as class evidence for metric tests."""
    def forward(self, features, a_logits):
        return torch.nn.functional.one_hot(features[:, 0].long(), num_classes=7).float() * 10


def _bundle(records):
    metadata, arrays, chains = {}, {}, []
    for group, kind, labels, visual, revised, durations in records:
        ids = []
        for i, (label, a_pred, pred, duration) in enumerate(zip(labels, visual, revised, durations)):
            clip = f"private-{group}-{i}"
            ids.append(clip)
            metadata[clip] = {"clip_id": clip, "split": "val", "group_id": group,
                              "recording_key": f"record-{group}", "source_video_key": f"video-{group}",
                              "timebase_key": "shared-timebase", "source_kind": kind,
                              "source_collection": "clinical_collection" if kind == "clinical" else "network_collection",
                              "label_id": label, "duration": duration}
            arrays[clip] = {"global_tokens": np.asarray([[pred]], dtype=np.float32),
                            "local_tokens": np.zeros((1, 1), dtype=np.float32),
                            "logits": np.eye(7, dtype=np.float32)[a_pred] * 10}
        chains.append({"chain_id": f"private-chain-{group}", "split": "val", "source_kind": kind,
                       "ordered_clip_ids": ids, "eligible": [True] * (len(ids) - 1)})
    index = SimpleNamespace(read=lambda clip: arrays[clip], clips=metadata)
    return SimpleNamespace(index=index, metadata=metadata, chains={"val": chains, "train": []}, edges=[], fingerprint={})


@pytest.fixture
def arithmetic_bundle():
    return _bundle([
        ("clinical", "clinical", [3, 4, 0], [4, 3, 1], [3, 4, 2], [.5, 1., 2.]),
        ("network", "network", [3, 4], [3, 4], [4, 0], [5., 6.]),
    ])


def _scores(chains, value=.99):
    return {(left, right): value for chain in chains
            for left, right in zip(chain["ordered_clip_ids"], chain["ordered_clip_ids"][1:])}


def test_known_metrics_paired_changes_directions_and_duration_boundaries(arithmetic_bundle):
    result, private = evaluate_chains(arithmetic_bundle, arithmetic_bundle.chains["val"], FixedFeatureClassifier(), {},
                                     strategy="all", threshold=.9, seed=42)
    assert result["metrics"]["count"] == 5
    assert result["metrics"]["accuracy"] == pytest.approx(2 / 5)
    assert result["metrics"]["paired_vs_A"] == {
        "changed_count": 5, "changed_fraction": 1., "improved": 2, "worsened": 2,
        "wrong_to_different_wrong": 1, "net_correct_gain": 0,
    }
    confusion = result["metrics"]["confusion"]
    assert confusion[3][3] == 1 and confusion[3][4] == 1
    assert confusion[4][4] == 1 and confusion[4][0] == 1
    assert result["metrics"]["per_class"][3]["precision"] == 1.
    assert result["metrics"]["per_class"][3]["recall"] == .5
    assert result["metrics"]["per_class"][3]["f1"] == pytest.approx(2 / 3)
    assert result["metrics"]["macro_f1"] == pytest.approx((2 / 3 + .5) / 7)
    assert result["metrics"]["sweep_reperfusion"]["sweep_to_reperfusion"] == {
        "count": 1, "true_class_support": 2, "rate": .5,
    }
    assert result["metrics"]["sweep_reperfusion"]["reperfusion_to_sweep"]["count"] == 0
    assert result["slices"]["duration_le_1s"]["count"] == 2
    assert result["slices"]["duration_gt_1_le_5s"]["count"] == 2
    assert result["slices"]["duration_gt_5s"]["count"] == 1
    assert result["gate_audit"]["original_clinical"]["count"] == 2
    assert result["gate_audit"]["original_network"]["count"] == 1
    assert result["counts"]["candidate_edges"] == 3  # Edges differ from five clips.
    public_json = json.dumps(result)
    assert all(row["clip_id"] not in public_json and row["chain_id"] not in public_json for row in private)


def test_source_rule_preserves_network_predictions_exactly(arithmetic_bundle):
    result, rows = evaluate_chains(arithmetic_bundle, arithmetic_bundle.chains["val"], FixedFeatureClassifier(), {},
                                  strategy="source_rule", threshold=.9, seed=42)
    assert result["metrics"]["accuracy"] == .8
    assert result["slices"]["network"]["paired_vs_A"]["changed_count"] == 0
    assert result["gate_audit"]["original_network"] == {"count": 1, "open": 0, "open_fraction": 0.}
    assert result["counts"]["isolated_clips_after_gating"] == 2
    assert all(row["prediction"] == row["A_prediction"] for row in rows if row["source_kind"] == "network")


def test_all_closed_returns_a_and_random_matches_learned_edge_counts(arithmetic_bundle):
    chains = arithmetic_bundle.chains["val"]
    closed, _ = evaluate_chains(arithmetic_bundle, chains, FixedFeatureClassifier(), _scores(chains, .1),
                                strategy="learned", threshold=.9, seed=42)
    assert closed["metrics"]["paired_vs_A"]["changed_count"] == 0
    assert closed["metrics"]["accuracy"] == closed["A_metrics"]["accuracy"]
    assert closed["counts"]["isolated_clips_after_gating"] == 5
    scores = _scores(chains, .1)
    first = chains[0]["ordered_clip_ids"]
    scores[first[0], first[1]] = .99
    learned, _ = evaluate_chains(arithmetic_bundle, chains, FixedFeatureClassifier(), scores,
                                 strategy="learned", threshold=.9, seed=42)
    random, _ = evaluate_chains(arithmetic_bundle, chains, FixedFeatureClassifier(), scores,
                                strategy="random", threshold=.9, seed=42)
    assert learned["gate_audit"] == random["gate_audit"]
    assert learned["gate_audit"]["original_clinical"] == {"count": 2, "open": 1, "open_fraction": .5}


def test_only_singletons_need_no_relation_scores_and_have_zero_edge_denominators():
    bundle = _bundle([("a", "clinical", [3], [3], [4], [2.])])
    result, rows = evaluate_chains(bundle, bundle.chains["val"], FixedFeatureClassifier(), {},
                                  strategy="learned", threshold=.9, seed=42)
    assert result["metrics"]["accuracy"] == 1.
    assert result["counts"]["original_singleton_chains"] == 1
    assert result["counts"]["isolated_clips_after_gating"] == 1
    assert result["counts"]["candidate_edges"] == 0
    assert result["gate_audit"]["original_clinical"] == {"count": 0, "open": 0, "open_fraction": 0.}
    assert result["slices"]["network"]["count"] == 0
    assert result["slices"]["network"]["macro_f1"] == 0.
    assert rows[0]["prediction"] == rows[0]["A_prediction"]


def test_random_position_audit_counts_real_position_changes_and_degenerate_chains():
    bundle = _bundle([
        ("partial", "clinical", [0] * 6, [0] * 6, [0] * 6, [2.] * 6),
        ("on", "clinical", [0] * 4, [0] * 4, [0] * 4, [2.] * 4),
        ("off", "network", [0] * 2, [0] * 2, [0] * 2, [2.] * 2),
        ("singleton", "clinical", [0], [0], [0], [2.]),
    ])
    chains, scores = bundle.chains["val"], _scores(bundle.chains["val"], .1)
    ids = chains[0]["ordered_clip_ids"]
    scores[ids[0], ids[1]] = scores[ids[1], ids[2]] = .99
    for left, right in zip(chains[1]["ordered_clip_ids"], chains[1]["ordered_clip_ids"][1:]):
        scores[left, right] = .99
    learned, _ = evaluate_chains(bundle, chains, FixedFeatureClassifier(), scores,
                                 strategy="learned", threshold=.9, seed=42)
    random, rows = evaluate_chains(bundle, chains, FixedFeatureClassifier(), scores,
                                   strategy="random", threshold=.9, seed=42)
    repeated, repeated_rows = evaluate_chains(bundle, chains, FixedFeatureClassifier(), scores,
                                             strategy="random", threshold=.9, seed=42)
    # With this fixed seed the five partial-chain flags move from [1,1,0,0,0]
    # to [1,0,0,0,1]; fully open, fully closed and singleton chains cannot move.
    assert random["random_position_audit"] == {
        "different_edges": 2, "eligible_edges": 9,
        "chains_with_changed_positions": 1, "eligible_chains": 3,
    }
    assert random["random_position_audit"] == repeated["random_position_audit"]
    assert rows == repeated_rows
    assert random["gate_audit"] == learned["gate_audit"]
    public_json = json.dumps(random["random_position_audit"])
    assert not any(row["clip_id"] in public_json or row["chain_id"] in public_json for row in rows)


@pytest.mark.parametrize("score", [.1, .99])
def test_random_position_audit_exposes_all_off_or_all_on_degeneracy(arithmetic_bundle, score):
    result, _ = evaluate_chains(arithmetic_bundle, arithmetic_bundle.chains["val"], FixedFeatureClassifier(),
                                _scores(arithmetic_bundle.chains["val"], score),
                                strategy="random", threshold=.9, seed=42)
    assert result["random_position_audit"] == {
        "different_edges": 0, "eligible_edges": 3,
        "chains_with_changed_positions": 0, "eligible_chains": 2,
    }


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "exactly once"), ("missing", "cover all"), ("train", "val clips only"),
    ("test", "split=val"), ("cross_record", "cannot cross"), ("reordered", "sealed val"),
])
def test_rejects_leakage_duplicate_partial_and_unsealed_formal_chains(arithmetic_bundle, mutation, match):
    bundle, chains = deepcopy(arithmetic_bundle), deepcopy(arithmetic_bundle.chains["val"])
    if mutation == "duplicate":
        chains.append({**chains[0], "chain_id": "extra-chain"})
    elif mutation == "missing":
        chains.pop()
    elif mutation == "train":
        first = chains[0]["ordered_clip_ids"][0]
        bundle.metadata[first]["split"] = "train"
    elif mutation == "test":
        chains[0]["split"] = "test"
    elif mutation == "cross_record":
        second = chains[0]["ordered_clip_ids"][1]
        bundle.metadata[second]["recording_key"] = "other-recording"
    elif mutation == "reordered":
        chains[0]["ordered_clip_ids"].reverse()
    with pytest.raises(ValueError, match=match):
        validate_evaluation_chains(bundle, chains)


def test_refuses_missing_or_invalid_relation_scores_before_prediction(arithmetic_bundle):
    chains = arithmetic_bundle.chains["val"]
    with pytest.raises(ValueError, match="requires a relation score"):
        evaluate_chains(arithmetic_bundle, chains, FixedFeatureClassifier(), {},
                        strategy="learned", threshold=.9, seed=42)
    scores = _scores(chains, float("nan"))
    with pytest.raises(ValueError, match="finite scalars"):
        evaluate_chains(arithmetic_bundle, chains, FixedFeatureClassifier(), scores,
                        strategy="learned", threshold=.9, seed=42)


def _synthetic_cross_record_bundle():
    bundle = _bundle([
        ("a", "clinical", [3, 4], [3, 4], [3, 4], [2., 2.]),
        ("b", "clinical", [0, 1], [0, 1], [0, 1], [2., 2.]),
        ("web", "network", [5, 6], [5, 6], [5, 6], [2., 2.]),
    ])
    ids = bundle.chains["val"][0]["ordered_clip_ids"] + bundle.chains["val"][1]["ordered_clip_ids"]
    chains = [{"chain_id": "private-synthetic-clinical", "split": "val", "ordered_clip_ids": ids,
               "eligible": [True, True, True], "synthetic_breaks": [False, True, False],
               "synthetic_chain": True, "synthetic_origin": "fsn_tcn_clinical_cross_record_v1"},
              {**bundle.chains["val"][2], "synthetic_chain": True, "synthetic_breaks": [False],
               "synthetic_origin": "fsn_tcn_clinical_cross_record_v1"}]
    return bundle, chains


def test_synthetic_gate_denominator_is_known_breaks_not_clips():
    bundle, chains = _synthetic_cross_record_bundle()
    scores = _scores(chains)
    ids = chains[0]["ordered_clip_ids"]
    scores[ids[1], ids[2]] = .1
    result, _ = evaluate_chains(bundle, chains, FixedFeatureClassifier(), scores,
                                strategy="learned", threshold=.9, seed=42)
    assert result["condition"] == "synthetic_disruptions"
    assert result["counts"]["clips"] == 6
    assert result["counts"]["candidate_edges"] == 4
    assert result["gate_audit"]["synthetic_breaks"] == {"count": 1, "open": 0, "open_fraction": 0.}
    assert result["gate_audit"]["original_clinical"] == {"count": 2, "open": 2, "open_fraction": 1.}


def test_shuffle_break_must_be_nonoriginal_within_same_record():
    bundle = _bundle([("a", "clinical", [0, 1, 2], [0, 1, 2], [0, 1, 2], [2., 2., 2.])])
    ids = bundle.chains["val"][0]["ordered_clip_ids"]
    chain = {"chain_id": "private-shuffle", "split": "val", "ordered_clip_ids": [ids[0], ids[2], ids[1]],
             "eligible": [True, True], "synthetic_breaks": [True, True], "synthetic_chain": True,
             "synthetic_origin": "fsn_tcn_clinical_shuffle_v1"}
    assert len(validate_evaluation_chains(bundle, [chain])) == 1
    marked_original = {**chain, "ordered_clip_ids": ids}
    with pytest.raises(ValueError, match="non-original directed pairs"):
        validate_evaluation_chains(bundle, [marked_original])


@pytest.mark.parametrize("mutation,match", [
    ("unmarked", "original legal pairs"), ("mixed_modes", "separately"),
    ("same_record", "distinct group"), ("cross_collection", "source collection"),
    ("network_break", "clinical endpoints"), ("unrecognized", "recognized construction"),
])
def test_synthetic_break_provenance_cannot_bypass_formal_safety(mutation, match):
    bundle, chains = _synthetic_cross_record_bundle()
    ids = chains[0]["ordered_clip_ids"]
    if mutation == "unmarked":
        chains[0]["synthetic_breaks"][1] = False
    elif mutation == "mixed_modes":
        chains[1] = bundle.chains["val"][2]
    elif mutation == "same_record":
        bundle.metadata[ids[2]]["group_id"] = bundle.metadata[ids[1]]["group_id"]
    elif mutation == "cross_collection":
        bundle.metadata[ids[2]]["source_collection"] = "another-clinical-collection"
    elif mutation == "network_break":
        bundle.metadata[ids[2]]["source_kind"] = "network"
    elif mutation == "unrecognized":
        chains[0]["synthetic_origin"] = "made-up"
    with pytest.raises(ValueError, match=match):
        validate_evaluation_chains(bundle, chains)
