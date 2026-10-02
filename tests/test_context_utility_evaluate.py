"""Known arithmetic, singleton and provenance checks for context evaluation."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.context_utility.evaluate import (
    evaluate_chains, evaluate_same_checkpoint_gate_modes, summarize_comparison,
)


class FeatureClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lengths = []

    def forward(self, features, a_logits):
        self.lengths.append(len(features))
        return torch.nn.functional.one_hot(features[:, 0].long(), num_classes=7).float() * 10


class VisualClassifier(nn.Module):
    def forward(self, features, a_logits):
        return a_logits


class AuditedClassifier(FeatureClassifier):
    def forward(self, features, a_logits, *, gate_mode="learned", permutation_seed=0):
        positions = torch.arange(1, len(features))
        original = torch.arange(len(positions), dtype=torch.float32) / max(1, len(positions)) + .5
        coefficients = torch.ones_like(original) if gate_mode == "unit" else original
        if gate_mode == "permuted":
            coefficients = original.flip(0)
        self.coefficients = [{"path": "private_module.layer_0", "offset": -1,
                              "values": coefficients, "target_positions": positions,
                              "original_values": original}]
        self.seed = permutation_seed
        return a_logits if gate_mode == "unit" else super().forward(features, a_logits)

    def last_coefficients(self):
        return self.coefficients


def bundle(records):
    metadata, arrays, chains = {}, {}, []
    for group, kind, labels, visual, revised, durations in records:
        ids = []
        for i, (label, a_pred, pred, duration) in enumerate(zip(labels, visual, revised, durations)):
            clip = f"private-{group}-{i}"
            ids.append(clip)
            metadata[clip] = {"clip_id": clip, "split": "val", "group_id": group,
                              "recording_key": f"private-record-{group}", "source_video_key": f"private-video-{group}",
                              "timebase_key": "private-timebase", "source_kind": kind,
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
    return bundle([
        ("clinical", "clinical", [3, 4, 0], [4, 3, 1], [3, 4, 2], [.5, 1., 2.]),
        ("network", "network", [3, 4], [3, 4], [4, 0], [5., 6.]),
    ])


def test_paired_known_arithmetic_and_macro_f1_contribution(arithmetic_bundle):
    report, rows = evaluate_chains(arithmetic_bundle, arithmetic_bundle.chains["val"], FeatureClassifier(),
                                    baseline=VisualClassifier())
    assert report["metrics"]["accuracy"] == .4
    assert report["metrics"]["macro_f1"] == pytest.approx((2 / 3 + .5) / 7)
    assert report["metrics"]["paired_vs_A"] == {
        "changed_count": 5, "changed_fraction": 1., "improved": 2, "worsened": 2,
        "wrong_to_different_wrong": 1, "net_correct_gain": 0,
    }
    assert report["metrics"]["sweep_reperfusion"]["sweep_to_reperfusion"] == {
        "count": 1, "true_class_support": 2, "rate": .5,
    }
    assert report["metrics"]["sweep_reperfusion"]["reperfusion_to_sweep"]["count"] == 0
    deltas = report["metrics"]["per_class_delta_vs_A"]
    assert len(deltas) == 7
    assert sum(row["macro_f1_contribution"] for row in deltas) == pytest.approx(
        report["metrics"]["macro_f1"] - report["A_metrics"]["macro_f1"])
    assert report["paired_vs_initial_TCN"]["paired"]["changed_count"] == 5
    assert report["slices"]["duration_le_1s"]["count"] == 2
    assert report["slices"]["duration_gt_1_le_5s"]["count"] == 2
    assert report["slices"]["duration_gt_5s"]["count"] == 1
    assert report["recording_aggregate"]["count"] == 2
    assert report["cross_entropy"] == pytest.approx(np.mean([row["cross_entropy"] for row in rows]))
    public = json.dumps(report, allow_nan=False)
    for row in rows:
        assert row["clip_id"] not in public and row["chain_id"] not in public and row["recording_key"] not in public


def test_singleton_preserves_old_model_semantics_not_a_fallback():
    data = bundle([("solo", "clinical", [3], [3], [4], [2.])])
    scorer = FeatureClassifier()
    report, rows = evaluate_chains(data, data.chains["val"], scorer)
    assert scorer.lengths == [1]
    assert report["counts"]["singleton_segments"] == 1
    assert rows[0]["prediction"] == 4 != rows[0]["A_prediction"]
    assert report["slices"]["network"]["macro_f1"] == 0
    assert report["gate_audit"]["overall"]["count"] == 0


def test_structural_cuts_segment_before_forward_and_include_singletons():
    data = bundle([("record", "clinical", [0] * 4, [0] * 4, [1] * 4, [2.] * 4)])
    data.chains["val"][0]["eligible"] = [False, True, False]
    scorer = FeatureClassifier()
    report, rows = evaluate_chains(data, data.chains["val"], scorer)
    assert scorer.lengths == [1, 2, 1]
    assert report["counts"]["structural_cuts"] == 2
    assert report["counts"]["segments"] == 3
    assert [row["segment_length"] for row in rows] == [1, 2, 2, 1]


def _private(labels, visual, prediction):
    return [{"clip_id": f"private-{i}", "split": "val", "label_id": label,
             "A_prediction": a_pred, "prediction": pred, "source_kind": "clinical", "duration": 2.}
            for i, (label, a_pred, pred) in enumerate(zip(labels, visual, prediction))]


def test_comparison_tracks_preserved_rescues_lost_rescues_and_wrong_switches():
    labels, visual = [0, 1, 2, 3, 4], [6, 6, 2, 3, 5]
    baseline = _private(labels, visual, [0, 1, 4, 3, 6])
    current = _private(labels, visual, [0, 6, 2, 4, 0])
    report = summarize_comparison(current, baseline)
    assert report["baseline_rescues_vs_A"] == 2
    assert report["baseline_rescues_preserved"] == 1
    assert report["baseline_rescues_lost"] == 1
    assert report["baseline_spoils_vs_A"] == 1
    assert report["baseline_spoils_repaired"] == 1
    assert report["paired"] == {"changed_count": 4, "changed_fraction": .8, "improved": 1,
                                 "worsened": 2, "wrong_to_different_wrong": 1, "net_correct_gain": -1}
    assert len(report["per_class_delta"]) == 7
    assert sum(row["macro_f1_contribution"] for row in report["per_class_delta"]) == pytest.approx(report["macro_f1_delta"])
    assert "private-" not in json.dumps(report)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "label", "visual", "source", "duration", "test"])
def test_comparison_rejects_partial_duplicate_or_misaligned_rows(mutation):
    baseline = _private([0, 1], [0, 1], [0, 1])
    current = deepcopy(baseline)
    if mutation == "missing":
        current.pop()
    elif mutation == "duplicate":
        current.append(deepcopy(current[0]))
    else:
        field, value = {"label": ("label_id", 4), "visual": ("A_prediction", 3),
                        "source": ("source_kind", "network"), "duration": ("duration", 3.),
                        "test": ("split", "test")}[mutation]
        current[0][field] = value
    with pytest.raises(ValueError):
        summarize_comparison(current, baseline)


def test_gate_audit_and_same_checkpoint_modes_preserve_distribution_and_privacy():
    data = bundle([("a", "clinical", [0, 1, 2, 3], [1, 2, 3, 4], [0, 1, 2, 3], [2.] * 4)])
    report, rows = evaluate_chains(data, data.chains["val"], AuditedClassifier(), gate_mode="permuted")
    audit = report["gate_audit"]
    assert audit["overall"]["count"] == 3
    assert audit["groups"]["side"]["left"]["count"] == 3
    assert audit["groups"]["source"]["clinical"]["count"] == 3
    assert audit["groups"]["span"]["1"]["count"] == 3
    assert audit["permutation"] == {"compared_positions": 3, "changed_positions": 2, "shuffled_index_positions": 0}
    assert "private_module" not in json.dumps(report)
    suite, private = evaluate_same_checkpoint_gate_modes(data, data.chains["val"], AuditedClassifier())
    assert suite["learned"]["metrics"]["accuracy"] == 1.
    assert suite["unit"]["metrics"]["accuracy"] == 0.
    assert suite["paired_learned_vs_unit"]["paired"]["improved"] == 4
    assert set(private) == {"learned", "unit", "permuted"}
    assert suite["permuted"]["gate_audit"]["overall"] == suite["learned"]["gate_audit"]["overall"]


def test_seven_class_metric_includes_all_labels():
    data = bundle([("all", "network", list(range(7)), list(range(7)), list(range(7)), [2.] * 7)])
    report, _ = evaluate_chains(data, data.chains["val"], VisualClassifier())
    assert report["metrics"]["macro_f1"] == 1.
    assert [row["support"] for row in report["metrics"]["per_class"]] == [1] * 7


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "train", "test", "unsealed"])
def test_validation_provenance_is_checked_before_model_calls(arithmetic_bundle, mutation):
    chains = deepcopy(arithmetic_bundle.chains["val"])
    if mutation == "missing":
        chains.pop()
    elif mutation == "duplicate":
        chains.append({**chains[0], "chain_id": "private-extra"})
    elif mutation == "train":
        arithmetic_bundle.metadata[chains[0]["ordered_clip_ids"][0]]["split"] = "train"
    elif mutation == "test":
        chains[0]["split"] = "test"
    else:
        chains[0]["ordered_clip_ids"].reverse()
    scorer = FeatureClassifier()
    with pytest.raises(ValueError):
        evaluate_chains(arithmetic_bundle, chains, scorer)
    assert scorer.lengths == []


def test_synthetic_breaks_are_separate_and_have_explicit_provenance():
    data = bundle([
        ("a", "clinical", [0, 1], [0, 1], [0, 1], [2., 2.]),
        ("b", "clinical", [2, 3], [2, 3], [2, 3], [2., 2.]),
    ])
    ids = data.chains["val"][0]["ordered_clip_ids"] + data.chains["val"][1]["ordered_clip_ids"]
    chain = {"chain_id": "private-synthetic", "split": "val", "ordered_clip_ids": ids,
             "eligible": [True, True, True], "synthetic_breaks": [False, True, False],
             "synthetic_chain": True, "synthetic_origin": "fsn_tcn_clinical_cross_record_v1"}
    report, rows = evaluate_chains(data, [chain], AuditedClassifier())
    assert report["condition"] == "synthetic_disruptions"
    assert report["counts"]["synthetic_breaks"] == 1
    assert report["gate_audit"]["groups"]["connection"]["crosses_synthetic_break"]["count"] == 1
    chain["synthetic_breaks"][1] = False
    with pytest.raises(ValueError, match="original legal pairs"):
        evaluate_chains(data, [chain], AuditedClassifier())


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_rejects_nonfinite_outputs(value):
    class Invalid(nn.Module):
        def forward(self, features, a_logits):
            return a_logits * value
    data = bundle([("solo", "clinical", [0], [0], [0], [2.])])
    with pytest.raises(ValueError, match="finite logits"):
        evaluate_chains(data, data.chains["val"], Invalid())


def test_real_wrapper_audit_integration_and_same_initial_predictions():
    from experiments.context_utility.model import ContextUtilityTCN
    from experiments.fsn_tcn.model import TCNConfig, TemporalResidualTCN
    data = bundle([("record", "clinical", [0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [0, 1, 2, 3, 4], [2.] * 5)])
    base = TemporalResidualTCN(TCNConfig(input_dim=2, width=4, layers=2, dropout=0)).eval()
    torch.nn.init.normal_(base.output_projection.weight, std=.1)
    initial = deepcopy(base)
    model = ContextUtilityTCN(base, 2, ["blocks.0.temporal", "blocks.1.temporal"], dim=4)
    report, private = evaluate_chains(data, data.chains["val"], model, baseline=initial)
    assert report["paired_vs_initial_TCN"]["paired"]["changed_count"] == 0
    assert report["gate_audit"]["overall"]["count"] == 2 * 4 + 2 * 3
    assert report["gate_audit"]["overall"]["mean"] == 1.
    assert set(report["gate_audit"]["groups"]["span"]) == {"1", "2"}
    suite, rows = evaluate_same_checkpoint_gate_modes(data, data.chains["val"], model)
    assert suite["paired_learned_vs_unit"]["paired"]["changed_count"] == 0
    assert suite["permuted"]["gate_audit"]["permutation"]["changed_positions"] == 0
    assert "blocks.0.temporal" not in json.dumps(suite)
