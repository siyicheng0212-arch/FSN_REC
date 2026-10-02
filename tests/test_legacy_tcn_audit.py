"""Meaningful sanity checks for paired history and unchanged-model cuts."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.legacy_tcn_audit.audit import (
    _chain_positions, _historical_assertion, _paired, _run_counterfactual,
    _summary, load_historical_predictions,
)


class TinyIndex:
    def __init__(self, values):
        self.values = values

    def read(self, clip):
        value = self.values[clip]
        return {"global_tokens": np.array([[value]], dtype=np.float32),
                "local_tokens": np.zeros((1, 1), dtype=np.float32),
                "logits": np.array([1., 0., 0., 0., 0., 0., 0.], dtype=np.float32)}


class RightNeighborModel(nn.Module):
    def forward(self, features, a_logits):
        output = a_logits.clone()
        for i in range(len(features) - 1):
            if features[i + 1, 0] > 1:
                output[i, 0] = -1
                output[i, 1] = 2
        return output


def tiny_bundle():
    ids = ["first", "middle", "last"]
    return SimpleNamespace(
        index=TinyIndex({"first": 0, "middle": 0, "last": 10}),
        metadata={clip: {"split": "val", "source_kind": "clinical", "label_id": 0}
                  for clip in ids},
        chains={"val": [{"ordered_clip_ids": ids, "eligible": [True, True]}]},
        edges=[{"split": "val", "left_clip_id": "first", "right_clip_id": "middle", "status": "C"},
               {"split": "val", "left_clip_id": "middle", "right_clip_id": "last", "status": "D"}],
    )


def test_cut_invokes_unchanged_model_before_inference_and_repairs_spoil():
    bundle = tiny_bundle()
    history = {clip: {"clip_id": clip, "label_id": 0, "A_prediction": 0,
                      "prediction": 1 if clip == "middle" else 0}
               for clip in bundle.chains["val"][0]["ordered_clip_ids"]}
    report = _run_counterfactual(bundle, history, RightNeighborModel(),
                                 chain_layout="eligible_segments", device="cpu", logits_atol=1e-5)
    assert report["legacy_exact_prediction_parity_clips"] == 3
    assert report["removed_D_edges"] == 1
    assert report["cut_vs_original"]["improved"] == 1
    assert report["old_TCN_spoils_repaired"] == 1
    assert report["resulting_singleton_model_calls"] == 1
    assert report["all_cut_metrics"]["accuracy"] == 1.0
    positions = _chain_positions(bundle)
    assert positions["middle"]["adjacent_status"] == "C+D"
    assert positions["first"]["distance_to_D"] == 1


def test_counterfactual_fails_before_reporting_an_unreproduced_checkpoint():
    bundle = tiny_bundle()
    history = {clip: {"clip_id": clip, "label_id": 0, "A_prediction": 0, "prediction": 0}
               for clip in bundle.chains["val"][0]["ordered_clip_ids"]}
    with pytest.raises(ValueError, match="exact class parity"):
        _run_counterfactual(bundle, history, RightNeighborModel(),
                            chain_layout="eligible_segments", device="cpu", logits_atol=1e-5)


def test_paired_counts_wrong_to_wrong_and_known_history_is_strict():
    assert _paired([0, 0, 0], [1, 0, 1], [0, 1, 2]) == {
        "changed": 3, "improved": 1, "worsened": 1, "wrong_to_different_wrong": 1,
    }
    with pytest.raises(ValueError, match="historical report"):
        _historical_assertion(_summary([
            {"label_id": 0, "A_prediction": 0, "prediction": 0}
        ]))


def test_private_predictions_require_same_823_ids_labels_and_frozen_A(tmp_path):
    ids = [f"c{i}" for i in range(823)]
    bundle = SimpleNamespace(
        metadata={clip: {"split": "val", "label_id": 0} for clip in ids},
        index=TinyIndex({clip: 0 for clip in ids}),
    )
    path = tmp_path / "old.jsonl"
    rows = [{"clip_id": clip, "label_id": 0, "A_prediction": 0, "prediction": 0}
            for clip in ids]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert len(load_historical_predictions(bundle, path)) == 823
    rows[19]["A_prediction"] = 2
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed frozen-A evidence"):
        load_historical_predictions(bundle, path)
