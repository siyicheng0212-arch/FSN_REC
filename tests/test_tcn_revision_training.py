"""Check the full old-chain contract and the explicit revision experiment plan."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.selective_context.revision_config import VARIANTS, load_config
from experiments.selective_context.revision_train import _train_chain
from experiments.selective_context.revision_model import RevisionGateTCN


class _OldTemporal(nn.Module):
    def forward(self, features, a_logits):
        # A legacy singleton is *not* forced to A. Its chain mean changes when
        # the caller crosses a structural edge, making the layout test useful.
        delta = torch.zeros_like(a_logits)
        delta[:, 0] = features.mean(0)[0]
        return a_logits + delta


def _configuration():
    return {"schema": "fsn-tcn-revision-v1", "seeds": [42],
            "variants": list(VARIANTS), "module": {"gate_dim": 64},
            "optimization": {"epochs": 30, "patience": 5, "chains_per_step": 4,
                             "lr": .001, "weight_decay": .01, "grad_clip": 5.}}


def test_revision_config_requires_all_matched_variants_and_no_val_threshold(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(_configuration()))
    assert load_config(path)["seeds"] == [42]
    value = _configuration()
    value["variants"].remove("scalar")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_config(path)
    value = _configuration()
    value["module"]["edge_threshold"] = .5
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_config(path)


def test_revision_training_preserves_full_or_preexisting_segment_layout(monkeypatch):
    import experiments.selective_context.revision_train as trainer

    features = torch.tensor([[0.], [2.], [9.]])
    a_logits = torch.tensor([[1., 0., 0., 0., 0., 0., 0.],
                             [0., 1., 0., 0., 0., 0., 0.],
                             [0., 0., 1., 0., 0., 0., 0.]])
    labels = torch.tensor([0, 1, 2])
    chain = {"ordered_clip_ids": ["a", "b", "c"], "eligible": [True, False]}
    monkeypatch.setattr(trainer, "load_chain", lambda bundle, current, device: (features, a_logits, labels))
    model = RevisionGateTCN(_OldTemporal(), feature_dim=1, variant="scalar")
    with torch.no_grad():
        model.scalar_logit.fill_(10)
    full, count = _train_chain(SimpleNamespace(), chain, model, "cpu", "full_chain")
    segmented, same_count = _train_chain(SimpleNamespace(), chain, model, "cpu", "eligible_segments")
    assert count == same_count == 3
    expected_full = F.cross_entropy(model(features, a_logits), labels, reduction="sum")
    expected_segments = (
        F.cross_entropy(model(features[:2], a_logits[:2]), labels[:2], reduction="sum") +
        F.cross_entropy(model(features[2:], a_logits[2:]), labels[2:], reduction="sum")
    )
    assert torch.equal(full, expected_full)
    assert torch.equal(segmented, expected_segments)
    assert float(full) != float(segmented)
    assert all(not p.requires_grad for p in model.base_tcn.parameters())
