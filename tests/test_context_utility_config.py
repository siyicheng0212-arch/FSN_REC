"""Meaningful provenance and true-model adaptation checks, using tiny assets."""

import copy
import importlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.context_utility.adapters import LegacyPredictionAdapter
from experiments.context_utility.config import (
    _architecture, baseline_source_hashes, build_base, code_hashes, load_config,
)
from experiments.fsn_tcn.model import TCNConfig, TemporalResidualTCN
from experiments.relation.data import sha256_file


PACKAGE = Path(__file__).resolve().parents[1] / "experiments" / "context_utility"
FP = {"sealed_evidence": "test-only-frozen-A"}


def reference_config():
    return json.loads((PACKAGE / "reference.json").read_text())


def save_config(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def reference_checkpoint(tmp_path, config=None):
    config = config or reference_config()
    model = TemporalResidualTCN(TCNConfig(input_dim=5, **config["baseline"]["config"]))
    with torch.no_grad():
        model.output_projection.weight.normal_()
        model.output_projection.bias.normal_()
    saved = {"model": model.state_dict(), "fingerprint": FP, "strategy": "all",
             "config": {"baseline": {"kind": "reference", "config": config["baseline"]["config"]}}}
    path = tmp_path / "reference.pt"
    torch.save(saved, path)
    return model, path, saved


def test_explicit_reference_config_and_code_seals():
    config = load_config(PACKAGE / "reference.json")
    assert config["seeds"] == [42]
    assert config["augmentation"]["anchor_loss_weight"] == 1
    hashes = code_hashes()
    assert "experiments/context_utility/config.py" in hashes
    assert "experiments/fsn_tcn/model.py" in hashes
    assert any(key.startswith("experiments/relation/") for key in hashes)
    assert all(len(digest) == 64 for digest in hashes.values())


@pytest.mark.parametrize("section,key,value", [
    (None, "schema", "other"), (None, "seeds", [42, 42]),
    (None, "seeds", [True]), (None, "challenge_seed", -1),
    ("module", "dim", 0), ("augmentation", "max_neighbors", 0),
    ("augmentation", "anchor_loss_weight", float("nan")),
    ("optimization", "loss", "weightedCE"), ("optimization", "optimizer", "RMSprop"),
    ("optimization", "lr", float("inf")), ("optimization", "batch_size", False),
    ("baseline", "kind", "pretend_legacy"), ("baseline", "feature_contract", "unknown"),
    ("baseline", "temporal_paths", ["blocks.0.temporal"]),
    ("baseline", "checkpoint_state_key", 1),
])
def test_reject_invalid_configuration(tmp_path, section, key, value):
    config = reference_config()
    (config if section is None else config[section])[key] = value
    with pytest.raises(ValueError):
        load_config(save_config(tmp_path, config))


def test_missing_architecture_value_is_not_filled_from_defaults(tmp_path):
    config = reference_config()
    config["baseline"]["config"].pop("dropout")
    with pytest.raises(ValueError, match="all five"):
        load_config(save_config(tmp_path, config))


def test_zero_weight_decay_and_explicit_sgd_momentum(tmp_path):
    config = reference_config()
    config["optimization"].update(optimizer="SGD", momentum=.9, weight_decay=0)
    assert load_config(save_config(tmp_path, config))["optimization"]["optimizer"] == "SGD"
    config["optimization"].pop("momentum")
    with pytest.raises(ValueError, match="momentum"):
        load_config(save_config(tmp_path, config))


def test_reference_strict_load_reproduces_nonzero_checkpoint(tmp_path):
    config = reference_config()
    original, path, _ = reference_checkpoint(tmp_path, config)
    loaded, audit = build_base(config, 5, path, fingerprint=FP)
    original.eval()
    loaded.eval()
    features, logits = torch.randn(6, 5), torch.randn(6, 7)
    torch.testing.assert_close(loaded(features, logits), original(features, logits), rtol=0, atol=0)
    assert audit["checkpoint_sha256"] == sha256_file(path)
    assert audit["strict_state_load"] is True
    assert audit["checkpoint_base_source_check"] == {"available": False, "checked": False, "key": None}
    assert audit["architecture"]["blocks.3.temporal"]["dilation"] == [8]


def test_reference_checks_recorded_base_source_without_requiring_entire_old_code_seal(tmp_path):
    config = reference_config()
    _, path, saved = reference_checkpoint(tmp_path, config)
    model_key = "experiments/fsn_tcn/model.py"
    model_source = PACKAGE.parent / "fsn_tcn" / "model.py"
    saved["protocol"] = {"code_sha256": {
        model_key: sha256_file(model_source),
        "experiments/fsn_tcn/train.py": "historical-other-code-not-current-package",
    }}
    torch.save(saved, path)
    _, audit = build_base(config, 5, path, fingerprint=FP)
    check = audit["checkpoint_base_source_check"]
    assert check["available"] is True and check["checked"] is True
    assert check["sha256"] == sha256_file(model_source)
    saved["protocol"]["code_sha256"][model_key] = "0" * 64
    torch.save(saved, path)
    with pytest.raises(ValueError, match="base-model source SHA"):
        build_base(config, 5, path, fingerprint=FP)


@pytest.mark.parametrize("field,value,match", [
    ("fingerprint", {"wrong": "evidence"}, "fingerprint"),
    ("strategy", "learned", "full-open"),
    ("config", {"baseline": {"kind": "reference", "config": {}}}, "architecture"),
])
def test_reference_rejects_wrong_evidence_strategy_or_architecture(tmp_path, field, value, match):
    config = reference_config()
    _, path, saved = reference_checkpoint(tmp_path, config)
    saved[field] = value
    torch.save(saved, path)
    with pytest.raises(ValueError, match=match):
        build_base(config, 5, path, fingerprint=FP)


def test_reference_rejects_unsealed_or_missing_or_prefix_modified_state(tmp_path):
    config = reference_config()
    _, path, saved = reference_checkpoint(tmp_path, config)
    with pytest.raises(ValueError, match="fingerprint"):
        build_base(config, 5, path)
    saved["model"] = {"module." + key: value for key, value in saved["model"].items()}
    torch.save(saved, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        build_base(config, 5, path, fingerprint=FP)
    config["baseline"]["checkpoint_state_key"] = "state_dict"
    with pytest.raises(ValueError, match="explicitly declared"):
        build_base(config, 5, path, fingerprint=FP)


@pytest.mark.parametrize("prediction_input", ["logits", "probabilities"])
@pytest.mark.parametrize("layout", ["BCT", "BTC", "TC", "CT"])
def test_adapter_uses_declared_predictions_and_layout(prediction_input, layout):
    class Capture(nn.Module):
        def forward(self, values):
            self.values = values
            if layout == "BCT":
                return values[:, -7:, :]
            if layout == "BTC":
                return values[:, :, -7:]
            if layout == "CT":
                return values[-7:, :]
            return values[:, -7:]
    raw = Capture()
    adapter = LegacyPredictionAdapter(raw, prediction_input=prediction_input,
                                      input_layout=layout, output_layout=layout)
    features, logits = torch.randn(3, 5), torch.randn(3, 7)
    actual = adapter(features, logits)
    expected = logits if prediction_input == "logits" else logits.softmax(-1)
    torch.testing.assert_close(actual, expected)
    assert raw.values.numel() == 3 * 12


def test_adapter_preserves_actual_singleton_behavior():
    raw = nn.Linear(12, 7)
    adapter = LegacyPredictionAdapter(raw, prediction_input="logits", input_layout="TC", output_layout="TC")
    features, logits = torch.randn(1, 5), torch.randn(1, 7)
    expected = raw(torch.cat([features, logits], -1))
    torch.testing.assert_close(adapter(features, logits), expected)
    assert not torch.equal(expected, logits)
    fallback = LegacyPredictionAdapter(raw, prediction_input="logits", input_layout="TC", output_layout="TC", singleton="a_logits")
    assert fallback(features, logits) is logits


def test_adapter_refuses_hidden_output_conversion_or_invalid_data():
    class TupleOutput(nn.Module):
        def forward(self, values):
            return (values, values)
    adapter = LegacyPredictionAdapter(TupleOutput(), prediction_input="logits", input_layout="TC", output_layout="TC")
    with pytest.raises(ValueError, match="must return a tensor"):
        adapter(torch.randn(3, 5), torch.randn(3, 7))
    with pytest.raises(ValueError, match="finite"):
        adapter(torch.full((3, 5), float("nan")), torch.randn(3, 7))


@pytest.fixture
def legacy_assets(tmp_path, monkeypatch):
    """An actual independent module, not a claimed reconstruction of user code."""
    source = tmp_path / "test_verified_legacy.py"
    source.write_text('''from torch import nn\nfrom experiments.context_utility.adapters import LegacyPredictionAdapter\n\nclass ActualToy(nn.Module):\n    def __init__(self, input_dim):\n        super().__init__()\n        self.temporal = nn.Conv1d(input_dim+7,7,3,padding=1)\n    def forward(self, x):\n        return self.temporal(x)\n\ndef build(input_dim):\n    return LegacyPredictionAdapter(ActualToy(input_dim),prediction_input="probabilities",input_layout="BCT",output_layout="BCT",singleton="delegate")\n''')
    monkeypatch.syspath_prepend(str(tmp_path))
    # Separate test files can use the same name after a fresh fixture directory.
    import sys
    sys.modules.pop("test_verified_legacy", None)
    importlib.invalidate_caches()
    config = reference_config()
    config["baseline"] = {
        "kind": "legacy", "factory": "test_verified_legacy:build", "kwargs": {},
        "feature_contract": "pooled_global_local_mean_frozen_A_logits_v1",
        "temporal_paths": ["model.temporal"], "source_dependencies": [str(source)],
        "checkpoint_state_key": None, "checkpoint_target": "model",
        "legacy_audit": str(tmp_path / "legacy_audit.json"),
    }
    model = importlib.import_module("test_verified_legacy").build(input_dim=5)
    checkpoint = tmp_path / "legacy_raw.pt"
    torch.save(model.model.state_dict(), checkpoint)
    audit = {
        "schema": "fsn-context-utility-legacy-audit-v1",
        "checkpoint_sha256": sha256_file(checkpoint), "fingerprint": FP,
        "source_sha256": baseline_source_hashes(config),
        "semantics": {
            "input_a_predictions": "probabilities", "feature_normalization": "identity after frozen pooled means",
            "input_layout": "BCT", "temporal_architecture": _architecture(model, ["model.temporal"]),
            "singleton": "delegate", "loss": "CE", "optimizer": "AdamW",
        },
    }
    Path(config["baseline"]["legacy_audit"]).write_text(json.dumps(audit))
    return config, model, checkpoint, audit


def test_true_legacy_factory_explicit_raw_state_target_and_audit(legacy_assets):
    config, original, checkpoint, _ = legacy_assets
    loaded, audit = build_base(config, 5, checkpoint, fingerprint=FP)
    features, logits = torch.randn(3, 5), torch.randn(3, 7)
    torch.testing.assert_close(loaded(features, logits), original(features, logits), rtol=0, atol=0)
    assert audit["checkpoint_target"] == "model"
    assert audit["legacy_audit"]["semantics"]["singleton"] == "delegate"


@pytest.mark.parametrize("field,value,match", [
    ("checkpoint_sha256", "0" * 64, "checkpoint SHA"),
    ("fingerprint", {"wrong": "bundle"}, "fingerprint"),
    ("source_sha256", {}, "source hashes"),
    ("schema", "invented", "schema"),
])
def test_legacy_refuses_invalid_provenance(legacy_assets, field, value, match):
    config, _, checkpoint, audit = legacy_assets
    audit[field] = value
    Path(config["baseline"]["legacy_audit"]).write_text(json.dumps(audit))
    with pytest.raises(ValueError, match=match):
        build_base(config, 5, checkpoint, fingerprint=FP)


@pytest.mark.parametrize("field,value", [
    ("input_a_predictions", "logits"), ("input_layout", "BTC"),
    ("singleton", "a_logits"), ("temporal_architecture", {}),
    ("loss", "weightedCE"), ("optimizer", "SGD"),
])
def test_legacy_refuses_mismatched_verified_semantics(legacy_assets, field, value):
    config, _, checkpoint, audit = legacy_assets
    audit["semantics"][field] = value
    Path(config["baseline"]["legacy_audit"]).write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        build_base(config, 5, checkpoint, fingerprint=FP)


def test_legacy_template_does_not_create_a_fake_three_block_model():
    config = load_config(PACKAGE / "legacy_template.json")
    with pytest.raises(ValueError, match="not runnable"):
        baseline_source_hashes(config)
