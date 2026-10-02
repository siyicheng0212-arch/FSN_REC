"""CPU end-to-end engineering fixtures, never historical FSN result claims."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from test_fsn_tcn_suite import tiny_bundle
from test_context_utility_config import legacy_assets, reference_checkpoint, reference_config
from experiments.context_utility.augmentation import apply_view, build_epoch_plan
from experiments.context_utility.config import VARIANTS, build_base
from experiments.context_utility.data import validate_source_kinds
from experiments.context_utility.evaluate_suite import evaluate_suite
from experiments.context_utility.smoke import main as smoke_main, run_checks
from experiments.context_utility.train import (
    load_trained, make_model, sealed_inputs, train_variant, validate_smoke,
)
from experiments.fsn_tcn.io import load_chain
from experiments.relation.data import sha256_file


def small_config():
    config = reference_config()
    config["baseline"]["config"].update(width=4, layers=2, dropout=0.)
    config["baseline"]["temporal_paths"] = ["blocks.0.temporal", "blocks.1.temporal"]
    config["module"]["dim"] = 4
    config["optimization"].update(epochs=2, patience=2, batch_size=2)
    return config


def reference_start(tmp_path, config, bundle):
    original, checkpoint, saved = reference_checkpoint(tmp_path, config)
    saved["fingerprint"] = bundle.fingerprint
    # Deliberately small nonzero trained-head stand-in so gate gradients are
    # exercised. This is not a recovered 79.20% checkpoint.
    with torch.no_grad():
        original.output_projection.weight.normal_(std=.05)
        original.output_projection.bias.zero_()
    saved["model"] = original.state_dict()
    torch.save(saved, checkpoint)
    return checkpoint


def assert_suite(tmp_path, bundle, config, checkpoint):
    torch.set_num_threads(1)
    before = sha256_file(checkpoint)
    models = tmp_path / "models"
    plan_seals = []
    for variant in VARIANTS:
        result = train_variant(bundle, config, checkpoint, 42, variant, models / variant, "cpu")
        assert result["metrics"]["count"] == 8
        assert result["A_trainable_parameters"] == 0
        assert result["added_parameters"] > 0 if variant in {"scalar_aug", "dynamic_aug"} else result["added_parameters"] == 0
        history = json.loads((models / variant / "history.json").read_text())
        assert all(row["original_clip_visits"] == 8 for row in history)
        assert all(row["optimizer_steps"] > 0 for row in history)
        protocol = json.loads((models / variant / "protocol.json").read_text())
        plan_seals.append(protocol["plan_sha256_per_epoch"])
        loaded, state, _ = load_trained(models / variant / "best.pt", bundle, config,
                                       checkpoint, variant, 42, "cpu")
        assert state["epoch"] == result["best_epoch"]
    assert all(value == plan_seals[0] for value in plan_seals)
    assert sha256_file(checkpoint) == before
    summary = evaluate_suite(bundle, config, checkpoint, 42, models, tmp_path / "evaluation", "cpu")
    assert set(summary["groups"]) == set(VARIANTS)
    for group in summary["groups"].values():
        assert set(group["conditions"]) == {"original", "clinical_cross_record", "clinical_shuffle"}
        assert all(report["metrics"]["count"] == 8 for report in group["conditions"].values())
        assert all("paired_vs_augmented_TCN" in report for report in group["conditions"].values())
    dynamic = summary["groups"]["dynamic_aug"]
    assert all("paired_learned_vs_unit" in report for report in dynamic["same_checkpoint_gate_controls"].values())
    public = json.dumps(summary)
    assert all(clip not in public for clip in bundle.metadata)
    assert all(row["recording_key"] not in public for row in bundle.metadata.values())
    assert "/private/" not in public and str(tmp_path) not in public
    assert summary["same_predeclared_plans"] and summary["independent_test_read"] is False
    with pytest.raises(FileExistsError):
        train_variant(bundle, config, checkpoint, 42, "continue", models / "continue", "cpu")


def test_reference_all_four_cpu_training_and_paired_evaluation(tmp_path):
    bundle, config = tiny_bundle(tmp_path), small_config()
    checkpoint = reference_start(tmp_path, config, bundle)
    assert_suite(tmp_path, bundle, config, checkpoint)


def test_actual_legacy_factory_four_cpu_groups_without_rebuilding_legacy(tmp_path, legacy_assets):
    config, original, checkpoint, audit = legacy_assets
    bundle = tiny_bundle(tmp_path)
    config["module"]["dim"] = 4
    config["optimization"].update(epochs=2, patience=2, batch_size=2)
    audit["fingerprint"] = bundle.fingerprint
    Path(config["baseline"]["legacy_audit"]).write_text(json.dumps(audit))
    assert_suite(tmp_path, bundle, config, checkpoint)


def test_initialization_shares_base_weights_and_continuation_rng(tmp_path):
    bundle, config = tiny_bundle(tmp_path), small_config()
    checkpoint = reference_start(tmp_path, config, bundle)
    states, random_draws = [], []
    for variant in VARIANTS:
        model, _ = make_model(bundle, config, checkpoint, variant, 42)
        # Wrapper changes only the path of the original conv tensor.
        states.append({key.replace(".temporal.conv.", ".temporal."): value
                       for key, value in model.base.state_dict().items()})
        random_draws.append(torch.rand(5))
    assert all(all(torch.equal(states[0][key], value) for key, value in state.items()) for state in states)
    assert all(torch.equal(random_draws[0], value) for value in random_draws)


def test_real_pair_view_cpu_smoke_exercises_later_gate_gradients(tmp_path):
    bundle, config = tiny_bundle(tmp_path), small_config()
    checkpoint = reference_start(tmp_path, config, bundle)
    plan, _ = build_epoch_plan(bundle, seed=42, epoch=1, max_neighbors=1)
    entry = next(row for row in plan if row["replacements"])
    features, logits, labels = load_chain(bundle, entry)
    def read_clip(clip):
        values, scores, _ = load_chain(bundle, {"ordered_clip_ids": [clip]})
        return values[0], scores[0]
    view = apply_view(features, logits, entry, read_clip)
    model, _ = make_model(bundle, config, checkpoint, "dynamic_aug", 42)
    baseline, _ = build_base(config, 5, checkpoint, fingerprint=bundle.fingerprint)
    report = run_checks(model, baseline, features, logits, labels, anchor_view=view)
    assert report["updated_gate_parameter_tensors"] > 0
    assert report["dynamic_projection_later_step_gradient"]
    assert report["dynamic_early_gate_later_step_gradient"]
    assert report["paired_anchor_view_exercised"]


def test_cpu_cannot_certify_real_cuda_smoke(tmp_path):
    output = tmp_path / "smoke.json"
    with pytest.raises(RuntimeError, match="real CUDA"):
        smoke_main(["--feature-index", "missing", "--source-protocol-dir", "missing", "--config", "missing",
                    "--base-checkpoint", "missing", "--output", str(output), "--device", "cpu"])
    report = json.loads(output.read_text())
    assert report["all_passed"] is False and report["real_cuda_context_utility_smoke_completed"] is False


def test_smoke_seals_include_legacy_audit_bytes(tmp_path, legacy_assets):
    config, _, checkpoint, audit = legacy_assets
    bundle = tiny_bundle(tmp_path)
    audit["fingerprint"] = bundle.fingerprint
    path = Path(config["baseline"]["legacy_audit"])
    path.write_text(json.dumps(audit))
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config))
    report = {**sealed_inputs(bundle, config, checkpoint), "config_sha256": sha256_file(cfg),
              "all_passed": True, "real_cuda_context_utility_smoke_completed": True}
    validate_smoke(report, bundle, config, cfg, checkpoint)
    path.write_text(json.dumps(audit, indent=2))
    with pytest.raises(RuntimeError, match="legacy_audit_sha256"):
        validate_smoke(report, bundle, config, cfg, checkpoint)


def test_trained_state_refuses_code_or_variant_mismatch(tmp_path):
    bundle, config = tiny_bundle(tmp_path), small_config()
    config["optimization"].update(epochs=1, patience=1)
    checkpoint = reference_start(tmp_path, config, bundle)
    out = tmp_path / "single"
    train_variant(bundle, config, checkpoint, 42, "dynamic_aug", out, "cpu")
    with pytest.raises(ValueError, match="variant/seed"):
        load_trained(out / "best.pt", bundle, config, checkpoint, "continue", 42)
    saved = torch.load(out / "best.pt", weights_only=True)
    saved["source_sha256"] = {}
    torch.save(saved, out / "altered.pt")
    with pytest.raises(ValueError, match="source_sha256"):
        load_trained(out / "altered.pt", bundle, config, checkpoint, "dynamic_aug", 42)


def test_fsn_network_source_is_checked_without_rewriting_artifacts(tmp_path):
    bundle = tiny_bundle(tmp_path)
    clip = next(clip for clip, row in bundle.metadata.items() if row["source_kind"] == "network")
    bundle.metadata[clip]["source_collection"] = "FSN"
    assert validate_source_kinds(bundle) is bundle
    bundle.metadata[clip]["source_kind"] = "clinical"
    with pytest.raises(ValueError, match="known source map"):
        validate_source_kinds(bundle)
    assert bundle.metadata[clip]["source_kind"] == "clinical"
