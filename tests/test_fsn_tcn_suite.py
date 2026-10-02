"""Meaningful CPU integration and formal-launch blocking tests; no CUDA claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.fsn_tcn.config import build_base, load_config
from experiments.fsn_tcn.evaluate_suite import evaluate_suite
from experiments.fsn_tcn.io import Bundle, validate_chains
from experiments.fsn_tcn.run_suite import aggregate_seeds, build_commands, execute_plan, gpu_indices
from experiments.fsn_tcn.smoke import main as smoke_main, run_tcn_checks
from experiments.fsn_tcn.train import seed_everything, train_relation, train_tcn
from experiments.relation.data import load_feature_index, sha256_file
from experiments.relation.source_edges import build_source_relations

ROOT = Path(__file__).resolve().parents[1]


def config():
    result = load_config(ROOT / "experiments/fsn_tcn/minimal_reference.json")
    result["threshold"] = .0001  # Engineering fixture only: ensure a trainable learned segment.
    result["baseline"]["config"].update(width=4, layers=2, dropout=0.)
    for section in ("relation", "optimization"):
        result[section].update(epochs=2, patience=1, batch_size=2)
    return result


def tiny_bundle(tmp_path):
    rows = []
    for split in ("train", "val"):
        for group, source, labels in (("c1", "lishui", [0, 3, 4]), ("c2", "lishui", [1, 3, 4]),
                                      ("web", "youtube", [5, 6])):
            for i, label in enumerate(labels):
                rows.append({"clip_id": f"{split}-{group}-{i}", "group_id": f"{split}-{group}",
                             "split": split, "source_collection": source,
                             "video_path": f"/private/{split}-{group}.mp4", "record_id": f"record-{split}-{group}",
                             "label_id": label, "clip_start_sec": float(i * 2), "clip_end_sec": float(i * 2 + 1)})
    artifacts = build_source_relations(rows, {"lishui": "clinical", "youtube": "network"})
    metadata = {row["clip_id"]: row for row in artifacts["metadata"]}
    directory = tmp_path / "features"
    directory.mkdir()
    feature_rows = []
    rng = np.random.default_rng(123)
    for number, (clip, row) in enumerate(metadata.items()):
        path = directory / f"{number}.npz"
        logits = np.zeros(7, dtype=np.float32)
        logits[row["label_id"]] = .1
        np.savez(path, global_tokens=rng.normal(size=(8, 2)).astype(np.float32),
                 local_tokens=rng.normal(size=(12, 3)).astype(np.float32),
                 global_positions=np.linspace(0, 1, 8, dtype=np.float32),
                 local_positions=np.linspace(0, 1, 12, dtype=np.float32), logits=logits)
        feature_rows.append(row | {"feature_path": path.name, "checkpoint_sha256": "a" * 64,
                                   "feature_sha256": sha256_file(path)})
    index_path = directory / "index.jsonl"
    index_path.write_text("".join(json.dumps(row) + "\n" for row in feature_rows))
    index = load_feature_index(index_path)
    # Tiny Bundle is internal helper-test data; public load_bundle rejects these counts.
    return Bundle(index, metadata, artifacts["chains"], artifacts["edges"], {"unit_fixture_only": True})


def test_shared_initialization_and_cpu_updates_after_isolation_checks():
    torch.set_num_threads(1)
    cfg = config()
    seed_everything(42)
    a = build_base(cfg, 5)
    seed_everything(42)
    b = build_base(cfg, 5)
    assert all(torch.equal(a.state_dict()[key], value) for key, value in b.state_dict().items())
    features, logits = torch.randn(4, 5), torch.randn(4, 7)
    report = run_tcn_checks(a, features, logits, torch.tensor([0, 3, 4, 6]))
    assert report["all_passed"] and report["updated_parameter_tensors"] > 0
    assert report["right_to_left_isolated_before_and_after_updates"]


def test_cpu_full_pipeline_relation_four_groups_and_challenges(tmp_path):
    torch.set_num_threads(1)
    bundle, cfg = tiny_bundle(tmp_path), config()
    rdir = tmp_path / "relation"
    rmetrics = train_relation(bundle, cfg, 42, rdir, "cpu")
    assert rmetrics["hard_negative_audit"]
    models = tmp_path / "models"
    for strategy in ("all", "source_rule", "random", "learned"):
        metrics = train_tcn(bundle, cfg, 42, strategy, rdir, models / strategy, "cpu")
        assert metrics["metrics"]["count"] == 8
        assert metrics["A_trainable_parameters"] == metrics["R_trainable_parameters_during_TCN"] == 0
        if strategy == "source_rule":
            assert metrics["slices"]["network"]["paired_vs_A"]["changed_count"] == 0
    summary = evaluate_suite(bundle, cfg, 42, rdir, models, tmp_path / "evaluation", "cpu")
    assert set(summary["groups"]) == {"all", "source_rule", "random", "learned"}
    for group in summary["groups"].values():
        assert set(group["conditions"]) == {"original", "clinical_cross_record", "clinical_shuffle"}
        assert all(value["metrics"]["count"] == 8 for value in group["conditions"].values())
    safe = json.dumps(summary)
    assert all(clip not in safe for clip in bundle.metadata)
    assert "/private/" not in safe
    assert "checkpoint_sha256" not in safe
    for path in (rdir, models / "all", tmp_path / "evaluation"):
        with pytest.raises(FileExistsError):
            if path == rdir:
                train_relation(bundle, cfg, 42, path, "cpu")
            elif path == tmp_path / "evaluation":
                evaluate_suite(bundle, cfg, 42, rdir, models, path, "cpu")
            else:
                train_tcn(bundle, cfg, 42, "all", rdir, path, "cpu")


def test_original_chain_validation_rejects_test_cross_record_and_coverage(tmp_path):
    bundle = tiny_bundle(tmp_path)
    validate_chains(bundle.chains["train"], bundle.metadata, "train")
    cases = []
    case = deepcopy(bundle.chains["train"])
    case[0]["split"] = "test"
    cases.append(case)
    case = deepcopy(bundle.chains["train"])
    case[0]["ordered_clip_ids"][1] = case[1]["ordered_clip_ids"][1]
    cases.append(case)
    cases.append(bundle.chains["train"][:-1])
    for chains in cases:
        with pytest.raises(ValueError):
            validate_chains(chains, bundle.metadata, "train")


def test_cuda_preflight_cannot_be_replaced_with_cpu(tmp_path):
    output = tmp_path / "smoke.json"
    with pytest.raises(RuntimeError, match="real CUDA"):
        smoke_main(["--feature-index", "missing", "--source-protocol-dir", "missing", "--config", "missing",
                    "--raw-cache-report", "missing", "--output", str(output), "--device", "cpu"])
    report = json.loads(output.read_text())
    assert report["all_passed"] is False and report["real_cuda_tcn_smoke_completed"] is False


def test_launcher_commands_and_failure_blocking(tmp_path):
    args = SimpleNamespace(output=str(tmp_path / "suite"), python="/usr/bin/python", feature_index="index",
                           source_protocol_dir="protocol", config="config", manifest_dir="manifests",
                           cache_root="cache", original_checkpoint="A.pt", gpus=[0, 1, 2, 3])
    cfg = config()
    commands = build_commands(args, cfg)
    assert "--execute" not in sum(commands.values(), [])
    for strategy, gpu in zip(("all", "source_rule", "random", "learned"), (0, 1, 2, 3)):
        assert f"cuda:{gpu}" in commands[f"{strategy}_42"]
        assert "--smoke-report" in commands[f"{strategy}_42"]
    plan = {"arguments": vars(args), "config": cfg, "commands": commands,
            "repository": {"commit": "unit-commit", "clean": True}}
    visited = []
    def fail_raw(stage, command):
        visited.append(stage)
        return 7 if stage == "raw_cache_smoke" else 0
    with pytest.raises(RuntimeError, match="exit 7"):
        execute_plan(plan, run=fail_raw)
    assert visited == ["unit_tests", "raw_cache_smoke"]
    assert json.loads((Path(args.output) / "suite_result.json").read_text())["training_completed"] is False
    with pytest.raises(FileExistsError):
        execute_plan(plan, run=fail_raw)


@pytest.mark.parametrize("change", ["seed", "threshold", "challenge_seed", "kind", "ratio"])
def test_reject_config_silent_protocol_changes(tmp_path, change):
    cfg = config()
    if change == "seed":
        cfg["seeds"] = [42, 42]
    elif change == "threshold":
        cfg["threshold"] = float("nan")
    elif change == "challenge_seed":
        cfg.pop("challenge_seed")
    elif change == "kind":
        cfg["baseline"]["kind"] = "guessed_legacy"
    else:
        cfg["hard_negative_ratio"] = 0
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        load_config(path)


def test_four_distinct_physical_gpu_validation():
    assert gpu_indices("0,1,2,3") == [0, 1, 2, 3]
    for value in ("0,1", "0,0,2,3", "-1,1,2,3", "cuda:0"):
        with pytest.raises(Exception):
            gpu_indices(value)


def test_all_seeds_summary_reports_paired_gain_and_single_seed_uncertainty():
    def summary(value, a):
        return {"groups": {strategy: {"conditions": {condition: {
            "metrics": {"macro_f1": value, "accuracy": value},
            "A_metrics": {"macro_f1": a, "accuracy": a}}
            for condition in ("original", "clinical_cross_record", "clinical_shuffle")}}
            for strategy in ("all", "source_rule", "random", "learned")}}
    single = aggregate_seeds([summary(.8, .75)])
    assert single["groups"]["learned"]["original"]["macro_f1"]["sample_std"] is None
    report = aggregate_seeds([summary(.8, .75), summary(.82, .75)])
    gain = report["groups"]["learned"]["original"]["macro_f1_gain_vs_A"]
    assert gain["mean"] == pytest.approx(.06)
    assert gain["sample_std"] == pytest.approx(np.sqrt(.0002))
