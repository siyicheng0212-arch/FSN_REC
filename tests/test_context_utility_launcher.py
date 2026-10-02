"""GPU mapping and failure isolation without a GPU or real training claim."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import sys

import pytest

from experiments.context_utility import run_suite
from experiments.context_utility import config as config_module
from experiments.fsn_tcn.model import TemporalResidualTCN, TCNConfig
from experiments.relation.data import sha256_file


@pytest.fixture
def planning_fixture(tmp_path, monkeypatch):
    argument = args(tmp_path, [0, 1])
    argument.python = sys.executable
    protocol = tmp_path / "sealed-protocol"
    protocol.mkdir()
    for name in ("audit.json", "metadata.jsonl"):
        (protocol / name).write_text("{}\n")
    index, checkpoint = tmp_path / "index.jsonl", tmp_path / "base.pt"
    index.write_text("test fixture index\n")
    checkpoint.write_bytes(b"checkpoint-loading-is-mocked-in-this-read-only-test")
    cfg = json.loads((run_suite.ROOT / "experiments/context_utility/reference.json").read_text())
    cfg["baseline"]["config"].update(width=4, layers=2, dropout=0.)
    cfg["baseline"]["temporal_paths"] = ["blocks.0.temporal", "blocks.1.temporal"]
    cfg["module"]["dim"] = 2
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    argument.feature_index, argument.base_checkpoint = str(index), str(checkpoint)
    argument.source_protocol_dir, argument.config = str(protocol), str(config)
    import numpy as np
    index_object = SimpleNamespace(read=lambda key: {"global_tokens": np.ones((8, 2)), "local_tokens": np.ones((12, 3))})
    fingerprint = {"feature_index_sha256": sha256_file(index), "source_audit_sha256": sha256_file(protocol / "audit.json"),
        "source_files_sha256": {"metadata.jsonl": sha256_file(protocol / "metadata.jsonl")}}
    bundle = SimpleNamespace(metadata={"fixture-clip": {}}, index=index_object, fingerprint=fingerprint, chains={"val": []})
    monkeypatch.setattr(run_suite, "load_bundle", lambda *a: bundle)
    monkeypatch.setattr(config_module, "build_base", lambda config, input_dim, checkpoint, **kw:
        (TemporalResidualTCN(TCNConfig(input_dim=input_dim, **config["baseline"]["config"])), {"legacy_audit": None}))
    monkeypatch.setattr(run_suite, "build_epoch_plan", lambda *a, **kw:
        ([], {"actual_replacements": 3, "epoch": kw["epoch"]}))
    monkeypatch.setattr(run_suite, "build_challenge", lambda *a, **kw: ([], {"fixture_only": True}))
    monkeypatch.setattr(run_suite, "repository_state", lambda: {"commit": "fixed-fixture", "clean": True})
    return argument


def args(tmp_path, gpus):
    return SimpleNamespace(output=str(tmp_path / "new-private-suite"), python="/usr/bin/python",
        feature_index="/private/features/index.jsonl", source_protocol_dir="/private/protocol",
        config="/private/config.json", base_checkpoint="/private/TCN_best.pt", gpus=gpus,
        expected_commit=None, execute=False)


def plan(tmp_path, gpus):
    arguments = args(tmp_path, gpus)
    config = {"seeds": [42]}
    return {"arguments": vars(arguments), "config": config,
        "repository": {"commit": "unit-test-commit", "clean": True},
        "commands": run_suite.build_commands(arguments, config), "waves": run_suite.wave_schedule(gpus),
        "GPU_environment": {"physical_gpu_uuid": {0: "GPU-unit-0", 1: "GPU-unit-1", 2: "GPU-unit-2", 3: "GPU-unit-3"}}}


@pytest.mark.parametrize("raw,expected", [("0,1", [0, 1]), ("3,1,0,2", [3, 1, 0, 2])])
def test_two_or_four_physical_gpu_indices(raw, expected):
    assert run_suite.gpu_indices(raw) == expected


@pytest.mark.parametrize("raw", ["0", "0,1,2", "0,0", "-1,1", "cuda:0", "0,1,2,2", ""])
def test_reject_ambiguous_gpu_mapping(raw):
    with pytest.raises(Exception):
        run_suite.gpu_indices(raw)


def test_two_waves_assign_each_independent_job_one_physical_gpu():
    assert run_suite.wave_schedule([3, 1]) == [
        [{"variant": "continue", "physical_gpu": 3}, {"variant": "aug", "physical_gpu": 1}],
        [{"variant": "scalar_aug", "physical_gpu": 3}, {"variant": "dynamic_aug", "physical_gpu": 1}]]
    assert len(run_suite.wave_schedule([0, 1, 2, 3])) == 1


def test_commands_use_visible_cuda_zero_and_the_old_base_checkpoint(tmp_path):
    commands = run_suite.build_commands(args(tmp_path, [3, 1]), {"seeds": [42]})
    for stage in ("cuda_smoke", "continue_42", "aug_42", "scalar_aug_42", "dynamic_aug_42", "evaluate_42"):
        command = commands[stage]
        assert command[command.index("--device") + 1] == "cuda:0"
        assert command[command.index("--base-checkpoint") + 1] == "/private/TCN_best.pt"
        assert "--execute" not in command
    assert "--smoke-report" in commands["dynamic_aug_42"]
    assert all(value.endswith(".py") for value in commands["unit_tests"][6:])


def test_worker_environment_isolates_one_uuid_without_mutating_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-setting")
    mapping = plan(tmp_path, [3, 1])
    assert run_suite.worker_environment(mapping, 3)["CUDA_VISIBLE_DEVICES"] == "GPU-unit-3"
    mapping["GPU_environment"]["physical_gpu_uuid"] = {"1": "GPU-unit-1"}
    assert run_suite.worker_environment(mapping, 1)["CUDA_VISIBLE_DEVICES"] == "GPU-unit-1"
    assert run_suite.worker_environment(mapping, None)["CUDA_VISIBLE_DEVICES"] == ""
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent-setting"
    with pytest.raises(RuntimeError, match="UUID"):
        run_suite.worker_environment(mapping, 2)


def test_precheck_failure_never_starts_training_or_reuses_output(tmp_path):
    mapping = plan(tmp_path, [0, 1])
    visited = []
    def fail(stage, command):
        visited.append(stage)
        return 7 if stage == "cuda_smoke" else 0
    with pytest.raises(RuntimeError, match="exit 7"):
        run_suite.execute_plan(mapping, run=fail)
    assert visited == ["unit_tests", "cuda_smoke"]
    saved = json.loads((Path(mapping["arguments"]["output"]) / "suite_result.json").read_text())
    assert saved["training_completed"] is False and saved["status"] == "failed"
    with pytest.raises(FileExistsError):
        run_suite.execute_plan(mapping, run=fail)


def test_failed_wave_finishes_its_other_job_and_blocks_next_wave(tmp_path):
    mapping = plan(tmp_path, [0, 1])
    failed = threading.Event()
    completed = []
    def run(stage, command):
        if stage == "continue_42":
            failed.set()
            return 9
        if stage == "aug_42":
            assert failed.wait(5)
            completed.append(stage)
        return 0
    with pytest.raises(RuntimeError, match="exit 9"):
        run_suite.execute_plan(mapping, run=run)
    assert completed == ["aug_42"]
    saved = json.loads((Path(mapping["arguments"]["output"]) / "suite_result.json").read_text())
    assert saved["exit_codes"]["aug_42"] == 0
    assert not any(stage.startswith(("scalar_aug", "dynamic_aug", "evaluate")) for stage in saved["exit_codes"])


def test_two_card_success_orders_waves_before_evaluation_without_claiming_cuda(tmp_path):
    mapping = plan(tmp_path, [0, 1])
    completed = []
    lock = threading.Lock()
    def run(stage, command):
        with lock:
            if stage in {"scalar_aug_42", "dynamic_aug_42"}:
                assert {"continue_42", "aug_42"} <= set(completed)
            if stage == "evaluate_42":
                assert {f"{variant}_42" for variant in run_suite.VARIANTS} <= set(completed)
            completed.append(stage)
        return 0
    run_suite.execute_plan(mapping, run=run)
    saved = json.loads((Path(mapping["arguments"]["output"]) / "suite_result.json").read_text())
    assert saved["status"] == "complete" and saved["test_injection_only"] is True
    assert saved["training_completed"] is False


def test_smoke_report_requires_cuda_and_exact_all_seals(tmp_path):
    mapping = {"fingerprint": {"count": 823}, "config_sha256": "a", "base_checkpoint_sha256": "b",
        "source_sha256": {"model": "c"}, "external_baseline_sha256": {"old": "d"}, "legacy_audit_sha256": "e"}
    report = {**mapping, "all_passed": True, "real_cuda_context_utility_smoke_completed": True}
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(report))
    run_suite._verify_smoke(mapping, path)
    for key in report:
        bad = deepcopy(report)
        bad.pop(key)
        path.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError, match="CUDA smoke"):
            run_suite._verify_smoke(mapping, path)


def test_no_evaluation_until_all_four_results_exist_and_parse(tmp_path):
    for variant in run_suite.VARIANTS:
        directory = tmp_path / "seed_42" / variant
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps({"metrics": {"count": 823, "macro_f1": .79, "accuracy": .81},
                                                          "counts": {"clips": 823}}))
        (directory / "best.pt").write_bytes(b"test-file-only")
        (directory / "history.json").write_text(json.dumps([{"epoch": 0}]))
    run_suite._verify_group_results(tmp_path, 42)
    path = tmp_path / "seed_42" / "dynamic_aug" / "result.json"
    path.write_text("not-json")
    with pytest.raises(json.JSONDecodeError):
        run_suite._verify_group_results(tmp_path, 42)
    path.write_text(json.dumps({"status": "failed"}))
    with pytest.raises(RuntimeError, match="valid completed"):
        run_suite._verify_group_results(tmp_path, 42)


def test_read_only_plan_does_not_probe_gpu_or_create_output(planning_fixture, monkeypatch):
    def prohibited(*a, **kw):
        raise AssertionError("GPU probing is forbidden in plan mode")
    monkeypatch.setattr(run_suite, "assert_idle_gpus", prohibited)
    prepared = run_suite.prepare_plan(planning_fixture)
    assert not Path(planning_fixture.output).exists()
    assert prepared["first_epoch_plan_audits"]["42"]["epoch"] == 1
    assert prepared["parameter_counts"]["scalar_aug"]["added"] == 1
    assert prepared["parameter_counts"]["dynamic_aug"]["added"] > 0
    assert prepared["training_completed"] is False


def test_plan_rejects_wrong_fixed_commit_before_execution(planning_fixture):
    planning_fixture.expected_commit = "different-commit"
    with pytest.raises(ValueError, match="fixed commit"):
        run_suite.prepare_plan(planning_fixture)
    assert not Path(planning_fixture.output).exists()


def test_runtime_detects_checkpoint_or_source_chain_drift(planning_fixture):
    prepared = run_suite.prepare_plan(planning_fixture)
    run_suite.validate_runtime(prepared)
    checkpoint = Path(planning_fixture.base_checkpoint)
    contents = checkpoint.read_bytes()
    checkpoint.write_bytes(contents + b"changed")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        run_suite.validate_runtime(prepared)
    checkpoint.write_bytes(contents)
    metadata = Path(planning_fixture.source_protocol_dir) / "metadata.jsonl"
    metadata.write_text("changed\n")
    with pytest.raises(RuntimeError, match="metadata/chains/edges changed"):
        run_suite.validate_runtime(prepared)


def test_plan_rejects_zero_replacement_control(planning_fixture, monkeypatch):
    monkeypatch.setattr(run_suite, "build_epoch_plan", lambda *a, **kw: ([], {"actual_replacements": 0}))
    with pytest.raises(ValueError, match="no train-only"):
        run_suite.prepare_plan(planning_fixture)
    assert not Path(planning_fixture.output).exists()
