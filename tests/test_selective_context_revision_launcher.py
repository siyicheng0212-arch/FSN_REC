"""Formal launcher checks: fake workers never claim training or CUDA success."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import threading

import pytest

from experiments.selective_context import revision_run_suite as launcher
from experiments.selective_context.revision_config import VARIANTS


def args(tmp_path, gpus=(0, 1)):
    return SimpleNamespace(output=str(tmp_path / "private-new"), python=sys.executable,
        feature_index="/private/frozen-A/index.jsonl", source_protocol_dir="/private/protocol",
        legacy_config="/private/old-TCN.json", legacy_checkpoint="/private/old-TCN.pt",
        historical_predictions="/private/old-val823.jsonl", historical_column_map=None,
        historical_logits_key=None, chain_layout="eligible_segments", config="/private/revision.json",
        gpus=list(gpus), expected_commit=None, execute=False)


def plan(tmp_path, gpus=(0, 1)):
    argument = args(tmp_path, gpus)
    config = {"seeds": [42]}
    seal = {"fingerprint": {"val_clips": 823}, "legacy_checkpoint_sha256": "old-checkpoint",
            "historical_column_map_sha256": None, "chain_layout": "eligible_segments"}
    return {"arguments": vars(argument), "config": config,
        "repository": {"commit": "fixture-commit", "clean": True},
        "commands": launcher.build_commands(argument, config), "waves": launcher.wave_schedule(argument.gpus),
        "smoke_seal": seal, "GPU_environment": {"physical_gpu_uuid": {
            0: "GPU-unit-0", 1: "GPU-unit-1", 2: "GPU-unit-2", 3: "GPU-unit-3"}},
        "evaluation_role": "val823_development_not_independent_test"}


@pytest.mark.parametrize("raw, expected", [("0,1", [0, 1]), ("3,1,0,2", [3, 1, 0, 2])])
def test_requires_two_or_four_distinct_physical_gpus(raw, expected):
    assert launcher.gpu_indices(raw) == expected


@pytest.mark.parametrize("raw", ["0", "0,1,2", "0,0", "-1,2", "cuda:0", "0,1,2,2", ""])
def test_rejects_invalid_gpu_indices(raw):
    with pytest.raises(Exception):
        launcher.gpu_indices(raw)


def test_two_card_and_four_card_waves_are_independent():
    assert launcher.wave_schedule([3, 1]) == [
        [{"variant": "scalar", "physical_gpu": 3},
         {"variant": "class_conditioned", "physical_gpu": 1}],
        [{"variant": "logits_only", "physical_gpu": 3},
         {"variant": "visual_logits", "physical_gpu": 1}],
    ]
    assert len(launcher.wave_schedule([0, 1, 2, 3])) == 1
    assert [job["variant"] for job in launcher.wave_schedule([0, 1, 2, 3])[0]] == list(VARIANTS)


def test_commands_use_existing_old_checkpoint_and_strict_cuda_zero(tmp_path):
    commands = launcher.build_commands(args(tmp_path, (3, 1)), {"seeds": [42]})
    assert set(commands) == {"unit_tests", "cuda_smoke", *(f"{name}_42" for name in VARIANTS)}
    for stage in commands:
        assert "--execute" not in commands[stage]
        if stage == "unit_tests":
            assert "-p" in commands[stage] and "test_selective_context_revision_launcher.py" in " ".join(commands[stage])
            continue
        assert commands[stage][commands[stage].index("--device") + 1] == "cuda:0"
        assert commands[stage][commands[stage].index("--legacy-checkpoint") + 1] == "/private/old-TCN.pt"
        assert "--relation-checkpoint" not in commands[stage]
    assert "--smoke-report" in commands["scalar_42"]


def test_each_child_sees_one_uuid_without_changing_inherited_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-inherited")
    checked = plan(tmp_path, (3, 1))
    assert launcher.worker_environment(checked, 3)["CUDA_VISIBLE_DEVICES"] == "GPU-unit-3"
    assert launcher.worker_environment(checked, None)["CUDA_VISIBLE_DEVICES"] == ""
    assert launcher.worker_environment(checked, 1)["CUDA_VISIBLE_DEVICES"] == "GPU-unit-1"
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent-inherited"
    with pytest.raises(RuntimeError, match="mapping"):
        launcher.worker_environment(checked, 9)


def test_failed_smoke_never_starts_a_training_job_and_never_reuses_output(tmp_path):
    checked = plan(tmp_path)
    visited = []
    def fake_worker(stage, command):
        visited.append(stage)
        return 7 if stage == "cuda_smoke" else 0
    with pytest.raises(RuntimeError, match="cuda_smoke failed with exit 7"):
        launcher.execute_plan(checked, run=fake_worker)
    assert visited == ["unit_tests", "cuda_smoke"]
    result = json.loads((Path(checked["arguments"]["output"]) / "suite_result.json").read_text())
    assert result["status"] == "failed" and result["training_completed"] is False
    with pytest.raises(FileExistsError):
        launcher.execute_plan(checked, run=fake_worker)


def test_failed_first_wave_waits_for_other_child_but_blocks_second_wave(tmp_path):
    checked = plan(tmp_path)
    other_finished = threading.Event()
    def fake_worker(stage, command):
        if stage == "scalar_42":
            assert other_finished.wait(3)
            return 5
        if stage == "class_conditioned_42":
            other_finished.set()
        return 0
    with pytest.raises(RuntimeError, match="scalar_42 failed with exit 5"):
        launcher.execute_plan(checked, run=fake_worker)
    result = json.loads((Path(checked["arguments"]["output"]) / "suite_result.json").read_text())
    assert result["exit_codes"]["class_conditioned_42"] == 0
    assert all(key not in result["exit_codes"] for key in ("logits_only_42", "visual_logits_42"))


def test_fake_workers_order_two_waves_but_never_report_real_training(tmp_path):
    checked = plan(tmp_path)
    completed = set()
    lock = threading.Lock()
    def fake_worker(stage, command):
        with lock:
            if stage in ("logits_only_42", "visual_logits_42"):
                assert {"scalar_42", "class_conditioned_42"} <= completed
            completed.add(stage)
        return 0
    launcher.execute_plan(checked, run=fake_worker)
    report = json.loads((Path(checked["arguments"]["output"]) / "suite_result.json").read_text())
    assert report["status"] == "complete"
    assert report["training_completed"] is False and report["test_injection_only"] is True


def test_cuda_smoke_requires_exact_existing_seals_and_real_flag(tmp_path):
    checked = plan(tmp_path)
    success = {**checked["smoke_seal"], "all_passed": True,
               "real_cuda_tcn_revision_smoke_completed": True}
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(success))
    launcher._verify_smoke(checked, path)
    for key in success:
        changed = deepcopy(success)
        changed.pop(key)
        path.write_text(json.dumps(changed))
        with pytest.raises(RuntimeError, match="real CUDA"):
            launcher._verify_smoke(checked, path)


def test_all_four_parseable_results_and_checkpoints_are_required(tmp_path):
    baseline = {"count": 823, "macro_f1": .79, "accuracy": .81,
                "confusion": [[0] * 7 for _ in range(7)]}
    for variant in VARIANTS:
        directory = tmp_path / "seed_42" / variant
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps({
            "counts": {"clips": 823},
            "metrics": {"A": baseline, "old_TCN": baseline,
                        "new": {"count": 823, "macro_f1": .8, "accuracy": .82}}}))
        (directory / "history.json").write_text(json.dumps([{"epoch": 1}]))
        (directory / "best.pt").write_bytes(b"unit-test-only")
    assert set(launcher._verify_group_results(tmp_path, 42)) == set(VARIANTS)
    (tmp_path / "seed_42" / "visual_logits" / "result.json").write_text("not json")
    with pytest.raises(json.JSONDecodeError):
        launcher._verify_group_results(tmp_path, 42)
    (tmp_path / "seed_42" / "visual_logits" / "result.json").write_text(json.dumps({
        "counts": {"clips": 822}, "metrics": {"A": baseline, "old_TCN": baseline,
            "new": {"count": 822, "macro_f1": .8, "accuracy": .82}}}))
    with pytest.raises(RuntimeError, match="823"):
        launcher._verify_group_results(tmp_path, 42)
    changed_old = {**baseline, "macro_f1": .78}
    (tmp_path / "seed_42" / "visual_logits" / "result.json").write_text(json.dumps({
        "counts": {"clips": 823}, "metrics": {"A": baseline, "old_TCN": changed_old,
            "new": {"count": 823, "macro_f1": .8, "accuracy": .82}}}))
    with pytest.raises(RuntimeError, match="matched arms"):
        launcher._verify_group_results(tmp_path, 42)


def test_read_only_plan_does_not_create_output_or_probe_gpu(tmp_path, monkeypatch):
    sample = args(tmp_path)
    for name in ("feature_index", "legacy_config", "legacy_checkpoint", "historical_predictions", "config"):
        path = tmp_path / f"{name}.json"
        path.write_text("fixture")
        setattr(sample, name, str(path))
    sample.source_protocol_dir = str(tmp_path)
    bundle = SimpleNamespace(fingerprint={"features": "fixture"})
    audit = {"legacy_audit": {"path": str(tmp_path / "audit.json"), "sha256": "seal"}}
    prepared = {"bundle": bundle, "base_audit": audit, "parity": {"clips": 823}}
    fake_legacy = {"baseline": {"kind": "legacy", "legacy_audit": str(tmp_path / "audit.json"),
                                 "source_dependencies": [str(tmp_path / "old_source.py")]}}
    monkeypatch.setattr(launcher, "load_config", lambda *a: {"seeds": [42]})
    monkeypatch.setattr(launcher, "load_legacy_config", lambda *a: fake_legacy)
    monkeypatch.setattr(launcher, "baseline_source_hashes", lambda *a: {"old_source.py": "hash"})
    monkeypatch.setattr(launcher, "source_hashes", lambda: {"revision.py": "hash"})
    monkeypatch.setattr(launcher, "repository_state", lambda: {"commit": "fixture", "clean": True})
    def forbid_gpu(*a, **kw):
        raise AssertionError("no GPU inventory in read-only plan")
    monkeypatch.setattr(launcher, "assert_idle_gpus", forbid_gpu)
    from experiments.selective_context import revision_train
    monkeypatch.setattr(revision_train, "prepare_revision_inputs", lambda *a, **kw: prepared)
    def fixture_seal(args, prepared):
        from experiments.relation.data import sha256_file
        return {"fingerprint": bundle.fingerprint, "source_sha256": {"revision.py": "hash"},
            "external_baseline_sha256": {"old_source.py": "hash"}, "legacy_audit_sha256": "seal",
            "config_sha256": sha256_file(args.config),
            "legacy_config_sha256": sha256_file(args.legacy_config),
            "legacy_checkpoint_sha256": sha256_file(args.legacy_checkpoint),
            "historical_predictions_sha256": sha256_file(args.historical_predictions),
            "historical_column_map_sha256": None, "chain_layout": args.chain_layout}
    monkeypatch.setattr(revision_train, "revision_smoke_seal", fixture_seal)
    index_protocol = Path(sample.feature_index).parent / "protocol.json"
    index_protocol.write_text("fixture export")
    result = launcher.prepare_plan(sample)
    assert result["historical_parity"]["clips"] == 823
    assert not Path(sample.output).exists()
    assert len(result["waves"]) == 2
    sample.expected_commit = "different"
    with pytest.raises(ValueError, match="fixed commit"):
        launcher.prepare_plan(sample)
