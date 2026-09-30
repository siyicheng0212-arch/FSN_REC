"""CPU correctness checks, including real frozen manifests and 36f224 caches."""
from pathlib import Path
import copy
import json
from unittest.mock import patch

import numpy as np
import tempfile
import unittest
import torch
from torch import nn

from cvm.models import ClinicalHierarchyModel, compute_loss
from cvm.protocol import audit_protocol, load_protocol
from cvm.taxonomy import CLASS_NAMES, clinical_taxonomy
from cvm.train import (CachedProtocolDataset, StopRequest, make_scaler, parser,
                       run_train, sha256_file, train_epoch, evaluate_model)
from cvm.run_suite import (FORMAL_SEEDS, CORE_BACKBONES, declared_configurations, declared_contrasts,
                          execute_plan, trainer_command, generate_plan, parser as suite_parser)
from experiments.full_data import cache_paths, request_digest


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)

    def forward(self, video):
        return self.linear(video.mean(dim=(2, 3, 4)))


def raises(exception, match=None):
    case = unittest.TestCase()
    return case.assertRaisesRegex(exception, match) if match else case.assertRaises(exception)


def toy_model(mode="flat", taxonomy=None):
    torch.manual_seed(123)
    return ClinicalHierarchyModel(TinyBackbone(), 4, mode=mode, taxonomy=taxonomy or clinical_taxonomy(),
                                   load_report={"pretrained": True, "source": "explicit test injection"})


def tiny_batches(sizes):
    generator = torch.Generator().manual_seed(19)
    return [{"video": torch.randn(size, 3, 2, 3, 3, generator=generator),
             "label": torch.arange(offset, offset + size) % 7}
            for size, offset in zip(sizes, np.cumsum([0] + list(sizes[:-1])))]


def test_partial_accumulation_matches_actual_effective_objective(weighted, mode):
    torch.set_num_threads(1)
    model = toy_model(mode)
    reference = copy.deepcopy(model)
    batches = tiny_batches([2, 1, 2])
    weights = torch.tensor([1., 7., 2., 4., 3., 5., 9.]) if weighted else None
    loss = lambda out, target: compute_loss(out, target, mode, model.taxonomy, class_weights=weights)
    optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=.03)
    ref_optimizer = torch.optim.SGD([parameter for parameter in reference.parameters() if parameter.requires_grad], lr=.03)
    result = train_epoch(model, batches, optimizer, loss, lambda frames, **_: frames,
                         torch.device("cpu"), accum_steps=2, clip_grad=0.,
                         loss_normalizer=(lambda target: weights[target].sum()) if weighted else None)
    for window in (batches[:2], batches[2:]):
        inputs = torch.cat([batch["video"] for batch in window])
        labels = torch.cat([batch["label"] for batch in window])
        ref_optimizer.zero_grad(set_to_none=True)
        compute_loss(reference(inputs), labels, mode, reference.taxonomy, class_weights=weights).backward()
        ref_optimizer.step()
    assert result["samples"] == 5
    assert result["optimizer_steps"] == 2
    for actual, expected in zip(model.parameters(), reference.parameters()):
        assert torch.allclose(actual, expected, atol=2e-7, rtol=1e-6)


def test_disabled_scaler_and_nonfinite_loss_stop():
    model = toy_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    assert not make_scaler(torch.device("cpu"), "none").is_enabled()
    before = copy.deepcopy(model.state_dict())
    with raises(FloatingPointError):
        train_epoch(model, tiny_batches([2]), optimizer,
                    lambda output, _: output["flat_logits"].sum() * torch.tensor(float("nan")),
                    lambda frames, **_: frames, torch.device("cpu"))
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def real_cached_protocol(tmp_path):
    torch.set_num_threads(1)
    manifests = {}
    for split in ("train", "val"):
        path = tmp_path / (split + ".jsonl")
        rows = [{"clip_id": f"clip-{split}-{label}", "split": split, "label_id": label,
                 "normalized_label": CLASS_NAMES[label], "source_collection": "private-hospital-location",
                 "group_id": f"group-{split}-{label}", "video_path": str(tmp_path / f"{split}-{label}.mp4"),
                 "clip_start_sec": 0., "clip_end_sec": 1., "clip_duration_sec": 1.,
                 "cache_split": "train" if split == "val" else split}
                for label in range(7)]
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        manifests[split] = path
    protocol_path = audit_protocol(manifests["train"], manifests["val"], tmp_path / "protocol")
    protocol = load_protocol(protocol_path)
    cache_root = tmp_path / "cache"
    for split in ("train", "val"):
        for record in protocol.records[split]:
            routed = protocol.cache_record(record)
            array_path, metadata_path = cache_paths(cache_root, routed)
            array_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(array_path, np.full((36, 224, 224, 3), 20 + record.label_id * 20, dtype=np.uint8))
            metadata_path.write_text(json.dumps({"request_digest": request_digest(routed, 36, 224),
                                                 "pixel_repeat_fraction": 35 / 36}), encoding="utf-8")
    fake_checkpoint = tmp_path / "injected-test.pth"
    fake_checkpoint.write_bytes(b"test-only dependency injection; not used by the real pretrained loader")
    return protocol_path, cache_root, fake_checkpoint


def test_one_epoch_real_cache_protocol_and_outputs(real_cached_protocol, tmp_path, mode):
    protocol_path, cache_root, checkpoint = real_cached_protocol
    output = tmp_path / ("run-" + mode)
    args = parser().parse_args(["train", "--protocol", str(protocol_path), "--cache-root", str(cache_root),
                                "--output", str(output), "--checkpoint", str(checkpoint),
                                "--backbone", "r2plus1d_18", "--mode", mode,
                                "--main-decoder", "soft" if mode == "hierarchy" else "flat",
                                "--warmup-epochs", "0", "--epochs", "1", "--workers", "0",
                                "--batch-size", "2", "--accum-steps", "2", "--amp", "none",
                                "--augmentation", "none"])
    def inject(**kwargs):
        assert kwargs["weights"] == "DEFAULT"  # production code never requested random initialization
        assert kwargs["weights_path"] == checkpoint
        return toy_model(kwargs["mode"], kwargs["taxonomy"])
    with patch("cvm.train.require_device", return_value=torch.device("cpu")), patch("cvm.models.build_model", side_effect=inject):
        run_train(args)
    result = json.loads((output / "result.json").read_text())
    history = json.loads((output / "history.json").read_text())
    config = json.loads((output / "config.json").read_text())
    assert result["training_completed"] is True and result["test_evaluated"] is False
    assert result["best_stage"] == "finetune" and result["best_epoch"] == 1
    assert len(history) == 1 and history[0]["train"]["samples"] == 7
    assert config["preprocessing"]["frames"] == 16
    assert config["checkpoint_sha256"] == sha256_file(checkpoint)
    for filename in ("best.pt", "last.pt", "load_report.json", "protocol_summary.json", "prediction_metadata.json", "val_report.json"):
        assert (output / filename).is_file()
    rows = [json.loads(line) for line in (output / "val_predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 7 and rows[0]["group_id"] == "group-val-0"
    assert rows[0]["repeated_frame_fraction"] == 15 / 16
    assert history[0]["train"]["first_step_gradient_norms"]["backbone"] > 0
    assert (rows[0]["group_logits"] is None) == (mode in ("flat", "capacity_control"))
    assert (rows[0]["flat_logits"] is None) == (mode == "hierarchy")
    assert "private-hospital-location" not in (output / "val_report.json").read_text()
    with patch("cvm.train.require_device", return_value=torch.device("cpu")), patch("cvm.models.build_model", side_effect=inject), raises(FileExistsError):
        run_train(args)


def test_missing_cache_fails_without_decoder_or_rebuild(real_cached_protocol):
    protocol_path, cache_root, _ = real_cached_protocol
    protocol = load_protocol(protocol_path)
    record = protocol.cache_record(protocol.records["val"][0])
    cache_paths(cache_root, record)[0].unlink()
    with patch("experiments.full_data.extract_clip", side_effect=AssertionError("must never decode")), raises(RuntimeError, match="invalid/missing"):
        CachedProtocolDataset(protocol, "val", cache_root)


def test_stop_request_does_not_claim_epoch_complete():
    stop = StopRequest()
    stop.handle(2, None)
    model = toy_model()
    result = train_epoch(model, tiny_batches([2]), torch.optim.SGD(model.parameters(), lr=.01),
                         lambda out, target: compute_loss(out, target, "flat", model.taxonomy),
                         lambda frames, **_: frames, torch.device("cpu"), stop=stop)
    assert result["interrupted"] is True and result["samples"] == 0
    assert result["optimizer_steps"] == 0


def test_suite_fixed_matrix_dry_run_no_launch(tmp_path):
    assert len(declared_configurations("pilot")) == 4
    assert len(declared_configurations("formal")) == 13
    assert FORMAL_SEEDS == (42, 2026, 2027)
    plan = {"schema_version": "fsn-cvm-plan-1", "stage": "pilot", "jobs": [{}] * 4}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with patch("subprocess.Popen", side_effect=AssertionError("dry run must not launch")):
        execute_plan(path, python="python", execute=False)
    assert not (tmp_path / ".launch_once").exists()


def test_generated_command_has_no_test_or_resume():
    plan = {"protocol": "/protocol.json", "cache_root": "/cache", "protocol_sha256": "a" * 64,
            "code_hash": "b" * 64, "settings": {"epochs": 60, "batch_size": 2, "accum_steps": 16}}
    job = {"output": "/new", "backbone": "r2plus1d_18", "mode": "hierarchy", "seed": 42,
           "checkpoint": "/official.pth", "main_decoder": "soft", "taxonomy": "clinical"}
    command = trainer_command(plan, job, "python")
    assert "--expected-protocol-sha256" in command
    assert "--resume" not in command and "--allow-test" not in command and "--allow-download" not in command


class TrainingTests(unittest.TestCase):
    def test_disabled_scaler_nonfinite(self):
        test_disabled_scaler_and_nonfinite_loss_stop()

    def test_stop_request(self):
        test_stop_request_does_not_claim_epoch_complete()

    def test_command_scope(self):
        test_generated_command_has_no_test_or_resume()

    def test_plan_dry_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            test_suite_fixed_matrix_dry_run_no_launch(Path(temporary))

    def test_missing_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            test_missing_cache_fails_without_decoder_or_rebuild(real_cached_protocol(Path(temporary)))

    def test_native_selected16_set_is_preserved_by_shuffle_probe(self):
        model = toy_model()
        captured = []
        original = model.forward
        def capture(inputs):
            captured.append(inputs.clone())
            return original(inputs)
        model.forward = capture
        frames = torch.arange(36).float().view(1, 36, 1, 1, 1).expand(1, 36, 3, 2, 2)
        batch = {"video": frames, "label": torch.tensor([0]), "clip_id": ["clip-probe"],
                 "source": ["private"], "duration": torch.tensor([1.])}
        def select(video, **_):
            indices = torch.linspace(0, 35, 16).round().long()
            return video[:, indices].permute(0, 2, 1, 3, 4)
        for probe in ("none", "shuffle", "static"):
            evaluate_model(model, [batch], select, torch.device("cpu"), model.taxonomy, "flat", "flat", probe=probe)
        original_values, shuffled_values, static_values = [inputs[0, 0, :, 0, 0] for inputs in captured]
        self.assertTrue(torch.equal(original_values.sort().values, shuffled_values.sort().values))
        self.assertEqual(static_values.unique().numel(), 1)
        self.assertIn(float(static_values[0]), original_values.tolist())

    def test_warmup_high_score_cannot_select_full_finetune_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            protocol, cache_root, checkpoint = real_cached_protocol(path)
            output = path / "warmup-selection"
            args = parser().parse_args(["train", "--protocol", str(protocol), "--cache-root", str(cache_root),
                                       "--checkpoint", str(checkpoint), "--output", str(output),
                                       "--backbone", "r2plus1d_18", "--mode", "flat", "--main-decoder", "flat",
                                       "--warmup-epochs", "1", "--epochs", "1", "--workers", "0", "--amp", "none"])
            calls = [0]
            def evaluate(*arguments, **kwargs):
                metrics, rows, seconds = evaluate_model(*arguments, **kwargs)
                metrics["all"]["macro_f1"] = .99 if calls[0] == 0 else .1
                calls[0] += 1
                return metrics, rows, seconds
            with patch("cvm.train.require_device", return_value=torch.device("cpu")), patch("cvm.models.build_model", side_effect=lambda **kwargs: toy_model(kwargs["mode"], kwargs["taxonomy"])), patch("cvm.train.evaluate_model", side_effect=evaluate):
                run_train(args)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["best_epoch"], 2)
            self.assertEqual(result["best_val_macro_f1"], .1)
            self.assertEqual(result["best_stage"], "finetune")

    def test_interrupted_run_keeps_last_and_never_claims_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            protocol, cache_root, checkpoint = real_cached_protocol(path)
            output = path / "interrupted"
            args = parser().parse_args(["train", "--protocol", str(protocol), "--cache-root", str(cache_root),
                                       "--checkpoint", str(checkpoint), "--output", str(output),
                                       "--backbone", "r2plus1d_18", "--mode", "flat", "--main-decoder", "flat",
                                       "--warmup-epochs", "0", "--epochs", "1", "--workers", "0", "--amp", "none"])
            def interrupt(*arguments, **kwargs):
                report = train_epoch(*arguments, **kwargs)
                kwargs["stop"].handle(2, None)
                report["interrupted"] = True
                return report
            with patch("cvm.train.require_device", return_value=torch.device("cpu")), patch("cvm.models.build_model", side_effect=lambda **kwargs: toy_model()), patch("cvm.train.train_epoch", side_effect=interrupt), self.assertRaises(SystemExit) as exit_context:
                run_train(args)
            self.assertEqual(exit_context.exception.code, 130)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["status"], "interrupted")
            self.assertFalse(result["training_completed"])
            self.assertIsNone(result["best_epoch"])
            self.assertTrue(torch.load(output / "last.pt", weights_only=False)["interrupted"])
            self.assertFalse((output / "best.pt").exists())

    def test_plan_generation_freezes_seed42_without_launching(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            protocol, cache_root, checkpoint = real_cached_protocol(path)
            weights = path / "weights.json"
            weights.write_text(json.dumps({"r2plus1d_18": str(checkpoint)}))
            args = suite_parser().parse_args(["plan", "--phase", "pilot", "--protocol", str(protocol),
                                              "--cache-root", str(cache_root), "--output", str(path / "plan"),
                                              "--checkpoints", str(weights)])
            with patch("subprocess.check_output", return_value="test-git-commit\n"), patch("subprocess.Popen", side_effect=AssertionError("plan cannot launch")):
                plan_path = generate_plan(args)
            plan = json.loads(plan_path.read_text())
            self.assertEqual(len(plan["jobs"]), 4)
            self.assertEqual({job["seed"] for job in plan["jobs"]}, {42})
            self.assertEqual(plan["settings"]["epochs"], 60)
            self.assertEqual(plan["settings"]["class_weights"], "none")
            self.assertTrue(all(job["checkpoint_sha256"] == sha256_file(checkpoint) for job in plan["jobs"]))
            self.assertFalse((plan_path.parent / ".launch_once").exists())

    def test_full_matrix_is_deduplicated_and_one_factor_sensitivities_are_declared(self):
        expected = {"smoke": 4, "pilot": 4, "benchmark": 8, "ablation": 7,
                    "sensitivity": 6, "representation": 4, "formal": 13, "extended": 19}
        by_stage = {}
        for stage, count in expected.items():
            configurations = declared_configurations(stage)
            identifiers = {configuration["configuration_id"] for configuration in configurations}
            self.assertEqual(len(configurations), count)
            self.assertEqual(len(identifiers), count)
            by_stage[stage] = identifiers
            for contrast in declared_contrasts(configurations):
                self.assertIn(contrast["baseline_configuration_id"], identifiers)
                self.assertIn(contrast["candidate_configuration_id"], identifiers)
        self.assertEqual(by_stage["formal"], by_stage["benchmark"] | by_stage["ablation"])
        self.assertEqual(by_stage["extended"], by_stage["formal"] | by_stage["sensitivity"] | by_stage["representation"])
        self.assertEqual(set(CORE_BACKBONES), {"r2plus1d_18", "mvit_v2_s", "videomamba_tiny16", "videomae_base16"})
        sensitivities = declared_configurations("sensitivity")
        self.assertNotIn((32, "sqrt_inverse"), {(c["overrides"]["frames"], c["overrides"]["class_weights"]) for c in sensitivities})
        contrasts = declared_contrasts(sensitivities)
        for contrast in contrasts:
            if contrast["comparison_kind"] == "single_factor":
                self.assertEqual(len(contrast["varying_factors"]), 1)
        visual = declared_configurations("formal", visual_taxonomy="/private/frozen-visual.json")
        self.assertEqual(len(visual), 14)

    def test_all_declared_training_plan_counts_and_argv_parse(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            protocol, cache_root, checkpoint = real_cached_protocol(path)
            weights = path / "weights.json"
            weights.write_text(json.dumps({backbone: {"checkpoint": str(checkpoint), "external_repo": str(path / "official")}
                                          for backbone in CORE_BACKBONES}))
            counts = {"smoke": 4, "pilot": 4, "benchmark": 24, "ablation": 21,
                      "sensitivity": 18, "representation": 12, "formal": 39, "extended": 57}
            for stage, count in counts.items():
                arguments = suite_parser().parse_args(["plan", "--phase", stage, "--protocol", str(protocol),
                                                       "--cache-root", str(cache_root), "--checkpoints", str(weights),
                                                       "--output", str(path / stage)])
                with patch("subprocess.check_output", return_value="test-git-commit\n"), patch("subprocess.Popen", side_effect=AssertionError("matrix planning cannot launch")):
                    plan_path = generate_plan(arguments)
                plan = json.loads(plan_path.read_text())
                self.assertEqual(len(plan["jobs"]), count)
                self.assertEqual(len({job["name"] for job in plan["jobs"]}), count)
                self.assertEqual(plan["job_count"], count)
                self.assertEqual({job["seed"] for job in plan["jobs"]}, {42} if stage in ("smoke", "pilot") else set(FORMAL_SEEDS))
                for job in plan["jobs"]:
                    command = trainer_command(plan, job, "python")
                    parsed = parser().parse_args(command[3:])
                    self.assertEqual(parsed.frames, job["overrides"]["frames"])
                    self.assertEqual(parsed.class_weights, job["overrides"]["class_weights"])
                    self.assertEqual(parsed.backbone_training, job["overrides"]["backbone_training"])
                    self.assertIn(job["configuration_id"], job["name"])
                    self.assertEqual(job["task_type"], "train")

    def test_robustness_plan_is_same_checkpoint_validation_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            protocol, cache_root, checkpoint = real_cached_protocol(path)
            weights = path / "weights.json"
            weights.write_text(json.dumps({"r2plus1d_18": str(checkpoint)}))
            args = suite_parser().parse_args(["plan", "--phase", "pilot", "--protocol", str(protocol),
                                              "--cache-root", str(cache_root), "--checkpoints", str(weights),
                                              "--output", str(path / "training")])
            with patch("subprocess.check_output", return_value="test-commit\n"):
                train_plan_path = generate_plan(args)
            training = json.loads(train_plan_path.read_text())
            for job in training["jobs"]:
                run = Path(job["output"])
                run.mkdir(parents=True)
                (run / "best.pt").write_bytes(b"test probe checkpoint")
                (run / "result.json").write_text(json.dumps({"status": "completed", "training_completed": True}))
            arguments = suite_parser().parse_args(["plan", "--phase", "robustness", "--trained-plan", str(train_plan_path),
                                                   "--output", str(path / "probes")])
            with patch("subprocess.Popen", side_effect=AssertionError("probe planning cannot launch")):
                probe_path = generate_plan(arguments)
            probes = json.loads(probe_path.read_text())
            self.assertEqual(probes["training_jobs"], 0)
            self.assertFalse(probes["test_evaluated"])
            self.assertEqual(len(probes["jobs"]), 6)  # R2 clinical flat/hierarchy, same 3 probes
            self.assertEqual(len(probes["contrasts"]), 4)
            for job in probes["jobs"]:
                command = trainer_command(probes, job, "python")
                parsed = parser().parse_args(command[3:])
                self.assertEqual(parsed.command, "evaluate")
                self.assertEqual(parsed.split, "val")
                self.assertFalse(parsed.allow_test)
                self.assertEqual(parsed.probe, job["probe"])
                self.assertNotIn("--epochs", command)
                self.assertEqual(job["checkpoint_sha256"], sha256_file(Path(job["checkpoint"])))
                self.assertIn("__probe_", job["configuration_id"])
            arguments.probes = "shuffle"
            arguments.output = path / "bad-probes"
            with self.assertRaises(ValueError):
                generate_plan(arguments)

    def test_explicit_execute_launches_exact_planned_count_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            jobs = []
            for index in range(9):
                jobs.append({"name": f"test-{index}", "task_type": "train", "output": str(path / "runs" / f"test-{index}"),
                             "backbone": "r2plus1d_18", "mode": "flat", "taxonomy": "clinical", "seed": 42,
                             "checkpoint": "/test-only/mock-pretrained.pth", "main_decoder": "flat"})
            plan = {"schema_version": "fsn-cvm-plan-1", "stage": "formal", "gpus": ["0", "1", "2", "3"],
                    "protocol": "/test-only/protocol.json", "cache_root": "/test-only/cache", "protocol_sha256": "a" * 64,
                    "code_hash": "b" * 64, "settings": {}, "jobs": jobs}
            plan_path = path / "run_plan.json"
            plan_path.write_text(json.dumps(plan))
            calls = []
            class FakeProcess:
                def __init__(self, command, **kwargs):
                    self.pid = 1000 + len(calls)
                    calls.append((command, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
                def wait(self):
                    return 0
                def poll(self):
                    return 0
            with patch("cvm.run_suite.preflight"), patch("subprocess.Popen", side_effect=FakeProcess):
                execute_plan(plan_path, python="python", execute=True)
            self.assertEqual(len(calls), 9)
            self.assertEqual({gpu for _, gpu in calls}, {"0", "1", "2", "3"})
            result = json.loads((path / "launcher_result.json").read_text())
            self.assertTrue(result["all_jobs_completed"])
            self.assertEqual(result["training_jobs"], 9)
            self.assertEqual(len(result["exits"]), 9)
            with patch("cvm.run_suite.preflight"), patch("subprocess.Popen", side_effect=AssertionError("must never launch a second time")), self.assertRaises(FileExistsError):
                execute_plan(plan_path, python="python", execute=True)

    def test_missing_modern_dependency_blocks_all_jobs_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            protocol, cache_root, checkpoint = real_cached_protocol(path)
            weights = path / "weights.json"
            weights.write_text(json.dumps({backbone: {"checkpoint": str(checkpoint), "external_repo": str(path / "official")}
                                          for backbone in CORE_BACKBONES}))
            args = suite_parser().parse_args(["plan", "--phase", "formal", "--amp", "none", "--protocol", str(protocol),
                                              "--cache-root", str(cache_root), "--checkpoints", str(weights),
                                              "--output", str(path / "formal")])
            with patch("subprocess.check_output", return_value="test-commit\n"):
                plan_path = generate_plan(args)
            with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}), patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=4), patch("cvm.models.build_model", side_effect=[toy_model(), RuntimeError("missing modern dependency")]), patch("subprocess.Popen", side_effect=AssertionError("no job may start before all dependencies pass")), self.assertRaisesRegex(RuntimeError, "missing modern dependency"):
                execute_plan(plan_path, python="python", execute=True)
            self.assertFalse((plan_path.parent / ".launch_once").exists())
            self.assertFalse((plan_path.parent / "launcher_logs").exists())


def _accumulation_test(weighted, mode):
    def run(self):
        test_partial_accumulation_matches_actual_effective_objective(weighted, mode)
    return run


def _integration_test(mode):
    def run(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            test_one_epoch_real_cache_protocol_and_outputs(real_cached_protocol(path), path, mode)
    return run


for _mode in ("flat", "aux_flat", "hierarchy", "capacity_control"):
    for _weighted in (False, True):
        setattr(TrainingTests, f"test_accumulation_{_mode}_{_weighted}", _accumulation_test(_weighted, _mode))
    setattr(TrainingTests, f"test_real_cache_{_mode}", _integration_test(_mode))

if __name__ == "__main__":
    unittest.main()
