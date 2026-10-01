"""Launch dependencies, failure preservation, and paired completion checks."""
import json
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from experiments.relation.data import sha256_file
from experiments.relation.run_suite import (assert_idle_gpus, build_commands, execute_plan,
    gpu_indices, parser, validate_runtime_inputs)


class SourceLauncherTest(unittest.TestCase):
    def options(self, directory):
        return parser().parse_args(["--manifest-dir", directory, "--cache-root", directory,
            "--original-checkpoint", directory + "/best.pt", "--output", directory + "/suite"])

    def test_two_r_jobs_share_protocol_and_do_not_launch_old_training(self):
        commands = build_commands(self.options("/private"))
        self.assertEqual(set(commands), {"source_protocol", "cuda_smoke", "export", "fit_transition", "mlp", "dual", "evaluate"})
        for mode, device in (("mlp", "cuda:0"), ("dual", "cuda:1")):
            command = commands[mode]
            self.assertEqual(command[command.index("--device") + 1], device)
            self.assertEqual(command[command.index("--seed") + 1], "42")
            self.assertIn("--source-protocol-dir", command)
            self.assertEqual(command[command.index("--balance-loss") + 1], "train_ratio")
        self.assertNotIn("train_aligned_context", json.dumps(commands))
        self.assertNotIn("torchrun", json.dumps(commands))
        for invalid in ("0", "0,0", "-1,2", "0,1,2", "gpu0,1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                gpu_indices(invalid)

    def fake_run(self, output, fail=None, invalid_smoke=False, changed_export=False, changed_map=False):
        calls = []
        plan = {"commands": {stage: [stage] for stage in (
            "source_protocol", "cuda_smoke", "export", "fit_transition", "mlp", "dual", "evaluate")},
            "repository": {"commit": "frozen"}, "original_checkpoint_sha256": "a" * 64,
            "source_sha256": {"code.py": "b" * 64}, "manifest_sha256": {"train": "t", "val": "v"},
            "GPU_R_mapping": {"mlp": 0, "dual": 1}, "source_map_sha256": "map",
            "clip_counts": {"train": 7372, "val": 823}, "edge_counts": {"train": {"C": 3, "D": 2}},
            "settings": {"max_gap_seconds": 5.}}

        def run(stage, command):
            calls.append(stage)
            if stage == fail:
                return 1
            if stage == "source_protocol":
                (output / "source_protocol").mkdir()
                (output / "source_protocol" / "audit.json").write_text(json.dumps({
                    "source_map_sha256": "replaced" if changed_map else plan["source_map_sha256"],
                    "manifest_sha256": plan["manifest_sha256"], "clip_counts": plan["clip_counts"],
                    "edge_counts": plan["edge_counts"], "max_gap_seconds": 5.}))
            if stage == "cuda_smoke":
                smoke = {"all_passed": True, "real_cache_cuda_smoke_completed": not invalid_smoke,
                    "checkpoint_sha256": plan["original_checkpoint_sha256"], "source_sha256": plan["source_sha256"],
                    "manifest_sha256": plan["manifest_sha256"], "sampling": "36/8/12/128", "device": "cuda:0",
                    "source_audit_sha256": sha256_file(output / "source_protocol" / "audit.json")}
                (output / "cuda_smoke.json").write_text(json.dumps(smoke))
            if stage == "export":
                (output / "features").mkdir()
                (output / "features" / "index.jsonl").write_text("synthetic index")
                (output / "features" / "protocol.json").write_text(json.dumps({
                    "status": "complete", "A_frozen": True,
                    "checkpoint_sha256": "replaced" if changed_export else plan["original_checkpoint_sha256"],
                    "source_audit_sha256": sha256_file(output / "source_protocol" / "audit.json"),
                    "train_manifest_sha256": "t", "val_manifest_sha256": "v",
                    "index_sha256": sha256_file(output / "features" / "index.jsonl")}))
            if stage in ("mlp", "dual"):
                (output / stage).mkdir()
                (output / stage / "best.pt").write_bytes(b"synthetic")
                (output / stage / "metrics.json").write_text(json.dumps({"epochs_completed": 1, "val": {"bce": .5}}))
            if stage == "evaluate":
                (output / "evaluation").mkdir()
                (output / "evaluation" / "public_safe_summary.json").write_text(json.dumps({
                    "variants": {name: {} for name in ("A_only", "all_candidate", "source_rule", "learned_mlp", "learned_dual")}}))
            return 0
        return plan, calls, run

    def test_smoke_failure_blocks_export_and_keeps_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite"
            plan, calls, run = self.fake_run(output, invalid_smoke=True)
            with self.assertRaisesRegex(RuntimeError, "smoke report"):
                execute_plan(plan, output, run=run)
            self.assertEqual(calls, ["source_protocol", "cuda_smoke"])
            self.assertFalse(json.loads((output / "suite_result.json").read_text())["training_completed"])
            with self.assertRaises(FileExistsError):
                execute_plan(plan, output, run=run)

    def test_both_independent_r_exits_required_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite"
            plan, calls, run = self.fake_run(output, fail="mlp")
            with self.assertRaisesRegex(RuntimeError, "mlp exited"):
                execute_plan(plan, output, run=run)
            self.assertIn("dual", calls)
            self.assertNotIn("evaluate", calls)
            self.assertTrue((output / "dual" / "best.pt").exists())
            self.assertFalse(json.loads((output / "suite_result.json").read_text())["training_completed"])

    def test_success_requires_evaluation_and_saves_launch_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite"
            plan, calls, run = self.fake_run(output)
            status = execute_plan(plan, output, run=run)
            self.assertTrue(status["training_completed"])
            self.assertEqual(calls[:4], ["source_protocol", "cuda_smoke", "export", "fit_transition"])
            self.assertEqual(calls[-1], "evaluate")
            self.assertTrue((output / ".launch_once").exists())

    def test_changed_checkpoint_export_and_source_map_block_training(self):
        for changed_export, changed_map, message in ((True, False, "export differs"), (False, True, "source protocol differs")):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "suite"
                plan, calls, run = self.fake_run(output, changed_export=changed_export, changed_map=changed_map)
                with self.assertRaisesRegex(RuntimeError, message):
                    execute_plan(plan, output, run=run)
                self.assertNotIn("mlp", calls)
                self.assertFalse(json.loads((output / "suite_result.json").read_text())["training_completed"])

    def test_gpu_uuid_order_is_recorded_and_ambiguous_visibility_rejected(self):
        results = [SimpleNamespace(stdout="0, GPU-one\n1, GPU-two\n"),
                   SimpleNamespace(stdout=""), SimpleNamespace(stdout="")]
        with patch.dict("os.environ", {}, clear=True), patch("subprocess.run", side_effect=results):
            mapping = assert_idle_gpus([0, 1])
            self.assertEqual(mapping["CUDA_VISIBLE_DEVICES"], "GPU-one,GPU-two")
            self.assertEqual(mapping["physical_gpu_uuid"], {0: "GPU-one", 1: "GPU-two"})
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1,0"}, clear=True), patch("subprocess.run", return_value=results[0]):
            with self.assertRaisesRegex(RuntimeError, "identity mapping"):
                assert_idle_gpus([0, 1])

    def test_external_input_change_is_detected_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name in ("best.pt", "map.json", "train.jsonl", "val.jsonl"):
                (base / name).write_text(name)
            plan = {"arguments": {"original_checkpoint": str(base / "best.pt"),
                                  "source_map": str(base / "map.json"), "manifest_dir": directory},
                    "original_checkpoint_sha256": sha256_file(base / "best.pt"),
                    "source_map_sha256": sha256_file(base / "map.json"),
                    "manifest_sha256": {s: sha256_file(base / (s + ".jsonl")) for s in ("train", "val")},
                    "repository": {"commit": "fixed"}, "source_sha256": {"code": "digest"}}
            with patch("experiments.relation.run_suite.repository_state", return_value={"clean": True, "commit": "fixed"}), patch("experiments.relation.run_suite.source_hashes", return_value=plan["source_sha256"]):
                validate_runtime_inputs(plan)
                (base / "best.pt").write_text("replacement")
                with self.assertRaisesRegex(RuntimeError, "external checkpoint"):
                    validate_runtime_inputs(plan)


if __name__ == "__main__":
    unittest.main()
