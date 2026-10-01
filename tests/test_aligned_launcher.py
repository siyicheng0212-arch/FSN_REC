"""Run safety and queue sequencing tests without private clips or CUDA."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest

from experiments.run_aligned_context import (
    DATA_PROTOCOL, KNOWN_CHECKPOINT_SHA, KNOWN_COUNTS, KNOWN_MANIFEST_SHA,
    NON_CLI_PROTOCOL_FIELDS, PROTOCOL, VARIANTS, PreflightError,
    audit_manifests, build_command, claim_output, parse_args,
    run_queues, sha256_file, validate_smoke_report,
)

LABEL_NAMES = ("消毒", "进针", "运针", "扫散", "再灌注", "拔针", "固定")


class AlignedLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def manifest_fixture(self, inner_split_schema=False):
        directory = self.root / "manifests"
        directory.mkdir()
        for split in ("train", "val"):
            rows = [{"clip_id": f"clip-{split}-{label}", "split": split,
                     "label_id": label, "normalized_label": name,
                     "source_collection": "synthetic", "group_id": f"{split}-group-{label}",
                     "video_path": "/not-loaded/video.mp4", "clip_start_sec": 0.,
                     "clip_end_sec": 1., "clip_duration_sec": 1.}
                    for label, name in enumerate(LABEL_NAMES)]
            if inner_split_schema:
                for row in rows:
                    row["split"] = "train"
                    row["inner_split_role"] = split
            (directory / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        hashes = {split: sha256_file(directory / f"{split}.jsonl") for split in ("train", "val")}
        return directory, hashes, {"train": 7, "val": 7}

    def test_group_leakage_rejected_even_with_matching_file_hash(self):
        directory, hashes, counts = self.manifest_fixture()
        path = directory / "val.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["group_id"] = "train-group-0"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        hashes["val"] = sha256_file(path)
        with self.assertRaisesRegex(PreflightError, "overlapping group_id"):
            audit_manifests(directory, hashes, counts)

    def test_missing_group_does_not_use_clip_id_as_evidence(self):
        directory, hashes, counts = self.manifest_fixture()
        path = directory / "val.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        del rows[0]["group_id"]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        hashes["val"] = sha256_file(path)
        with self.assertRaisesRegex(PreflightError, "group_id"):
            audit_manifests(directory, hashes, counts)

    def test_inner_validation_role_is_logical_without_changing_manifest_sha(self):
        directory, hashes, counts = self.manifest_fixture(inner_split_schema=True)
        path = directory / "val.jsonl"
        original_bytes = path.read_bytes()
        records, summary = audit_manifests(directory, hashes, counts)
        self.assertEqual({record.split for record in records["val"]}, {"val"})
        self.assertEqual(summary["val"]["sha256"], hashes["val"])
        self.assertEqual(path.read_bytes(), original_bytes)
        self.assertEqual(sha256_file(path), hashes["val"])
        raw = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual({row["split"] for row in raw}, {"train"})

    def test_wrong_inner_role_is_rejected_despite_cache_origin_split(self):
        directory, hashes, counts = self.manifest_fixture(inner_split_schema=True)
        path = directory / "val.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["inner_split_role"] = "train"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        hashes["val"] = sha256_file(path)
        with self.assertRaisesRegex(PreflightError, "role"):
            audit_manifests(directory, hashes, counts)

    def test_inner_split_schema_does_not_allow_train_val_group_overlap(self):
        directory, hashes, counts = self.manifest_fixture(inner_split_schema=True)
        path = directory / "val.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["group_id"] = "train-group-0"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        hashes["val"] = sha256_file(path)
        with self.assertRaisesRegex(PreflightError, "overlapping group_id"):
            audit_manifests(directory, hashes, counts)

    def test_modified_manifest_is_rejected_before_parsing(self):
        directory, hashes, counts = self.manifest_fixture()
        with (directory / "train.jsonl").open("a") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(PreflightError, "SHA differs"):
            audit_manifests(directory, hashes, counts)

    def test_formal_manifest_count_defaults_require_restored_full_split(self):
        directory, hashes, _ = self.manifest_fixture()
        self.assertEqual(KNOWN_COUNTS, {"train": 7372, "val": 823})
        with self.assertRaisesRegex(PreflightError, "count differs"):
            audit_manifests(directory, expected_sha=hashes)

    def test_read_only_is_default_and_execute_requires_smoke(self):
        options = ["--manifest-dir", str(self.root), "--cache-dir", str(self.root),
                   "--checkpoint", str(self.root / "weight.pt"), "--output-dir", str(self.root / "out")]
        args = parse_args(options)
        self.assertFalse(args.execute)
        self.assertFalse((self.root / "out").exists())
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args([*options, "--execute"])

    def test_atomic_output_claim_refuses_even_an_empty_existing_directory(self):
        output = self.root / "out"
        output.mkdir()
        with self.assertRaisesRegex(PreflightError, "another launcher"):
            claim_output(output, {"repository": {"commit": "abc"}})
        self.assertEqual(list(output.iterdir()), [])

    def test_all_commands_share_protocol_and_official_checkpoint(self):
        args = argparse.Namespace(python="python", manifest_dir=self.root / "manifests",
                                  cache_dir=self.root / "cache", output_dir=self.root / "out",
                                  checkpoint=self.root / "official.pt")
        for variant in VARIANTS:
            command = build_command(variant, args)
            self.assertEqual(command[command.index("--checkpoint") + 1], str(args.checkpoint))
            self.assertEqual(int(command[command.index("--batch-size") + 1]) *
                             int(command[command.index("--accumulation-steps") + 1]), 64)
            self.assertNotIn("--allow-random-init", command)
            self.assertNotIn("--test-after-training", command)
            self.assertNotIn("--resume", command)

    def plan_fixture(self, output):
        return {"repository": {"commit": "abc", "clean": True},
                "data_protocol": DATA_PROTOCOL,
                "source_sha256": {"trainer.py": "def"},
                "checkpoint_sha256": KNOWN_CHECKPOINT_SHA,
                "manifest_sha256": dict(KNOWN_MANIFEST_SHA),
                "manifests": {split: {"clips": count} for split, count in KNOWN_COUNTS.items()},
                "cache_mapping_sha256": "cachehash",
                "commands": {variant: ["fake-trainer", variant] for variant in VARIANTS}}

    def smoke_fixture(self, plan):
        report = {key: plan[key] for key in ("data_protocol", "source_sha256", "checkpoint_sha256",
                                            "manifest_sha256", "cache_mapping_sha256")}
        report.update(schema="fsn-aligned-context-smoke-v1", all_passed=True,
                      split_audit={"data_protocol": DATA_PROTOCOL, "counts": dict(KNOWN_COUNTS),
                                   "manifest_sha256": dict(KNOWN_MANIFEST_SHA)},
                      configuration={key: value for key, value in PROTOCOL.items()
                                     if key not in NON_CLI_PROTOCOL_FIELDS},
                      variants={variant: {"passed": True, "steps": 3, "losses": [1., .9, .8],
                                          "initial_max_abs_logit_diff": 0.,
                                          "device_index": {"original": 0, "context_plain": 1,
                                                           "context_aligned": 2, "local_capacity": 3}[variant],
                                          "gradient_checks": [{"up.weight": 1., "down.weight": 0.},
                                                              {"up.weight": 1., "down.weight": .2},
                                                              {"up.weight": 1., "down.weight": .3}]}
                                for variant in VARIANTS})
        path = self.root / "smoke.json"
        path.write_text(json.dumps(report))
        return path, report

    def test_smoke_must_match_sources_and_have_three_finite_steps(self):
        plan = self.plan_fixture(self.root / "out")
        path, report = self.smoke_fixture(plan)
        validate_smoke_report(path, plan)
        report["source_sha256"] = {"trainer.py": "new"}
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(PreflightError, "source_sha256"):
            validate_smoke_report(path, plan)
        path, report = self.smoke_fixture(plan)
        report["variants"]["context_aligned"]["losses"][2] = float("nan")
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(PreflightError, "context_aligned"):
            validate_smoke_report(path, plan)

    def test_smoke_requires_actual_staged_gradients_and_declared_device(self):
        plan = self.plan_fixture(self.root / "out")
        for invalid in ("zero_up", "zero_downstream", "missing_downstream",
                        "nonfinite_downstream", "wrong_device", "negative_difference"):
            with self.subTest(invalid=invalid):
                path, report = self.smoke_fixture(plan)
                row = report["variants"]["context_aligned"]
                if invalid == "zero_up":
                    row["gradient_checks"][0]["up.weight"] = 0.
                elif invalid == "zero_downstream":
                    row["gradient_checks"][2]["down.weight"] = 0.
                elif invalid == "missing_downstream":
                    row["gradient_checks"][2]["down.weight"] = None
                elif invalid == "nonfinite_downstream":
                    row["gradient_checks"][2]["down.weight"] = float("inf")
                elif invalid == "wrong_device":
                    row["device_index"] = 0
                else:
                    row["initial_max_abs_logit_diff"] = -1.
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(PreflightError, "context_aligned"):
                    validate_smoke_report(path, plan)

    def test_smoke_with_other_batch_or_module_dimension_cannot_authorize_formal_run(self):
        plan = self.plan_fixture(self.root / "out")
        for field, invalid in (("batch_size", 2), ("context_dim", 32)):
            with self.subTest(field=field):
                path, report = self.smoke_fixture(plan)
                report["configuration"][field] = invalid
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(PreflightError, field):
                    validate_smoke_report(path, plan)

    def test_old_internal_validation_smoke_cannot_authorize_full_split_run(self):
        plan = self.plan_fixture(self.root / "out")
        path, report = self.smoke_fixture(plan)
        report["data_protocol"] = "old-internal-train6586-val786"
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(PreflightError, "data_protocol"):
            validate_smoke_report(path, plan)

    def test_smoke_must_bind_full_counts_and_frozen_manifest_hashes(self):
        plan = self.plan_fixture(self.root / "out")
        for field, invalid in (("counts", {"train": 6586, "val": 786}),
                               ("manifest_sha256", {"train": "old-inner", "val": "old-inner"})):
            with self.subTest(field=field):
                path, report = self.smoke_fixture(plan)
                report["split_audit"][field] = invalid
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(PreflightError, "split_audit"):
                    validate_smoke_report(path, plan)

    def run_fake_queue(self, failure=None, incomplete=None, dirty=False, old_internal_result=None):
        output = self.root / "out"
        plan = self.plan_fixture(output)
        claim_output(output, plan)
        calls, guard = [], threading.Lock()

        class FakeProcess:
            def __init__(self, command, **kwargs):
                variant = command[1]
                with guard:
                    calls.append((variant, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
                    self.pid = 100 + len(calls)
                self.variant = variant

            def wait(self):
                if self.variant == failure:
                    return 2
                if self.variant == incomplete:
                    return 0
                run = output / self.variant / "seed_42"
                run.mkdir(parents=True)
                for name in ("best.pt", "load_report.json", "val_predictions.jsonl"):
                    (run / name).write_text("fixture")
                (run / "history.json").write_text(json.dumps([{"phase": "finetune", "epoch": 0}]))
                (run / "result.json").write_text(json.dumps({"variant": self.variant, "seed": 42,
                                                            "training_completed": True,
                                                            "data_protocol": plan["data_protocol"],
                                                            "split_audit": {
                                                                "data_protocol": plan["data_protocol"],
                                                                "counts": {"train": 6586, "val": 786} if self.variant == old_internal_result else dict(KNOWN_COUNTS),
                                                                "manifest_sha256": dict(plan["manifest_sha256"])},
                                                            "best_val_macro_f1": .5, "test_metrics": None}))
                return 0

        summary = run_queues(plan, self.root, output, source_getter=lambda: plan["source_sha256"],
                             state_getter=lambda _: {"commit": "abc", "clean": not dirty},
                             popen=FakeProcess)
        return summary, calls, output

    def test_four_variants_have_independent_gpu_assignments(self):
        summary, calls, _ = self.run_fake_queue()
        self.assertTrue(summary["all_complete"])
        self.assertEqual(dict(calls), {"original": "0", "context_plain": "1",
                                      "context_aligned": "2", "local_capacity": "3"})

    def test_original_failure_does_not_stop_independent_capacity_job(self):
        summary, calls, output = self.run_fake_queue(failure="original")
        self.assertFalse(summary["all_complete"])
        self.assertIn(("local_capacity", "3"), calls)
        self.assertEqual(summary["variants"]["context_plain"]["status"], "complete")
        self.assertEqual(summary["variants"]["context_aligned"]["status"], "complete")
        self.assertEqual(summary["variants"]["local_capacity"]["status"], "complete")
        self.assertIn("process_failed", (output / "launcher_logs" / "exit_codes.tsv").read_text())

    def test_exit_zero_without_results_does_not_count_as_training_success(self):
        summary, calls, _ = self.run_fake_queue(incomplete="original")
        self.assertFalse(summary["all_complete"])
        self.assertIn(("local_capacity", "3"), calls)
        self.assertEqual(summary["variants"]["local_capacity"]["status"], "complete")

    def test_dirty_source_prevents_every_process_launch(self):
        summary, calls, _ = self.run_fake_queue(dirty=True)
        self.assertFalse(summary["all_complete"])
        self.assertEqual(calls, [])

    def test_old_internal_result_is_not_counted_as_completed_full_training(self):
        summary, calls, _ = self.run_fake_queue(old_internal_result="original")
        self.assertFalse(summary["all_complete"])
        self.assertEqual(summary["variants"]["original"]["status"], "result_data_protocol_mismatch")
        self.assertEqual(summary["variants"]["local_capacity"]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
