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
    KNOWN_CHECKPOINT_SHA, NON_CLI_PROTOCOL_FIELDS, PROTOCOL, VARIANTS, PreflightError,
    audit_manifests, build_command, claim_output, parse_args,
    run_queues, sha256_file, validate_smoke_report,
)

LABEL_NAMES = ("消毒", "进针", "运针", "扫散", "再灌注", "拔针", "固定")


class AlignedLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def manifest_fixture(self):
        directory = self.root / "manifests"
        directory.mkdir()
        for split in ("train", "val"):
            rows = [{"clip_id": f"clip-{split}-{label}", "split": split,
                     "label_id": label, "normalized_label": name,
                     "source_collection": "synthetic", "group_id": f"{split}-group-{label}",
                     "video_path": "/not-loaded/video.mp4", "clip_start_sec": 0.,
                     "clip_end_sec": 1., "clip_duration_sec": 1.}
                    for label, name in enumerate(LABEL_NAMES)]
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
        with self.assertRaisesRegex(PreflightError, "real group_id"):
            audit_manifests(directory, hashes, counts)

    def test_modified_manifest_is_rejected_before_parsing(self):
        directory, hashes, counts = self.manifest_fixture()
        with (directory / "train.jsonl").open("a") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(PreflightError, "SHA differs"):
            audit_manifests(directory, hashes, counts)

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
                "source_sha256": {"trainer.py": "def"},
                "checkpoint_sha256": KNOWN_CHECKPOINT_SHA,
                "manifest_sha256": {"train": "trainhash", "val": "valhash"},
                "cache_mapping_sha256": "cachehash",
                "commands": {variant: ["fake-trainer", variant] for variant in VARIANTS}}

    def smoke_fixture(self, plan):
        report = {key: plan[key] for key in ("source_sha256", "checkpoint_sha256",
                                            "manifest_sha256", "cache_mapping_sha256")}
        report.update(schema="fsn-aligned-context-smoke-v1", all_passed=True,
                      configuration={key: value for key, value in PROTOCOL.items()
                                     if key not in NON_CLI_PROTOCOL_FIELDS},
                      variants={variant: {"passed": True, "steps": 3, "losses": [1., .9, .8],
                                          "initial_max_abs_logit_diff": 0.,
                                          "device_index": {"original": 0, "context_plain": 1,
                                                           "context_aligned": 2, "local_capacity": 0}[variant],
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

    def run_fake_queue(self, failure=None, incomplete=None, dirty=False):
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
                                                            "best_val_macro_f1": .5, "test_metrics": None}))
                return 0

        summary = run_queues(plan, self.root, output, source_getter=lambda: plan["source_sha256"],
                             state_getter=lambda _: {"commit": "abc", "clean": not dirty},
                             popen=FakeProcess)
        return summary, calls, output

    def test_gpu0_capacity_runs_only_after_successful_original(self):
        summary, calls, _ = self.run_fake_queue()
        self.assertTrue(summary["all_complete"])
        sequence = [variant for variant, gpu in calls if gpu == "0"]
        self.assertEqual(sequence, ["original", "local_capacity"])
        self.assertEqual({gpu for _, gpu in calls}, {"0", "1", "2"})

    def test_failure_stops_its_queue_while_other_queues_finish(self):
        summary, calls, output = self.run_fake_queue(failure="original")
        self.assertFalse(summary["all_complete"])
        self.assertNotIn(("local_capacity", "0"), calls)
        self.assertEqual(summary["variants"]["context_plain"]["status"], "complete")
        self.assertEqual(summary["variants"]["context_aligned"]["status"], "complete")
        self.assertIn("skipped", (output / "launcher_logs" / "exit_codes.tsv").read_text())

    def test_exit_zero_without_results_does_not_release_capacity_job(self):
        summary, calls, _ = self.run_fake_queue(incomplete="original")
        self.assertFalse(summary["all_complete"])
        self.assertNotIn(("local_capacity", "0"), calls)

    def test_dirty_source_prevents_every_process_launch(self):
        summary, calls, _ = self.run_fake_queue(dirty=True)
        self.assertFalse(summary["all_complete"])
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
