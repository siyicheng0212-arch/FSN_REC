import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cvm.paired import run_pairs


class PairedBatchTests(unittest.TestCase):
    def fixture(self, root, *, probe=False, frames=False, incomplete=True):
        metadata = {"mode": "flat", "trained_heads": ["flat"],
                    "taxonomy": {"name": "clinical", "groups": [[0], [6], [1, 2, 5], [3, 4]]},
                    "protocol_sha256": "a" * 64, "split": "val", "selection_exposure": "development",
                    "checkpoint_sha256": "b" * 64,
                    "eval_metadata": {"backbone": "r2plus1d_18", "probe": "none",
                                      "preprocessing": {"frames": 16, "crop_size": [112, 112]}}}
        rows = [{"clip_id": f"PRIVATE_CLIP_{index}", "group_id": f"PRIVATE_GROUP_{index // 2}",
                 "source": "PRIVATE_HOSPITAL", "duration": 2., "target": index % 7,
                 "repeated_frame_fraction": .5,
                 "flat_logits": [8. if label == index % 7 else -8. for label in range(7)]}
                for index in range(14)]
        seeds = [42, 2026] if incomplete else [42]
        jobs = []
        for configuration in ("baseline", "candidate"):
            for seed in seeds:
                directory = root / configuration / str(seed)
                task = "evaluate" if probe else "train"
                jobs.append({"configuration_id": configuration, "seed": seed,
                             "output": str(directory), "task_type": task})
                if seed != 42:
                    continue
                directory.mkdir(parents=True)
                actual_metadata = json.loads(json.dumps(metadata))
                actual_rows = json.loads(json.dumps(rows))
                if configuration == "candidate":
                    if probe:
                        actual_metadata["eval_metadata"]["probe"] = "shuffle"
                    if frames:
                        actual_metadata["eval_metadata"]["preprocessing"]["frames"] = 32
                        for row in actual_rows:
                            row["repeated_frame_fraction"] = .75
                    actual_rows[0]["flat_logits"] = [-8., 8., -8., -8., -8., -8., -8.]
                name = "predictions.jsonl" if task == "evaluate" else "val_predictions.jsonl"
                (directory / name).write_text("\n".join(json.dumps(row) for row in actual_rows))
                (directory / "prediction_metadata.json").write_text(json.dumps(actual_metadata))
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps({"jobs": jobs}))
        contrast = {"contrast_id": "predeclared_pair", "baseline_configuration_id": "baseline",
                    "candidate_configuration_id": "candidate", "declared_seeds": seeds,
                    "completed_paired_seeds": [42],
                    "comparison_kind": "same_checkpoint_probe" if probe else "single_factor" if frames else "paired_training",
                    "varying_factors": ["probe"] if probe else ["frames"] if frames else []}
        summary = {"protocol_sha256": "a" * 64, "plan_sha256": ["c" * 64], "comparison_rows": [contrast]}
        return plan_path, summary

    def test_declared_complete_pair_and_missing_seed_are_both_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, summary = self.fixture(root)
            with patch("cvm.paired.collect_plans", return_value=summary):
                result = run_pairs([path], root / "aggregates", replicates=10)
            self.assertEqual(result["status"], "completed_with_missing_pairs")
            self.assertEqual([record["status"] for record in result["pairs"]], ["completed", "missing_or_incomplete"])
            report_path = root / "aggregates" / result["pairs"][0]["aggregate_file"]
            report = json.loads(report_path.read_text())
            self.assertLess(report["paired_uncertainty"]["metrics"]["macro_f1"]["paired_delta"], 0)
            public = report_path.read_text() + json.dumps(result)
            self.assertNotIn("PRIVATE_", public)
            self.assertNotIn(str(root), public)

    def test_preflight_rejection_never_creates_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new"
            with patch("cvm.paired.collect_plans", side_effect=ValueError("invalid plan")):
                with self.assertRaises(ValueError):
                    run_pairs([], output)
            self.assertFalse(output.exists())

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                run_pairs([], Path(folder))

    def test_temporal_probe_reads_evaluation_artifacts_and_preserves_sha_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, summary = self.fixture(root, probe=True, incomplete=False)
            with patch("cvm.paired.collect_plans", return_value=summary):
                result = run_pairs([path], root / "aggregate", replicates=10)
            report = json.loads((root / "aggregate" / result["pairs"][0]["aggregate_file"]).read_text())
            self.assertEqual(report["paired_uncertainty"]["comparison_type"], "same_checkpoint_temporal_probe")

    def test_declared_frame_sensitivity_permits_only_its_treatment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, summary = self.fixture(root, frames=True, incomplete=False)
            with patch("cvm.paired.collect_plans", return_value=summary):
                result = run_pairs([path], root / "aggregate", replicates=10)
            report = json.loads((root / "aggregate" / result["pairs"][0]["aggregate_file"]).read_text())
            self.assertEqual(report["paired_uncertainty"]["comparison_type"], "frames16_vs32")

    def test_private_cohort_mismatch_cannot_leave_success_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, summary = self.fixture(root, incomplete=False)
            predictions = root / "candidate" / "42" / "val_predictions.jsonl"
            rows = [json.loads(line) for line in predictions.read_text().splitlines()]
            rows[0]["clip_id"] = "other_private_id"
            predictions.write_text("\n".join(json.dumps(row) for row in rows))
            with patch("cvm.paired.collect_plans", return_value=summary):
                with self.assertRaises(ValueError):
                    run_pairs([path], root / "aggregate", replicates=10)
            status = json.loads((root / "aggregate" / "paired_status.json").read_text())
            self.assertEqual(status["status"], "failed_or_interrupted")


if __name__ == "__main__":
    unittest.main()
