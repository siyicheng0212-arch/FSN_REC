"""Synthetic CPU checks; these are not val823 or training performance claims."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import aligned_protocol
from experiments.metrics import compute_classification_metrics
from experiments.relation.data import sha256_file
from experiments.relation.tcn_baseline import FrozenEvidenceTCN, TCNConfig, collate_segments
from experiments.relation.videomae2_followup import SCHEMA as EXPORT_SCHEMA
from experiments.videomae_tcn import evidence, io, metrics, run, worker


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root):
    features, source = root / "features", root / "source"
    features.mkdir()
    source.mkdir()
    (features / "numeric").mkdir()
    rows, indexed, arrays = [], [], []
    rng = np.random.default_rng(42)
    specs = [("t0", "train", "clinical", "tc", 0), ("t1", "train", "clinical", "tc", 1),
             ("t2", "train", "network", "tn", 2), ("t3", "train", "network", "tn", 3),
             ("v0", "val", "clinical", "vc", 0), ("v1", "val", "clinical", "vc", 3),
             ("v2", "val", "network", "vn", 4)]
    for number, (clip, split, kind, record, label) in enumerate(specs):
        start = 3.0 * (number % 2)
        row = {"clip_id": clip, "split": split, "source_kind": kind, "group_id": record,
               "source_video_key": record, "recording_key": record, "timebase_key": record,
               "source_collection": "lishui" if kind == "clinical" else "FSN",
               "label_id": label, "duration": 2.0, "clip_start_sec": start, "clip_end_sec": start + 2}
        rows.append(row)
        logits = rng.normal(size=7).astype(np.float32)
        path = features / "numeric" / (clip + ".npz")
        np.savez(path, global_tokens=rng.normal(size=(8, 768)).astype(np.float32),
                 local_tokens=rng.normal(size=(8, 768)).astype(np.float32),
                 global_positions=np.linspace(0, 1, 8, dtype=np.float32),
                 local_positions=np.linspace(0, 1, 8, dtype=np.float32), logits=logits)
        indexed.append({**row, "feature_path": "numeric/" + clip + ".npz",
                        "feature_sha256": sha256_file(path), "checkpoint_sha256": "a" * 64})
        if split == "val":
            arrays.append((logits, label))
    chains = {"train": [{"chain_id": "tr", "split": "train",
                         "ordered_clip_ids": ["t0", "t1", "t2", "t3"], "eligible": [True, False, True]}],
              "val": [{"chain_id": "va", "split": "val", "ordered_clip_ids": ["v0", "v1", "v2"],
                       "eligible": [True, False]}]}
    edges = [{"edge_id": "e" + str(number), "left_clip_id": left, "right_clip_id": right,
              "split": split, "status": "C" if kind == "clinical" else "D", "source_kind": kind,
              "label_origin": "source_rule_v1"}
             for number, (left, right, split, kind) in enumerate([
                 ("t0", "t1", "train", "clinical"), ("t2", "t3", "train", "network"),
                 ("v0", "v1", "val", "clinical")])]
    artifacts = {"manifest_metadata.jsonl": rows, "edges.jsonl": edges,
                 "chains_train.jsonl": chains["train"], "chains_val.jsonl": chains["val"]}
    for name, values in artifacts.items():
        (source / name).write_text("".join(json.dumps(row) + "\n" for row in values))
    (features / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in indexed))
    counts = {"train": 4, "val": 3}
    audit = {"label_policy": "source_rule_v1", "data_protocol": aligned_protocol.DATA_PROTOCOL,
             "counts": counts, "manifest_sha256": aligned_protocol.EXPECTED_MANIFEST_SHA,
             "max_gap_seconds": 5.0,
             "files_sha256": {name: sha256_file(source / name) for name in artifacts}}
    write_json(source / "audit.json", audit)
    a_metrics = compute_classification_metrics(np.stack([v[0] for v in arrays]),
                   np.array([v[1] for v in arrays]), [{"duration": 2.0}] * 3)["all"]
    seal = {"schema": EXPORT_SCHEMA, "status": "complete", "A_frozen": True,
            "index_sha256": sha256_file(features / "index.jsonl"), "checkpoint_sha256": "a" * 64,
            "source_audit_sha256": sha256_file(source / "audit.json"),
            "data_protocol": aligned_protocol.DATA_PROTOCOL, "split_counts": counts,
            "test_read": False, "val_per_clip_A_logits_equal_separate_best_replay": True,
            "global_tokens": "last_block_spatial_mean_per_sampled_tubelet",
            "local_tokens": "last_block_spatial_max_per_sampled_tubelet", "A_val_metrics": a_metrics}
    seal.update({split + "_manifest_sha256": digest
                 for split, digest in aligned_protocol.EXPECTED_MANIFEST_SHA.items()})
    write_json(features / "protocol.json", seal)
    return features / "index.jsonl", source, counts


class VideoMAETCNTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.index, self.source, counts = fixture(self.root)
        self.count_patches = [patch.object(evidence, "EXPECTED_COUNTS", counts),
                              patch.object(aligned_protocol, "EXPECTED_COUNTS", counts)]
        for item in self.count_patches:
            item.start()
            self.addCleanup(item.stop)

    def prepare(self):
        return evidence.prepare(self.index, self.source)

    def test_existing_export_replay_layout_and_parameter_count(self):
        segments, metadata, info = self.prepare()
        self.assertEqual(info["counts"], {"train": 4, "val": 3})
        self.assertEqual([s.clip_ids for s in segments["train"]], [["t0", "t1"], ["t2", "t3"]])
        self.assertEqual([s.clip_ids for s in segments["val"]], [["v0", "v1"], ["v2"]])
        model = worker.model_from(info, torch.device("cpu"))
        self.assertIsInstance(model, FrozenEvidenceTCN)  # reuse the actual old class
        self.assertEqual(sum(p.numel() for p in model.parameters()), 148935)
        self.assertEqual(model.config.receptive_field_clips, 15)
        self.assertEqual(segments["train"][0].features.shape, (2, 1543))

    def test_three_real_numeric_batch_updates_and_A_is_not_an_optimizer_parameter(self):
        segments, _, info = self.prepare()
        batch = collate_segments(segments["train"])
        original_features, original_logits = batch[0].clone(), batch[1].clone()
        result = worker.smoke_batch(batch, info, torch.device("cpu"))
        self.assertTrue(result["all_passed"])
        self.assertTrue(result["initial_logits_equal_A"])
        self.assertFalse(result["A_instantiated"])
        self.assertEqual(result["steps"][0]["gradient_norms"]["stem"], 0.0)
        self.assertGreater(result["steps"][1]["gradient_norms"]["stem"], 0.0)
        self.assertGreater(result["steps"][1]["gradient_norms"]["blocks"], 0.0)
        self.assertTrue(torch.equal(batch[0], original_features))
        self.assertTrue(torch.equal(batch[1], original_logits))

    def test_singleton_preserves_old_forward_instead_of_forcing_A(self):
        segments, _, info = self.prepare()
        batch = collate_segments([segments["val"][-1]])
        model = worker.model_from(info, torch.device("cpu")).eval()
        with torch.no_grad():
            model.classifier.bias.fill_(0.4)
        output = model(batch[0], batch[1], batch[3])
        torch.testing.assert_close(output, batch[1] + 0.4)

    def test_nonfinite_evaluation_is_not_silently_converted_to_argmax(self):
        segments, _, info = self.prepare()
        model = worker.model_from(info, torch.device("cpu"))
        with torch.no_grad():
            model.classifier.bias.fill_(float("nan"))
        with self.assertRaisesRegex(RuntimeError, "nonfinite TCN evaluation"):
            worker.checked_evaluate(model, worker.loaders(segments)["val"], torch.device("cpu"))
        self.assertEqual(len(model._forward_hooks), 0)

    def test_candidate_time_cutoff_and_nonoverlap_are_validated(self):
        rows = [json.loads(line) for line in (self.source / "manifest_metadata.jsonl").read_text().splitlines()]
        edges = [json.loads(line) for line in (self.source / "edges.jsonl").read_text().splitlines()]
        with self.assertRaisesRegex(ValueError, "five-second"):
            evidence._candidate_time_checks(rows, edges, {"max_gap_seconds": 10})
        rows[1]["clip_start_sec"] = 1.0
        with self.assertRaisesRegex(ValueError, "overlapping"):
            evidence._candidate_time_checks(rows, edges, {"max_gap_seconds": 5})

    def test_protocol_schema_hash_wrong_model_and_feature_tampering_stop(self):
        seal_path = self.index.parent / "protocol.json"
        seal = json.loads(seal_path.read_text())
        for key, value in (("schema", "original-export"), ("test_read", True),
                           ("val_per_clip_A_logits_equal_separate_best_replay", False),
                           ("checkpoint_sha256", "b" * 64)):
            write_json(seal_path, {**seal, key: value})
            with self.assertRaises(ValueError):
                self.prepare()
        write_json(seal_path, seal)
        with (self.index.parent / "numeric" / "t0.npz").open("ab") as stream:
            stream.write(b"modified")
        with self.assertRaisesRegex(ValueError, "feature_sha256"):
            self.prepare()

    def test_A_replay_must_match_accuracy_confusion_and_not_just_F1(self):
        path = self.index.parent / "protocol.json"
        seal = json.loads(path.read_text())
        seal["A_val_metrics"]["accuracy"] += 0.01
        write_json(path, seal)
        with self.assertRaisesRegex(ValueError, "replay differs"):
            self.prepare()

    def test_source_file_seal_and_cross_recording_split_overlap_stop(self):
        path = self.source / "chains_val.jsonl"
        with path.open("a") as stream:
            stream.write(" \n")
        with self.assertRaisesRegex(ValueError, "artifact bytes"):
            self.prepare()
        rows = [{"split": "train", "group_id": "t", "source_video_key": "same", "recording_key": "t"},
                {"split": "val", "group_id": "v", "source_video_key": "same", "recording_key": "v"}]
        with self.assertRaisesRegex(ValueError, "overlap"):
            evidence._disjoint_recordings(rows)

    def test_read_only_plan_never_probes_gpu_or_creates_output(self):
        args = run.parser().parse_args(["--index", str(self.index), "--source-protocol-dir", str(self.source),
             "--output", str(self.root / "run"), "--expected-commit", "c" * 40,
             "--gpu", "1", "--gpu-uuid", "GPU-12345678-abcd-0000-0000-123456789abc"])
        with patch.object(run, "repository_state", return_value={"commit": "c" * 40, "clean": True}), \
                patch.object(run, "check_gpu", side_effect=AssertionError("GPU plan probe")):
            result = run.plan(args)
        self.assertFalse((self.root / "run").exists())
        self.assertEqual(result["gpu_physical_index"], 1)
        self.assertFalse(result["gpu_mapping_verified"])
        self.assertFalse(result["training_completed"])

    def test_output_overlap_and_overwrite_are_rejected(self):
        for path in (self.index.parent, self.index.parent / "new", self.root,
                     io.ROOT / "private_run"):
            with self.assertRaises((ValueError, FileExistsError)):
                run.fresh_output(path, self.index, self.source)
        path = self.root / "new"
        run.fresh_output(path, self.index, self.source)
        path.mkdir()
        with self.assertRaises(FileExistsError):
            run.fresh_output(path, self.index, self.source)

    def test_gpu_UUID_busy_guard_allows_idle_driver_bookkeeping(self):
        good = subprocess.CompletedProcess([], 0,
                "1, GPU-12345678, NVIDIA GeForce RTX 3090, 0, 1\n", "")
        empty = subprocess.CompletedProcess([], 0, "", "")
        with patch.object(run.subprocess, "run", side_effect=[good, empty]):
            self.assertEqual(run.check_gpu(1, "GPU-12345678")["memory_used_mib"], 1)
        busy = subprocess.CompletedProcess([], 0, "1245\n", "")
        with patch.object(run.subprocess, "run", side_effect=[good, busy]):
            with self.assertRaisesRegex(RuntimeError, "busy"):
                run.check_gpu(1, "GPU-12345678")
        with patch.object(run.subprocess, "run", return_value=good):
            with self.assertRaisesRegex(RuntimeError, "UUID"):
                run.check_gpu(0, "GPU-other")

    def test_stage_release_wait_requires_two_idle_samples_and_never_ignores_compute_PID(self):
        idle = {"uuid": "GPU-12345678", "utilization_percent": 0, "memory_used_mib": 1}
        with patch.object(run, "check_gpu", side_effect=[
                run.ReleaseTelemetryBusy("recent smoke use"), idle, idle]) as check, \
                patch.object(run.time, "sleep"):
            self.assertEqual(run.wait_released_gpu(1, "GPU-12345678"), idle)
            self.assertEqual(check.call_count, 3)
        with patch.object(run, "check_gpu", side_effect=RuntimeError("compute process busy")) as check:
            with self.assertRaisesRegex(RuntimeError, "compute process"):
                run.wait_released_gpu(1, "GPU-12345678")
            self.assertEqual(check.call_count, 1)
        with patch.object(run, "check_gpu", side_effect=run.ReleaseTelemetryBusy("active")), \
                patch.object(run.time, "monotonic", side_effect=[0, 0, 30]), \
                patch.object(run.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "30 seconds"):
                run.wait_released_gpu(1, "GPU-12345678")

    def test_safe_summary_has_slices_and_never_exports_clip_or_record_IDs(self):
        segments, metadata, _ = self.prepare()
        private = [{"clip_id": "v0", "label_id": 0, "a_prediction": 3, "tcn_prediction": 0},
                   {"clip_id": "v1", "label_id": 3, "a_prediction": 3, "tcn_prediction": 4},
                   {"clip_id": "v2", "label_id": 4, "a_prediction": 3, "tcn_prediction": 4}]
        report = metrics.summarize(private, metadata, segments["val"])
        pair = report["variants"]["VideoMAE_TCN"]["paired_vs_A"]
        self.assertEqual((pair["improved"], pair["worsened"]), (2, 1))
        self.assertEqual(report["slices"]["context"]["singleton"]["VideoMAE_TCN"]["count"], 1)
        self.assertEqual(report["recording_aggregate"]["recording_count"], 2)
        self.assertAlmostEqual(sum(row["macro_f1_contribution"] for row in report["per_class_changes"]),
              report["variants"]["VideoMAE_TCN"]["macro_f1"] - report["variants"]["VideoMAE_only"]["macro_f1"])
        text = json.dumps(report)
        for private_value in ("v0", "v1", "v2", "recording_key", str(self.root)):
            self.assertNotIn(private_value, text)

    def test_synthetic_training_checkpoint_and_strict_replay_pipeline(self):
        segments, metadata, info = self.prepare()
        config = {**evidence.TRAINING, "epochs": 2, "batch_size_segments": 2}
        output = self.root / "synthetic_run"
        output.mkdir()
        with patch.object(worker, "TRAINING", config):
            result = worker.train_head(segments, info, output / "tcn", torch.device("cpu"), "c" * 40)
            self.assertTrue(result["training_completed"])
            report = worker.evaluate_best(segments, metadata, info, output, torch.device("cpu"), "c" * 40)
            self.assertEqual(report["counts"]["val"], 3)
            self.assertTrue((output / "tcn" / "last.pt").is_file())
            self.assertEqual(json.loads((output / "tcn" / "history.json").read_text())[0]["train_clip_visits"], 4)
            path = output / "tcn" / "result.json"
            damaged = json.loads(path.read_text())
            damaged["best_checkpoint_sha256"] = "0" * 64
            write_json(path, damaged)
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                worker.evaluate_best(segments, metadata, info, output, torch.device("cpu"), "c" * 40)


if __name__ == "__main__":
    unittest.main()
