"""Schema leakage checks and a tiny synthetic R training integration test."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.relation.data import (
    RelationDataset, audit_edges, load_edges, load_feature, load_feature_index, sha256_file,
)
from experiments.relation.train import parser, train


class RelationDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.index_path, self.edges_path = self.root / "features.jsonl", self.root / "edges.jsonl"
        self.index_rows, self.edge_rows = [], []
        generator = np.random.default_rng(42)
        for split in ("train", "val"):
            for number in range(3):
                clip_id = f"{split}{number}"
                feature_path = self.root / f"{clip_id}.npz"
                np.savez(feature_path, logits=generator.normal(size=7),
                         global_tokens=generator.normal(size=(3, 4)),
                         local_tokens=generator.normal(size=(2, 6)),
                         global_positions=np.linspace(0, 1, 3), local_positions=np.linspace(0, 1, 2))
                self.index_rows.append(dict(clip_id=clip_id, group_id=split, split=split,
                                            feature_path=feature_path.name, checkpoint_sha256="a" * 64,
                                            feature_sha256=sha256_file(feature_path)))
            for status, left, right in (("C", 0, 1), ("D", 1, 0), ("U", 1, 2)):
                self.edge_rows.append(dict(edge_id=f"{split}{status}", left_clip_id=f"{split}{left}",
                                           right_clip_id=f"{split}{right}", split=split, status=status))
        self.write(self.index_path, self.index_rows)
        self.write(self.edges_path, self.edge_rows)

    @staticmethod
    def write(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_unknown_is_excluded_and_action_logits_metadata_are_not_inputs(self):
        index = load_feature_index(self.index_path)
        edges = load_edges(self.edges_path, index)
        dataset = RelationDataset(index, edges, "train")
        self.assertEqual(len(dataset), 2)
        self.assertEqual([dataset[i]["target"].item() for i in range(2)], [1., 0.])
        self.assertEqual(set(dataset[0]["left"]),
                         {"global_tokens", "local_tokens", "global_positions", "local_positions"})
        self.assertEqual(audit_edges(edges)["val"], {"C": 1, "D": 1, "U": 1})
        unknown = RelationDataset(index, edges, "val", statuses=("U",))
        self.assertEqual(unknown[0]["target"].item(), -1.)

    def test_group_leakage_checkpoint_mix_duplicate_and_bad_paths_rejected(self):
        mutations = (
            lambda rows: rows[3].update(group_id="train"),
            lambda rows: rows[3].update(checkpoint_sha256="b" * 64),
            lambda rows: rows[3].update(clip_id="train0"),
            lambda rows: rows[0].update(checkpoint_sha256="fake"),
            lambda rows: rows[0].update(feature_path="../outside.npz"),
            lambda rows: rows[0].update(feature_path=str(self.root / "train0.npz")),
            lambda rows: rows[0].update(split="test"),
        )
        for mutate in mutations:
            rows = [dict(row) for row in self.index_rows]
            mutate(rows)
            self.write(self.index_path, rows)
            with self.assertRaises(ValueError):
                load_feature_index(self.index_path)

    def test_edge_split_unknown_endpoint_duplicate_conflict_and_self_edge_rejected(self):
        index = load_feature_index(self.index_path)
        mutations = (
            lambda rows: rows[0].update(right_clip_id="val0"),
            lambda rows: rows[0].update(right_clip_id="absent"),
            lambda rows: rows[0].update(right_clip_id="train0"),
            lambda rows: rows[0].update(status="automatic"),
            lambda rows: rows[1].update(edge_id="trainC"),
            lambda rows: rows[1].update(left_clip_id="train0", right_clip_id="train1"),
        )
        for mutate in mutations:
            rows = [dict(row) for row in self.edge_rows]
            mutate(rows)
            self.write(self.edges_path, rows)
            with self.assertRaises(ValueError):
                load_edges(self.edges_path, index)

    def test_corrupt_positions_or_nonfinite_features_do_not_become_zeros(self):
        feature_path = self.root / "train0.npz"
        arrays = load_feature(feature_path)
        for key, invalid in (("global_tokens", np.full((3, 4), np.nan)),
                             ("local_positions", np.array([1., 0.])),
                             ("logits", np.zeros(8))):
            changed = dict(arrays)
            changed[key] = invalid
            np.savez(feature_path, **changed)
            self.index_rows[0]["feature_sha256"] = sha256_file(feature_path)
            self.write(self.index_path, self.index_rows)
            with self.assertRaises(ValueError):
                load_feature_index(self.index_path)

    def test_C_cross_group_is_rejected_but_D_and_U_cross_group_are_allowed(self):
        self.index_rows[2]["group_id"] = "other_train_episode"
        self.write(self.index_path, self.index_rows)
        self.edge_rows[1].update(left_clip_id="train2", right_clip_id="train0")
        self.write(self.edges_path, self.edge_rows)
        index = load_feature_index(self.index_path)
        edges = load_edges(self.edges_path, index)
        self.assertEqual(len(edges), 6)
        self.edge_rows[0].update(right_clip_id="train2")
        self.write(self.edges_path, self.edge_rows)
        with self.assertRaisesRegex(ValueError, "C edges must stay within"):
            load_edges(self.edges_path, index)

    def test_feature_hash_and_post_preflight_tampering_are_rejected(self):
        index = load_feature_index(self.index_path)
        feature_path = self.root / "train0.npz"
        arrays = load_feature(feature_path)
        arrays["global_tokens"] += .1
        np.savez(feature_path, **arrays)
        with self.assertRaisesRegex(ValueError, "feature_sha256 does not match"):
            index.read("train0")
        with self.assertRaisesRegex(ValueError, "feature_sha256 does not match"):
            load_feature_index(self.index_path)
        self.index_rows[0]["feature_sha256"] = "invalid_hash"
        self.write(self.index_path, self.index_rows)
        with self.assertRaisesRegex(ValueError, "feature_sha256 must be"):
            load_feature_index(self.index_path)

    def test_different_clips_cannot_alias_same_feature_file(self):
        self.index_rows[1]["feature_path"] = "train0.npz"
        self.write(self.index_path, self.index_rows)
        with self.assertRaisesRegex(ValueError, "same resolved feature file"):
            load_feature_index(self.index_path)

    def test_one_epoch_cpu_produces_reloadable_best_and_unknown_audit(self):
        args = parser().parse_args(["--index", str(self.index_path), "--edges", str(self.edges_path),
                                    "--output", str(self.root / "run"), "--epochs", "1",
                                    "--batch-size", "2", "--dim", "8", "--heads", "2",
                                    "--mode", "mlp", "--device", "cpu"])
        metrics = train(args)
        self.assertEqual(metrics["epochs_completed"], 1)
        self.assertEqual(metrics["val_unknown"]["count"], 1)
        self.assertFalse(metrics["val"]["threshold_is_calibrated"])
        for name in ("protocol.json", "history.json", "metrics.json", "best.pt"):
            self.assertTrue((self.root / "run" / name).is_file())
        protocol = json.loads((self.root / "run" / "protocol.json").read_text())
        self.assertEqual(set(protocol["source_sha256"]), {"network.py", "data.py", "train.py"})
        self.assertEqual(protocol["unhashed_feature_count"], 0)
        with self.assertRaises(FileExistsError):
            train(args)

    def test_one_class_annotations_block_training_before_output(self):
        self.edge_rows = [row for row in self.edge_rows if row["status"] != "D"]
        self.write(self.edges_path, self.edge_rows)
        args = parser().parse_args(["--index", str(self.index_path), "--edges", str(self.edges_path),
                                    "--output", str(self.root / "run")])
        with self.assertRaisesRegex(ValueError, "both human-defined C and D"):
            train(args)
        self.assertFalse((self.root / "run").exists())

    def test_interrupted_or_modified_canonical_export_is_rejected(self):
        index_path = self.root / "index.jsonl"
        self.write(index_path, self.index_rows)
        protocol = self.root / "protocol.json"
        protocol.write_text(json.dumps({"A_frozen": True, "status": "exporting"}))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_feature_index(index_path)
        protocol.write_text(json.dumps({"A_frozen": True, "status": "complete",
                                        "index_sha256": sha256_file(index_path)}))
        self.assertEqual(len(load_feature_index(index_path).clips), 6)
        index_path.write_text(index_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "index changed"):
            load_feature_index(index_path)

    def test_train_fit_infer_cli_pipeline_and_index_protocol_lock(self):
        from experiments.relation.fit_transition import main as fit_main
        from experiments.relation.infer import main as infer_main
        for row in self.index_rows:
            row["label_id"] = int(row["clip_id"][-1])
        self.write(self.index_path, self.index_rows)
        args = parser().parse_args(["--index", str(self.index_path), "--edges", str(self.edges_path),
                                   "--output", str(self.root / "dual"), "--mode", "dual",
                                   "--epochs", "1", "--dim", "8", "--heads", "2"])
        train(args)
        transition = self.root / "transition"
        fit_main(["--index", str(self.index_path), "--edges", str(self.edges_path),
                  "--output", str(transition)])
        with np.load(transition / "transition.npz") as data:
            self.assertEqual(int(data["counts"].sum()), 1)  # val C did not enter fitting.
        chains = self.root / "chains.jsonl"
        self.write(chains, [{"chain_id": "val_chain", "ordered_clip_ids": ["val0", "val1", "val2"],
                             "eligible": [True, True]},
                            {"chain_id": "val_single", "ordered_clip_ids": ["val0"], "eligible": []}])
        common = ["--index", str(self.index_path), "--relation-checkpoint", str(self.root / "dual" / "best.pt"),
                  "--transition", str(transition), "--chains", str(chains), "--threshold", "1"]
        infer_main(common + ["--output", str(self.root / "inference")])
        results = [json.loads(line) for line in (self.root / "inference" / "predictions.jsonl").read_text().splitlines()]
        result = results[0]
        self.assertEqual(result["predictions"], result["visual_predictions"])
        self.assertEqual(result["effective_reliability"], [0., 0.])
        self.assertEqual(len(result["relation_probability"]), 2)
        self.assertEqual(results[1]["predictions"], results[1]["visual_predictions"])
        self.index_path.write_text(self.index_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "exact feature index"):
            infer_main(common + ["--output", str(self.root / "invalid_inference")])
        self.assertFalse((self.root / "invalid_inference").exists())


if __name__ == "__main__":
    unittest.main()
