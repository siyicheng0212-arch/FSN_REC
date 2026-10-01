"""Source-policy end-to-end CPU integration and immutable split correspondence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.aligned_protocol import EXPECTED_COUNTS, EXPECTED_MANIFEST_SHA
from experiments.relation.data import load_edges, load_feature_index, sha256_file
from experiments.relation.evaluate_suite import main as evaluate_main
from experiments.relation.fit_transition import main as fit_main
from experiments.relation.protocol import edge_label_policy, validate_source_artifacts
from experiments.relation.source_edges import main as build_main
from experiments.relation.train import parser, train


class SourceProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.map_path = self.root / "map.json"
        self.map_path.write_text(json.dumps({"lishui": "clinical", "youtube": "network"}))
        self.manifests = self.root / "manifests"
        self.manifests.mkdir()
        self.rows = []
        for split in ("train", "val"):
            rows = []
            for source in ("lishui", "youtube"):
                group = split + source
                for i in range(2):
                    rows.append({"clip_id": f"clip-{group}-{i}", "split": split, "group_id": group,
                        "source_collection": source, "label_id": i, "normalized_label": ("消毒", "进针")[i],
                        "video_path": f"/private/{group}.mp4", "record_id": group,
                        "clip_start_sec": float(i), "clip_end_sec": float(i+1), "clip_duration_sec": 1.})
            self.rows.extend(rows)
            self.write(self.manifests / (split + ".jsonl"), rows)
        self.hashes = {s: sha256_file(self.manifests / (s + ".jsonl")) for s in ("train", "val")}
        self.addCleanup(patch.stopall)
        patch.dict(EXPECTED_COUNTS, {"train": 4, "val": 4}).start()
        patch.dict(EXPECTED_MANIFEST_SHA, self.hashes).start()
        self.protocol = self.root / "source"
        build_main(["--manifest-dir", str(self.manifests), "--source-map", str(self.map_path),
                    "--output", str(self.protocol)])
        self.features = self.root / "features"
        self.features.mkdir()
        metadata = [json.loads(x) for x in (self.protocol / "manifest_metadata.jsonl").read_text().splitlines()]
        feature_rows = []
        generator = np.random.default_rng(42)
        for row in metadata:
            filename = row["clip_id"] + ".npz"
            path = self.features / filename
            np.savez(path, logits=generator.normal(size=7), global_tokens=generator.normal(size=(3, 4)),
                     local_tokens=generator.normal(size=(2, 6)), global_positions=np.linspace(0, 1, 3),
                     local_positions=np.linspace(0, 1, 2))
            feature_rows.append({**row, "feature_path": filename, "feature_sha256": sha256_file(path),
                                 "checkpoint_sha256": "a" * 64})
        self.index = self.features / "index.jsonl"
        self.write(self.index, feature_rows)
        (self.features / "protocol.json").write_text(json.dumps({"A_frozen": True, "status": "complete",
            "index_sha256": sha256_file(self.index), "train_manifest_sha256": self.hashes["train"],
            "val_manifest_sha256": self.hashes["val"]}))

    @staticmethod
    def write(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_source_train_fit_evaluate_and_rules_share_one_fixed_protocol(self):
        common = ["--index", str(self.index), "--edges", str(self.protocol / "edges.jsonl"),
                  "--source-protocol-dir", str(self.protocol)]
        checkpoints = {}
        for mode in ("mlp", "dual"):
            run = self.root / mode
            args = parser().parse_args(common + ["--output", str(run), "--mode", mode, "--epochs", "1",
                                               "--dim", "8", "--heads", "2", "--balance-loss", "train_ratio"])
            metrics = train(args)
            self.assertEqual(metrics["val"]["count"], 2)
            saved = torch.load(run / "best.pt", weights_only=True)
            self.assertEqual(saved["protocol"]["label_policy"], "source_rule_v1")
            self.assertTrue(saved["protocol"]["source_confounding_possible"])
            self.assertEqual(saved["protocol"]["train_only_positive_weight"], 1.)
            checkpoints[mode] = str(run / "best.pt")
        transition = self.root / "transition"
        fit_main(common + ["--output", str(transition)])
        with np.load(transition / "transition.npz") as fitted:
            self.assertEqual(int(fitted["counts"].sum()), 1)  # Only train clinical C counts.
        summary = evaluate_main(["--index", str(self.index), "--protocol-dir", str(self.protocol),
            "--transition", str(transition), "--mlp-checkpoint", checkpoints["mlp"],
            "--dual-checkpoint", checkpoints["dual"], "--output", str(self.root / "evaluation")])
        self.assertEqual(set(summary["variants"]), {"A_only", "all_candidate", "source_rule", "learned_mlp", "learned_dual"})
        for metric in summary["variants"].values():
            self.assertEqual(metric["count"], 4)
        self.assertEqual(summary["variants"]["source_rule"]["network_candidate_gate"]["active"], 0)
        encoded = json.dumps(summary)
        self.assertNotIn("clip-train", encoded)
        self.assertNotIn("/private", encoded)

    def test_metadata_or_files_mutation_and_source_label_mismatch_are_rejected(self):
        index = load_feature_index(self.index)
        validate_source_artifacts(self.protocol, index, self.protocol / "edges.jsonl")
        path = self.protocol / "chains_val.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            validate_source_artifacts(self.protocol, index)
        edges = [json.loads(x) for x in (self.protocol / "edges.jsonl").read_text().splitlines()]
        edges[0]["source_kind"] = "network"
        other = self.root / "bad_edges.jsonl"
        self.write(other, edges)
        with self.assertRaisesRegex(ValueError, "label disagrees"):
            load_edges(other, index)

    def test_source_labels_cannot_silently_enter_manual_training(self):
        args = parser().parse_args(["--index", str(self.index), "--edges", str(self.protocol / "edges.jsonl"),
                                    "--output", str(self.root / "bad")])
        with self.assertRaisesRegex(ValueError, "require --source-protocol"):
            train(args)
        self.assertFalse((self.root / "bad").exists())
        with self.assertRaises(ValueError):
            edge_label_policy([{"label_origin": "source_rule_v1"}, {}])


if __name__ == "__main__":
    unittest.main()
