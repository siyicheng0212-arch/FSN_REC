"""Source-policy supervision and chronological candidate construction checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.relation.source_edges import (
    LABEL_ORIGIN, build_source_relations, load_fixed_manifests, main, source_inventory,
)


def clip(identifier, start=0, end=1, *, split="train", source="hospital", group="record-a", video="/private/a.mp4", **extra):
    return dict(clip_id="clip-" + identifier, split=split, label_id=3, normalized_label="扫散",
                source_collection=source, group_id=group, video_path=video,
                clip_start_sec=start, clip_end_sec=end, clip_duration_sec=end-start, **extra)


class SourceRelationsTest(unittest.TestCase):
    source_map = {"hospital": "clinical", "web": "network"}

    def test_targets_are_explicit_policy_and_actions_do_not_determine_links(self):
        rows = [clip("c2", 2, 3), clip("c1", 0, 1),
                clip("n1", source="web", group="web-a", video="/private/w.mp4"),
                clip("n2", 1, 2, source="web", group="web-a", video="/private/w.mp4")]
        result = build_source_relations(rows, self.source_map)
        self.assertEqual([(e["status"], e["source_kind"]) for e in result["edges"]],
                         [("C", "clinical"), ("D", "network")])
        self.assertTrue(all(e["label_origin"] == LABEL_ORIGIN for e in result["edges"]))
        self.assertEqual(result["edges"][1]["reason"], "source_disallowed_for_flow")
        self.assertEqual(result["chains"]["train"][1]["eligible"], [True])
        changed = copy.deepcopy(rows)
        for index, row in enumerate(changed):
            row["label_id"] = index
        other = build_source_relations(changed, self.source_map)
        self.assertEqual(result["edges"], other["edges"])
        self.assertEqual(result["chains"], other["chains"])
        self.assertFalse(result["audit"]["actions_used_to_construct_edges"])

    def test_order_is_timestamp_order_not_supplied_list_order(self):
        rows = [clip("3", 3, 4), clip("1", 0, 1), clip("2", 1, 2)]
        result = build_source_relations(rows, self.source_map)
        self.assertEqual(result["chains"]["train"][0]["ordered_clip_ids"], ["clip-1", "clip-2", "clip-3"])
        self.assertEqual([(e["left_clip_id"], e["right_clip_id"]) for e in result["edges"]],
                         [("clip-1", "clip-2"), ("clip-2", "clip-3")])

    def test_overlap_duplicate_and_gap_cut_without_skipping_a_clip(self):
        rows = [clip("1", 0, 3), clip("2", 2, 4), clip("3", 4, 5), clip("4", 11, 12)]
        result = build_source_relations(rows, self.source_map, max_gap_seconds=5)
        chain = result["chains"]["train"][0]
        self.assertEqual(chain["eligible"], [False, True, False])
        self.assertEqual([(e["left_clip_id"], e["right_clip_id"]) for e in result["edges"]], [("clip-2", "clip-3")])
        duplicate = build_source_relations([clip("a", 0, 1), clip("b", 0, 1), clip("c", 1, 2)], self.source_map)
        self.assertEqual(duplicate["edges"], [])
        self.assertEqual(duplicate["chains"]["train"][0]["eligible"], [False, False])
        nested = build_source_relations([clip("a", 0, 20), clip("b", 2, 3), clip("c", 4, 5)], self.source_map)
        self.assertEqual(nested["edges"], [])

    def test_same_group_different_video_timebase_or_recording_cannot_connect(self):
        rows = [clip("a"), clip("b", 1, 2, video="/private/b.mp4"),
                clip("c", 2, 3, timebase_id="other"), clip("d", 3, 4, record_id="session-b")]
        result = build_source_relations(rows, self.source_map)
        self.assertEqual(result["edges"], [])
        self.assertEqual(len(result["chains"]["train"]), 4)
        identifiers = [node for chain in result["chains"]["train"] for node in chain["ordered_clip_ids"]]
        self.assertCountEqual(identifiers, [r["clip_id"] for r in rows])
        self.assertTrue(all(chain["eligible"] == [] for chain in result["chains"]["train"]))

    def test_train_val_preserved_and_no_clip_group_video_record_leakage(self):
        val = clip("v", split="val", group="val-record", video="/private/val.mp4", record_id="v")
        result = build_source_relations([clip("t", record_id="t"), val], self.source_map)
        self.assertEqual(result["audit"]["clip_counts"], {"train": 1, "val": 1})
        self.assertEqual(len(result["chains"]["val"]), 1)
        for override in ({"group_id": "record-a"}, {"video_path": "/private/a.mp4"}, {"record_id": "t"}, {"clip_id": "clip-t"}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                build_source_relations([clip("t", record_id="t"), val | override], self.source_map)

    def test_explicit_source_map_rejects_unknown_mixed_and_bad_values(self):
        with self.assertRaisesRegex(ValueError, "unmapped"):
            build_source_relations([clip("a", source="another_hospital")], self.source_map)
        with self.assertRaisesRegex(ValueError, "mixed source"):
            build_source_relations([clip("a"), clip("b", 1, 2, source="web")], self.source_map)
        for mapping in ({}, {"hospital": "clinical-ish"}, {"": "clinical"}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                build_source_relations([clip("a")], mapping)

    def test_bad_intervals_and_parameters_rejected(self):
        invalid = [{"clip_start_sec": -1}, {"clip_end_sec": 0}, {"clip_start_sec": float("nan")},
                   {"clip_end_sec": float("inf")}, {"clip_duration_sec": 2}, {"label_id": True},
                   {"clip_start_sec": True}, {"timebase_id": ""}, {"group_id": ""}, {"record_id": ""}]
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                build_source_relations([clip("a") | override], self.source_map)
        for gap in (-1, float("nan"), float("inf"), True):
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                build_source_relations([clip("a")], self.source_map, max_gap_seconds=gap)

    def test_inventory_has_aggregate_counts_without_paths_and_metadata_is_private(self):
        rows = [clip("a"), clip("b", 1, 2)]
        inventory = source_inventory(rows)
        self.assertEqual(inventory["hospital"]["train"], {"clips": 2, "groups": 1})
        self.assertNotIn("/private", json.dumps(inventory))
        result = build_source_relations(rows, self.source_map)
        self.assertNotIn("/private", json.dumps(result))
        self.assertEqual(result["metadata"][0]["duration"], 1.0)
        self.assertEqual(len(result["metadata"][0]["source_video_key"]), 64)
        self.assertEqual(result["audit"]["label_policy"], LABEL_ORIGIN)

    def test_formal_cli_rejects_subsets_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val"):
                row = clip(split, split=split, group=split, video=f"/private/{split}.mp4")
                (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
            (root / "map.json").write_text(json.dumps(self.source_map))
            with self.assertRaisesRegex(RuntimeError, "requires counts"):
                main(["--manifest-dir", str(root), "--source-map", str(root / "map.json"), "--output", str(root / "new")])
            self.assertFalse((root / "new").exists())
            with self.assertRaises(RuntimeError):
                load_fixed_manifests(root)

    def test_cli_writes_fresh_sealed_files_and_will_not_overwrite(self):
        rows = [clip("a"), clip("b", 1, 2), clip("v", split="val", group="val", video="/private/val.mp4")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "map.json").write_text(json.dumps(self.source_map))
            with patch("experiments.relation.source_edges.load_fixed_manifests", return_value=(rows, {"train": "a"*64, "val": "b"*64})):
                args = ["--manifest-dir", str(root), "--source-map", str(root / "map.json"), "--output", str(root / "new")]
                main(args)
                audit = json.loads((root / "new" / "audit.json").read_text())
                self.assertEqual(set(audit["files_sha256"]), {"edges.jsonl", "chains_train.jsonl", "chains_val.jsonl", "manifest_metadata.jsonl"})
                self.assertTrue(all(len(sha) == 64 for sha in audit["files_sha256"].values()))
                with self.assertRaises(FileExistsError):
                    main(args)


if __name__ == "__main__":
    unittest.main()
