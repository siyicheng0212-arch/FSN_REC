"""Video-level and reproducibility tests for the IEA inner split."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from experiments.build_iea_inner_split import run, sha256
from experiments.pilot_data import ClipRecord, EXPECTED_LABELS


class IEAInnerSplitTest(unittest.TestCase):
    def test_train_only_group_split_is_reproducible_and_exhaustive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            rows = []
            for label in range(7):
                for index in range(12):
                    row = ClipRecord(
                        clip_id=f"clip-unit-{label}-{index}",
                        split="train",
                        label_id=label,
                        normalized_label=EXPECTED_LABELS[label],
                        source_collection="unit",
                        group_id=f"group-{label}-{index}",
                        video_path=f"/tmp/video-{label}-{index}.mp4",
                        clip_start_sec=0.0,
                        clip_end_sec=1.0,
                        clip_duration_sec=1.0,
                        record_id=f"record-{label}-{index}",
                    )
                    rows.append(row.to_manifest_dict())
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            common = dict(
                source_manifest=source,
                expected_sha256=sha256(source),
                val_fraction=0.2,
                min_per_class=1,
                seed=20260929,
            )
            first = run(Namespace(**common, output_dir=root / "first"))
            second = run(Namespace(**common, output_dir=root / "second"))
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            self.assertEqual(sum(first["counts"].values()), len(rows))
            self.assertFalse(first["group_overlap"])
            self.assertFalse(first["video_path_overlap"])
            self.assertFalse(first["independent_test_used"])
            self.assertTrue(all(
                count >= 1 for count in first["class_counts"]["val"].values()
            ))
            with self.assertRaises(FileExistsError):
                run(Namespace(**common, output_dir=root / "first"))

    def test_rejects_non_train_source(self):
        from experiments.build_iea_inner_split import choose_groups

        with self.assertRaisesRegex(ValueError, "train rows"):
            choose_groups([{
                "split": "val", "group_id": "a", "source_collection": "x",
                "label_id": 0,
            }], fraction=0.1, seed=1, min_per_class=1)


if __name__ == "__main__":
    unittest.main()
