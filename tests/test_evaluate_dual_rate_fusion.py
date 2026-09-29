"""Regression tests for the fixed-fusion diagnostic and duration controls."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.evaluate_dual_rate_fusion import _duration_slices, _read_rows
from experiments.pilot_data import ClipRecord, EXPECTED_LABELS


def record(index: int, duration: float, label: int) -> ClipRecord:
    return ClipRecord(
        clip_id=f"clip-unit-{index}",
        split="val",
        label_id=label,
        normalized_label=EXPECTED_LABELS[label],
        source_collection="unit",
        group_id=f"group-{index}",
        video_path="/tmp/unused.mp4",
        clip_start_sec=0.0,
        clip_end_sec=duration,
        clip_duration_sec=duration,
        record_id=f"record-{index}",
    )


class DualRateFusionDiagnosticTest(unittest.TestCase):
    def test_duration_slices_partition_at_three_seconds(self):
        records = [record(0, 3.0, 3), record(1, 3.01, 4), record(2, 11.0, 1)]
        targets = torch.tensor([row.label_id for row in records])
        logits = torch.full((3, 7), -2.0)
        logits[torch.arange(3), targets] = 2.0
        slices = _duration_slices(logits, targets, records)
        self.assertEqual(slices["duration_le_3s"]["support"], 1)
        self.assertEqual(slices["duration_gt_3s"]["support"], 2)
        self.assertEqual(slices["duration_gt_10s"]["support"], 1)
        self.assertEqual(slices["duration_gt_3s"]["accuracy"], 1.0)

    def test_private_rows_must_match_frozen_clip_set_and_logits(self):
        records = [record(0, 2.0, 3), record(1, 4.0, 4)]
        rows = [
            {"clip_id": row.clip_id, "target": row.label_id,
             "prediction": row.label_id,
             "visual_logits": [10.0 if class_id == row.label_id else 0.0
                               for class_id in range(7)]}
            for row in records
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "private_rows.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(_read_rows(path, records).shape, (2, 7))
            path.write_text(json.dumps(rows[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "frozen validation clips"):
                _read_rows(path, records)


if __name__ == "__main__":
    unittest.main()
