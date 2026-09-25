from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments import pilot_data


class PilotDataTests(unittest.TestCase):
    def _source_row(self, root: Path, split: str, label_id: int, index: int) -> dict:
        video = (root / f"opaque-{split}-{label_id}-{index}.mp4").resolve()
        video.touch()
        return {
            "clip_id": f"clip-{split}-{label_id}-{index}",
            "split": split,
            "label_id": label_id,
            "normalized_label": pilot_data.EXPECTED_LABELS[label_id],
            "source_collection": f"source-{index % 2}",
            "group_id": f"group-{split}-{label_id}-{index}",
            "video_path": str(video),
            "clip_start_sec": 0.0,
            "clip_end_sec": 1.0,
            "clip_duration_sec": 1.0,
            "materialization_status": "ready_to_materialize",
        }

    def test_balanced_builder_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            for split in pilot_data.SPLITS:
                rows = [
                    self._source_row(root, split, label_id, index)
                    for label_id in pilot_data.EXPECTED_LABELS
                    for index in range(3)
                ]
                (source / f"{split}.jsonl").write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8",
                )
            first, second = root / "first", root / "second"
            summary = pilot_data.build_balanced_pilot_manifests(
                source, first, per_class=2, seed=7
            )
            pilot_data.build_balanced_pilot_manifests(source, second, per_class=2, seed=7)
            self.assertEqual(summary["splits"]["train"]["selected"], 14)
            for split in pilot_data.SPLITS:
                self.assertEqual(
                    (first / f"{split}.jsonl").read_bytes(),
                    (second / f"{split}.jsonl").read_bytes(),
                )

    def test_cache_and_dataset_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self._source_row(root, "test", 3, 0)
            manifest = root / "test.jsonl"
            manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            frames = np.arange(4 * 16 * 16 * 3, dtype=np.uint8).reshape(4, 16, 16, 3)
            with mock.patch.object(pilot_data, "extract_uniform_frames", return_value=frames):
                result = pilot_data.cache_manifest(
                    manifest, root / "cache", num_frames=4, crop_size=16
                )
            self.assertEqual(result, {"total": 1, "cached": 1, "reused": 0})
            dataset = pilot_data.PilotClipDataset(
                manifest, root / "cache", num_frames=4, crop_size=16
            )
            item = dataset[0]
            self.assertEqual(tuple(item["video"].shape), (4, 3, 16, 16))
            self.assertEqual(item["label"], 3)
            self.assertEqual(item["clip_id"], row["clip_id"])
            self.assertEqual(item["source"], row["source_collection"])
            self.assertEqual(item["duration"], 1.0)

    def test_decode_failure_is_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self._source_row(root, "test", 0, 0)
            record = pilot_data._parse_record(
                row, expected_split="test", require_video=True
            )
            failed = mock.Mock(returncode=1, stdout=b"", stderr=b"private path")
            with mock.patch.object(
                pilot_data, "_get_ffmpeg_exe", return_value="/fake/ffmpeg"
            ), mock.patch("subprocess.run", return_value=failed):
                with self.assertRaises(pilot_data.VideoDecodeError) as context:
                    pilot_data.extract_uniform_frames(record, num_frames=4, crop_size=16)
            self.assertNotIn(record.video_path, str(context.exception))


if __name__ == "__main__":
    unittest.main()

