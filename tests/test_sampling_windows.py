import hashlib
import json
import unittest
from pathlib import Path

from experiments.audit_focus_selection import frozen_hashes
from experiments.full_data import (
    CACHE_VERSION,
    request_digest,
    requested_timestamps,
    sampling_windows,
)
from experiments.pilot_data import ClipRecord, EXPECTED_LABELS


def record(duration: float) -> ClipRecord:
    return ClipRecord(
        clip_id="clip-sampling-test",
        split="train",
        label_id=3,
        normalized_label=EXPECTED_LABELS[3],
        source_collection="unit",
        group_id="unit-record",
        video_path="/tmp/not-read-by-uniform.mp4",
        clip_start_sec=10.0,
        clip_end_sec=10.0 + duration,
        clip_duration_sec=duration,
        record_id="unit-record",
    )


class SamplingWindowsTest(unittest.TestCase):
    def test_focus_audit_uses_frozen_protocol_hashes(self):
        protocol = Path(__file__).resolve().parents[1] / "configs/motion_sampling_protocol.json"
        hashes = frozen_hashes(protocol)
        self.assertEqual(
            hashes["train"],
            "093d0adf08dc1d382c0d2e1863a2f4724cb24ebd5b3d0af070e0c76ca989f1d3",
        )
        self.assertEqual(
            hashes["val"],
            "ad785ec18b14b63616582a143384fac649e69e27d142dfcf87c6852f9d3c6f02",
        )

    def test_uniform_cache_digest_remains_compatible(self):
        item = record(12.0)
        original_payload = {
            "version": CACHE_VERSION,
            "clip_id": item.clip_id,
            "video_path": item.video_path,
            "start": item.clip_start_sec,
            "end": item.clip_end_sec,
            "num_frames": 36,
            "crop_size": 224,
        }
        expected = hashlib.sha256(json.dumps(
            original_payload, ensure_ascii=False, sort_keys=True
        ).encode()).hexdigest()
        self.assertEqual(request_digest(item, 36, 224), expected)

    def test_three_windows_preserve_budget_and_short_clips(self):
        short = record(3.0)
        self.assertEqual(
            sampling_windows(short, 36, "three_windows"),
            sampling_windows(short, 36, "uniform"),
        )
        long = record(12.0)
        self.assertEqual(sampling_windows(long, 36, "three_windows"), [
            (11.5, 12.5, 12),
            (15.5, 16.5, 12),
            (19.5, 20.5, 12),
        ])
        timestamps = requested_timestamps(long, 36, "three_windows")
        self.assertEqual(len(timestamps), 36)
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertTrue(all(10.0 < time < 22.0 for time in timestamps))
        self.assertGreater(timestamps[12] - timestamps[11], 3.0)


if __name__ == "__main__":
    unittest.main()
