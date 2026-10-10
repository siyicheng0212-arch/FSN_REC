import unittest
from pathlib import Path

from data_tools.build_dataset import infer_group, parse_blocks, parse_time, parse_tsv_rows, parse_wide


class ExplicitTimestampUnitTests(unittest.TestCase):
    def test_clock_time_is_unambiguous_and_normalized_to_seconds(self):
        self.assertAlmostEqual(parse_time("00:00:01.500"), 1.5)
        event = parse_blocks("消毒\n00:00:00.000-00:00:01.500")[0]
        self.assertAlmostEqual(event["start_sec"], 0.0)
        self.assertAlmostEqual(event["end_sec"], 1.5)
        self.assertEqual(event["start_time_interpretation"], "clock_hms")
        self.assertEqual(event["end_time_interpretation"], "clock_hms")

    def test_bare_numeric_time_requires_explicit_unit(self):
        with self.assertRaisesRegex(ValueError, "ambiguous numeric timestamp"):
            parse_time("1500")
        self.assertEqual(parse_time("1500", "seconds"), 1500.0)
        self.assertEqual(parse_time("1500", "milliseconds"), 1.5)

    def test_wide_integer_timestamp_does_not_use_magnitude_heuristic(self):
        # Align the label cell and the TC range at the same character offset.
        text = "default      消毒\nTC           0-1500"
        seconds_event = parse_wide(text, numeric_time_unit="seconds")[0]
        millis_event = parse_wide(text, numeric_time_unit="milliseconds")[0]
        self.assertEqual(seconds_event["normalized_label"], "消毒")
        self.assertAlmostEqual(seconds_event["end_sec"], 1500.0)
        self.assertAlmostEqual(millis_event["end_sec"], 1.5)
        self.assertEqual(seconds_event["end_time_interpretation"], "numeric_seconds")
        self.assertEqual(millis_event["end_time_interpretation"], "numeric_milliseconds")
        with self.assertRaisesRegex(ValueError, "ambiguous numeric timestamp"):
            parse_wide(text)

    def test_tsv_numeric_unit_is_explicit(self):
        columns = ["default", "x", "0", "1000", "x", "2500", "x", "x", "进针"]
        text = "\t".join(columns)
        with self.assertRaisesRegex(ValueError, "ambiguous numeric timestamp"):
            parse_tsv_rows(text)
        event = parse_tsv_rows(text, numeric_time_unit="milliseconds")[0]
        self.assertAlmostEqual(event["start_sec"], 1.0)
        self.assertAlmostEqual(event["end_sec"], 2.5)
        self.assertEqual(event["raw_start_time"], "1000")
        self.assertEqual(event["raw_end_time"], "2500")


class VideoUnitGroupingTests(unittest.TestCase):
    def test_same_video_hash_groups_together_even_with_different_names(self):
        first = infer_group(
            "lishui", "ACU001-record-A",
            video_path=Path("/data/one.mp4"), video_sha256="abc123",
        )
        second = infer_group(
            "lishui", "ACU001-record-B",
            video_path=Path("/data/two.mp4"), video_sha256="abc123",
        )
        self.assertEqual(first["group_id"], second["group_id"])
        self.assertEqual(first["basis"], "exact_video_sha256")

    def test_different_videos_for_same_acu_code_are_distinct_groups(self):
        first = infer_group(
            "lishui", "ACU001-record-A",
            video_path=Path("/data/one.mp4"), video_sha256="hash-A",
        )
        second = infer_group(
            "lishui", "ACU001-record-B",
            video_path=Path("/data/two.mp4"), video_sha256="hash-B",
        )
        self.assertNotEqual(first["group_id"], second["group_id"])

    def test_unmatched_video_key_is_low_confidence_video_proxy(self):
        first = infer_group("youtube", "demo-video", video_path=None, video_sha256=None)
        second = infer_group("youtube", "demo-video", video_path=None, video_sha256=None)
        other = infer_group("youtube", "other-demo", video_path=None, video_sha256=None)
        self.assertEqual(first["group_id"], second["group_id"])
        self.assertNotEqual(first["group_id"], other["group_id"])
        self.assertEqual(first["basis"], "unmatched_annotation_video_key")
        self.assertEqual(first["confidence"], "low")


if __name__ == "__main__":
    unittest.main()
