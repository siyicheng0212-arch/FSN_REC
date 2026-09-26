import tempfile
import unittest
from pathlib import Path

from scripts.dedupe_videos import apply_plan, plan_source


class DedupeVideosTest(unittest.TestCase):
    def test_only_exact_same_basename_is_quarantined(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "ks_video"
            source.mkdir()
            nested = source / "copy"
            nested.mkdir()
            (source / "same.mp4").write_bytes(b"same-content")
            (nested / "same.mp4").write_bytes(b"same-content")
            (source / "alias.mp4").write_bytes(b"same-content")
            (source / "conflict.mp4").write_bytes(b"first")
            (nested / "conflict.mp4").write_bytes(b"second")

            plan, summary = plan_source(source)
            self.assertEqual(len(plan), 1)
            self.assertEqual(Path(plan[0]["keep"]), source / "same.mp4")
            self.assertEqual(Path(plan[0]["duplicate"]), nested / "same.mp4")
            self.assertEqual(summary["same_name_different_content_groups"], 1)
            self.assertFalse(any(Path(item["duplicate"]).name == "alias.mp4" for item in plan))

            quarantine = root / "quarantine"
            log = apply_plan(plan, root, quarantine)
            self.assertTrue(source.joinpath("same.mp4").is_file())
            self.assertFalse(nested.joinpath("same.mp4").exists())
            self.assertTrue(quarantine.joinpath("ks_video/copy/same.mp4.duplicate").is_file())
            self.assertTrue(log.is_file())


if __name__ == "__main__":
    unittest.main()
