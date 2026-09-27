import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from data_tools.build_dataset import (
    duplicate_preference_key,
    normalized_key,
    upload_copy_suffix_rank,
)


class UploadCopySuffixRankTest(unittest.TestCase):
    def test_numeric_upload_copy_is_ranked_after_original(self):
        self.assertEqual(upload_copy_suffix_rank("示例视频"), 0)
        self.assertEqual(upload_copy_suffix_rank("示例视频-1"), 1)
        self.assertEqual(upload_copy_suffix_rank("示例视频-12"), 1)

    def test_non_numeric_hyphenated_name_is_not_treated_as_copy(self):
        self.assertEqual(upload_copy_suffix_rank("2b-演示"), 0)
        self.assertEqual(upload_copy_suffix_rank("case-A"), 0)

    def test_unsuffixed_original_wins_when_both_copies_lack_video(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            txt_root = root / "txt"
            source = txt_root / "youtube_txt"
            source.mkdir(parents=True)
            original = source / "示例视频.txt"
            copy = source / "示例视频-1.txt"

            chosen = min(
                [copy, original],
                key=lambda path: duplicate_preference_key(path, txt_root, {}),
            )
            self.assertEqual(chosen, original)

    def test_unique_video_match_still_wins_over_filename_rank(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            txt_root = root / "txt"
            source = txt_root / "youtube_txt"
            source.mkdir(parents=True)
            original = source / "示例视频.txt"
            copy = source / "示例视频-1.txt"
            matched_video = root / "youtube_video" / "示例视频-1.mp4"
            video_index = {("youtube", normalized_key("示例视频-1")): [matched_video]}

            chosen = min(
                [copy, original],
                key=lambda path: duplicate_preference_key(path, txt_root, video_index),
            )
            self.assertEqual(chosen, copy)


if __name__ == "__main__":
    unittest.main()
