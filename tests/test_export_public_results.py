import unittest

from experiments.export_public_results import assert_public


class PublicResultExportTest(unittest.TestCase):
    def test_accepts_metrics_and_hashes(self):
        assert_public({"macro_f1": 0.8, "sha256": "abc", "test_metrics": None})

    def test_rejects_private_paths_recursively(self):
        with self.assertRaisesRegex(ValueError, "private path"):
            assert_public({"nested": [{"checkpoint": "/root/private/best.pt"}]})
        with self.assertRaisesRegex(ValueError, "private path"):
            assert_public({"video": "/Users/person/dataset/video.mp4"})


if __name__ == "__main__":
    unittest.main()
