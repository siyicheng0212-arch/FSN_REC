import json
import unittest

import torch

from experiments.metrics import DEFAULT_CLASS_NAMES, compute_classification_metrics


class ClassificationMetricsTest(unittest.TestCase):
    def setUp(self):
        # argmax predictions: [0, 1, 1, 1, 2, 4, 4, 0, 6]
        predictions = torch.tensor([0, 1, 1, 1, 2, 4, 4, 0, 6])
        self.logits = torch.full((len(predictions), 7), -3.0)
        self.logits[torch.arange(len(predictions)), predictions] = 3.0
        self.targets = torch.tensor([0, 0, 1, 1, 2, 3, 4, 5, 6])
        durations = [5.0, 10.0, 10.0, 10.1, 20.0, 2.0, 11.0, 9.0, 12.0]
        self.metadata = [
            {
                "clip_id": f"clip-{index}",
                "source_collection": "source-a" if index < 5 else "source-b",
                "clip_duration_sec": duration,
            }
            for index, duration in enumerate(durations)
        ]

    def test_overall_metrics_and_confusion_matrix(self):
        result = compute_classification_metrics(self.logits, self.targets, self.metadata)
        overall = result["all"]

        self.assertEqual(overall["num_samples"], 9)
        self.assertAlmostEqual(overall["accuracy"], 6 / 9)
        self.assertAlmostEqual(overall["micro_f1"], 6 / 9)
        self.assertAlmostEqual(overall["macro_f1"], (0.5 + 0.8 + 1 + 0 + 2 / 3 + 0 + 1) / 7)
        self.assertAlmostEqual(overall["weighted_f1"], (2 * 0.5 + 2 * 0.8 + 1 + 2 / 3 + 1) / 9)

        expected_confusion = [
            [1, 1, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ]
        self.assertEqual(overall["confusion_matrix"], expected_confusion)

        class_zero = overall["per_class"][0]
        self.assertEqual(class_zero["class_name"], DEFAULT_CLASS_NAMES[0])
        self.assertEqual(class_zero["support"], 2)
        self.assertAlmostEqual(class_zero["precision"], 0.5)
        self.assertAlmostEqual(class_zero["recall"], 0.5)
        self.assertAlmostEqual(class_zero["f1"], 0.5)

    def test_source_and_duration_slices_use_the_same_predictions(self):
        result = compute_classification_metrics(self.logits, self.targets, self.metadata)
        slices = result["slices"]

        self.assertEqual(slices["source_collection"]["source-a"]["num_samples"], 5)
        self.assertAlmostEqual(slices["source_collection"]["source-a"]["accuracy"], 4 / 5)
        self.assertEqual(slices["source_collection"]["source-b"]["num_samples"], 4)
        self.assertAlmostEqual(slices["source_collection"]["source-b"]["accuracy"], 2 / 4)

        # The 10.0-second boundary belongs to the short-duration slice.
        self.assertEqual(slices["duration_le_10s"]["num_samples"], 5)
        self.assertAlmostEqual(slices["duration_le_10s"]["accuracy"], 2 / 5)
        self.assertEqual(slices["duration_gt_10s"]["num_samples"], 4)
        self.assertAlmostEqual(slices["duration_gt_10s"]["accuracy"], 1.0)

    def test_collated_metadata_missing_values_and_json_safety(self):
        collated_metadata = {
            "source_collection": ["source-a", None],
            "clip_duration_sec": torch.tensor([10.0, float("nan")]),
        }
        # Missing duration is represented by omitting a value, not NaN, so use
        # row-form metadata for that case and verify its coverage accounting.
        rows = [
            {"source_collection": "source-a", "clip_duration_sec": 10.0},
            {"source_collection": None},
        ]
        result = compute_classification_metrics(self.logits[:2], self.targets[:2], rows)
        self.assertEqual(result["slice_coverage"]["duration_unknown"], 1)
        self.assertEqual(result["slice_coverage"]["source_unknown"], 1)
        json.dumps(result, ensure_ascii=False, allow_nan=False)

        # A normal DataLoader-style mapping of columns is accepted as well.
        collated_metadata["clip_duration_sec"] = torch.tensor([10.0, 11.0])
        collated_result = compute_classification_metrics(
            self.logits[:2], self.targets[:2], collated_metadata
        )
        self.assertEqual(collated_result["slices"]["duration_le_10s"]["num_samples"], 1)
        self.assertEqual(collated_result["slices"]["duration_gt_10s"]["num_samples"], 1)

    def test_rejects_bad_shapes_and_labels(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            compute_classification_metrics(torch.zeros(2, 6), torch.tensor([0, 1]), self.metadata[:2])
        with self.assertRaisesRegex(ValueError, "class ids"):
            compute_classification_metrics(self.logits[:2], torch.tensor([0, 7]), self.metadata[:2])

    def test_very_short_duration_sensitivity_slices(self):
        metadata = [
            {"source": "a", "duration": 0.06},
            {"source": "a", "duration": 0.10},
        ]
        result = compute_classification_metrics(self.logits[:2], self.targets[:2], metadata)
        self.assertEqual(result["slices"]["duration_lt_0.1s"]["num_samples"], 1)
        self.assertEqual(result["slices"]["duration_ge_0.1s"]["num_samples"], 1)


if __name__ == "__main__":
    unittest.main()
