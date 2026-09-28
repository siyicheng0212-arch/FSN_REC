import json
import tempfile
import unittest
from pathlib import Path

from experiments.aggregate_sequence_results import aggregate


def metric(value):
    return {
        "macro_f1": value,
        "accuracy": value + 0.02,
        "per_class": [{"f1": value} for _ in range(7)],
    }


class AggregateSequenceResultsTest(unittest.TestCase):
    def write_result(self, root, seed, baseline, sequence, hashes=None, test=None):
        path = root / f"seed-{seed}.json"
        path.write_text(json.dumps({
            "seed": seed,
            "manifest_sha256": hashes or {"train": "a", "val": "b"},
            "transition_weight": 1.0,
            "transition_prior": {"smoothing": 1.0},
            "baseline_metrics": {"all": metric(baseline)},
            "sequence_metrics": {"all": metric(sequence)},
            "changed_predictions": 3,
            "test_metrics": test,
        }), encoding="utf-8")
        return path

    def test_three_seed_acceptance_and_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [
                self.write_result(root, 42, 0.79, 0.81),
                self.write_result(root, 123, 0.78, 0.795),
                self.write_result(root, 2026, 0.80, 0.812),
            ]
            result = aggregate(paths)
            self.assertTrue(result["acceptance"]["passed"])
            self.assertAlmostEqual(result["delta_macro_f1"]["mean"], 0.047 / 3)
            self.assertEqual(result["delta_macro_f1"]["degrees_of_freedom"], 2)
            self.assertIsNotNone(result["delta_macro_f1"]["t_95_interval"])

    def test_rejects_test_metrics_and_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = self.write_result(root, 1, 0.7, 0.8)
            bad_test = self.write_result(root, 2, 0.7, 0.8, test={"accuracy": 1})
            with self.assertRaisesRegex(ValueError, "test metrics"):
                aggregate([good, bad_test])
            mismatch = self.write_result(
                root, 3, 0.7, 0.8, hashes={"train": "x", "val": "b"}
            )
            with self.assertRaisesRegex(ValueError, "different manifests"):
                aggregate([good, mismatch])


if __name__ == "__main__":
    unittest.main()
