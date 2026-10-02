"""Pure paired audit checks runnable even on machines without PyTorch."""
import unittest

from experiments.legacy_tcn_audit.metrics import (
    assert_known_history, classification_metrics, paired, summary,
)


class LegacyTCNMetricsTest(unittest.TestCase):
    def test_incorrect_to_incorrect_is_not_a_rescue_or_spoil(self):
        self.assertEqual(
            paired([0, 0, 0], [1, 0, 1], [0, 1, 2]),
            {"changed": 3, "improved": 1, "worsened": 1,
             "wrong_to_different_wrong": 1},
        )

    def test_macro_f1_keeps_all_seven_classes_with_support(self):
        report = classification_metrics([0, 0, 1], [0, 1, 1])
        self.assertAlmostEqual(report["accuracy"], 2 / 3)
        self.assertAlmostEqual(report["macro_f1"], (2 / 3 + 2 / 3) / 7)
        self.assertEqual([v["support"] for v in report["per_class"]], [2, 1, 0, 0, 0, 0, 0])

    def test_known_historical_counts_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "historical report"):
            assert_known_history(summary([
                {"label_id": 0, "A_prediction": 0, "prediction": 0}
            ]))


if __name__ == "__main__":
    unittest.main()
