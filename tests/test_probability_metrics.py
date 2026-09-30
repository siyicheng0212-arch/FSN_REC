import unittest

import torch

from experiments.probability_metrics import probability_metrics


class ProbabilityMetricsTest(unittest.TestCase):
    def test_perfect_predictions_have_zero_brier_ece_and_nll(self):
        probabilities = torch.eye(7, dtype=torch.double)
        targets = torch.arange(7)
        metrics = probability_metrics(probabilities, targets)
        self.assertEqual(metrics["num_samples"], 7)
        self.assertEqual(metrics["decision_accuracy"], 1.0)
        self.assertEqual(metrics["top_label_ece"], 0.0)
        self.assertEqual(metrics["multiclass_brier"], 0.0)
        self.assertEqual(metrics["nll"], 0.0)

    def test_explicit_sequence_label_uses_its_probability(self):
        probabilities = torch.tensor([[0.6, 0.4, 0, 0, 0, 0, 0]], dtype=torch.double)
        targets = torch.tensor([1])
        metrics = probability_metrics(probabilities, targets, torch.tensor([1]))
        self.assertEqual(metrics["decision_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["mean_selected_probability"], 0.4)
        self.assertAlmostEqual(metrics["top_label_ece"], 0.6)
        self.assertAlmostEqual(metrics["multiclass_brier"], 0.72)

    def test_rejects_unnormalized_probabilities(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            probability_metrics(torch.ones((2, 7)), torch.tensor([0, 1]))


if __name__ == "__main__":
    unittest.main()
