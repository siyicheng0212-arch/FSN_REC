"""Contracts for the train-internal, non-role sweep/reperfusion probe."""

import unittest

import torch

from experiments.pair_evidence_probe import PairEvidenceProbe
from experiments.train_pair_probe import binary_metrics


class PairEvidenceProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_shape_backward_and_no_clinical_activity_outputs(self):
        model = PairEvidenceProbe("rgb_motion")
        frames = torch.rand(2, 36, 3, 224, 224)
        logits = model(frames)
        self.assertEqual(tuple(logits.shape), (2, 2))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        self.assertGreater(float(model.encoder[0].weight.grad.abs().sum()), 0)

    def test_equal_weights_compare_motion_to_rgb_only(self):
        torch.manual_seed(11)
        motion = PairEvidenceProbe("rgb_motion").eval()
        rgb = PairEvidenceProbe("rgb_only").eval()
        rgb.load_state_dict(motion.state_dict())
        still = torch.rand(1, 1, 3, 224, 224).repeat(1, 36, 1, 1, 1)
        moving = torch.rand(1, 36, 3, 224, 224)
        with torch.no_grad():
            self.assertTrue(torch.allclose(motion(still), rgb(still), atol=1e-6))
            self.assertFalse(torch.equal(motion(moving), rgb(moving)))

    def test_binary_metrics_uses_true_pair_labels(self):
        logits = torch.tensor([[3.0, 1.0], [2.0, 1.0], [0.0, 4.0], [1.0, 5.0]])
        target = torch.tensor([0, 1, 0, 1])
        result = binary_metrics(logits, target)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["macro_f1"], 0.5)

    def test_bad_shape_and_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            PairEvidenceProbe("role_detector")
        with self.assertRaisesRegex(ValueError, "expected RGB"):
            PairEvidenceProbe()(torch.rand(1, 35, 3, 224, 224))


if __name__ == "__main__":
    unittest.main()
