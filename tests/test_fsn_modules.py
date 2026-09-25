import sys
import unittest
from pathlib import Path

import torch


MODEL_ROOT = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "Uni-AdaFocus-TSM-FSN"
)
sys.path.insert(0, str(MODEL_ROOT))

from archs.fsn_modules import LocalContextInteraction, LocalTemporalAdapter  # noqa: E402


class LocalTemporalAdapterTest(unittest.TestCase):
    def test_zero_init_preserves_pretrained_features_and_has_gradient(self):
        torch.manual_seed(1)
        features = torch.randn(8, 32, 5, 5, requires_grad=True)
        module = LocalTemporalAdapter(channels=32, bottleneck=8, temporal_kernel=3)
        output = module(features, num_segments=4)
        self.assertTrue(torch.equal(output, features))
        output.sum().backward()
        self.assertIsNotNone(module.alpha.grad)

    def test_frame_control_shape(self):
        module = LocalTemporalAdapter(channels=16, bottleneck=4, temporal_kernel=1)
        output = module(torch.randn(6, 16, 3, 3), num_segments=3)
        self.assertEqual(tuple(output.shape), (6, 16, 3, 3))


class LocalContextInteractionTest(unittest.TestCase):
    def test_different_local_and_global_timelines(self):
        batch = 2
        local = torch.randn(batch, 4, 64, 3, 3, requires_grad=True)
        global_ = torch.randn(batch, 6, 48, 5, 5, requires_grad=True)
        local_positions = torch.tensor([[0.05, 0.25, 0.65, 0.90], [0.10, 0.30, 0.60, 0.95]])
        global_positions = torch.tensor(
            [[0.02, 0.20, 0.40, 0.60, 0.80, 0.98], [0.01, 0.22, 0.43, 0.61, 0.79, 0.99]]
        )
        for mode in ("mlp", "cross_attention"):
            module = LocalContextInteraction(
                64, 48, 32, 7, mode=mode, heads=4, dropout=0.0
            )
            logits, attention = module(
                local, global_, local_positions, global_positions, return_attention=True
            )
            self.assertEqual(tuple(logits.shape), (batch, 7))
            if mode == "cross_attention":
                self.assertEqual(tuple(attention.shape), (batch, 36, 54))
                self.assertEqual(module.beta.item(), 0.0)
            logits.sum().backward(retain_graph=True)


if __name__ == "__main__":
    unittest.main()
