import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = next(
    path for path in (
        ROOT / "third_party" / "Uni-AdaFocus" / "Uni-AdaFocus-TSM with Experiments on Sth-Sth V1&V2 and Jester",
        ROOT / "models" / "Uni-AdaFocus-TSM-FSN",
    )
    if path.is_dir()
)
sys.path.insert(0, str(MODEL_ROOT))

from archs.fsn_modules import (  # noqa: E402
    LocalContextInteraction,
    LocalTemporalAdapter,
    OrderedTemporalPool,
)


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
                self.assertEqual(module.beta.item(), 1.0)
            self.assertEqual(module.gamma.item(), 0.0)
            self.assertTrue(torch.equal(logits, torch.zeros_like(logits)))
            logits.sum().backward(retain_graph=True)

    def test_zero_head_is_equivalent_but_classifier_receives_gradient(self):
        module = LocalContextInteraction(
            16, 12, 8, 7, mode="cross_attention", heads=2,
            dropout=0.0, output_init="zero_head",
        )
        local = torch.randn(2, 4, 16, 2, 2)
        global_ = torch.randn(2, 3, 12, 2, 2)
        local_positions = torch.rand(2, 4)
        global_positions = torch.rand(2, 3)
        logits, _ = module(local, global_, local_positions, global_positions)
        self.assertTrue(torch.equal(logits, torch.zeros_like(logits)))
        self.assertEqual(module.gamma.item(), 1.0)
        logits.sum().backward()
        self.assertGreater(float(module.classifier[-1].weight.grad.abs().sum()), 0.0)

    def test_ordered_pool_is_order_sensitive(self):
        torch.manual_seed(7)
        pool = OrderedTemporalPool(8)
        sequence = torch.randn(2, 5, 8)
        forward = pool(sequence)
        backward = pool(sequence.flip(1))
        self.assertFalse(torch.allclose(forward, backward))


if __name__ == "__main__":
    unittest.main()
