import unittest
import numpy as np
import torch
from torch import nn

from experiments.relation.pipeline import predict_chain


class FixedRelation(nn.Module):
    def __init__(self, logit):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(float(logit)))
        self.calls = 0

    def forward(self, left, right):
        self.calls += 1
        return self.logit.expand(1)


class RelationPipelineTest(unittest.TestCase):
    def setUp(self):
        self.features = [{"logits": np.array(x),
                          "global_tokens": np.zeros((8, 1280)),
                          "local_tokens": np.zeros((12, 2048)),
                          "global_positions": np.linspace(0, 1, 8),
                          "local_positions": np.linspace(0, 1, 12)}
                         for x in ([0., 1.], [1., 0.])]
        self.transition = np.array([[.01, .99], [.01, .99]])

    def test_missing_order_singleton_low_q_return_independent_A(self):
        r = FixedRelation(10)
        result = predict_chain(r, self.features, self.transition)
        self.assertEqual(result["predictions"], [1, 0])
        self.assertEqual(r.calls, 0)
        result = predict_chain(r, self.features[:1], self.transition)
        self.assertEqual(result["predictions"], [1])
        result = predict_chain(r, self.features[:1], self.transition, eligible=[])
        self.assertEqual(result["predictions"], [1])
        low = predict_chain(FixedRelation(-10), self.features, self.transition, eligible=[True])
        self.assertEqual(low["predictions"], [1, 0])

    def test_high_q_enables_D_and_rejects_structural_nonbooleans(self):
        r = FixedRelation(10)
        result = predict_chain(r, self.features, self.transition, eligible=[True])
        self.assertEqual(result["predictions"], [1, 1])
        self.assertEqual(result["changed_indices"], [1])
        self.assertEqual(r.calls, 1)
        self.assertFalse(r.training)
        self.assertIsNone(r.logit.grad)
        for invalid in ([1], [True, False], []):
            with self.assertRaises(ValueError):
                predict_chain(r, self.features, self.transition, eligible=invalid)


if __name__ == "__main__":
    unittest.main()
