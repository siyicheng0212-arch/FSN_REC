import unittest

import torch

from cvm.probes import apply_probe


class ProbeTests(unittest.TestCase):
    def test_shuffle_is_batch_order_independent_and_preserves_multiset(self):
        x = torch.arange(2 * 8 * 3).reshape(2, 8, 3, 1, 1)
        a = apply_probe(x, ["a", "b"], "shuffle", 42)
        b = apply_probe(x.flip(0), ["b", "a"], "shuffle", 42).flip(0)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a.sort(dim=1).values, x.sort(dim=1).values))
        self.assertFalse(torch.equal(a, x))

    def test_static_repeats_single_frame_and_none_preserves_object(self):
        x = torch.randn(2, 8, 3, 4, 4)
        self.assertIs(apply_probe(x, ["a", "b"]), x)
        y = apply_probe(x, ["a", "b"], "static")
        self.assertTrue(torch.equal(y[:, 0], x[:, 4]))
        self.assertTrue(torch.equal(y[:, 0], y[:, -1]))

    def test_rejects_malformed_input(self):
        with self.assertRaises(ValueError):
            apply_probe(torch.zeros(1, 8, 3, 4, 4), [], "none")
        with self.assertRaises(ValueError):
            apply_probe(torch.zeros(1, 8, 3, 4, 4), ["a"], "reverse")


if __name__ == "__main__":
    unittest.main()
