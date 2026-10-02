"""Hard cuts must prevent temporal feature and gradient leakage at every layer."""

import unittest

import torch
import torch.nn.functional as F
from torch import nn

from experiments.fsn_tcn.gates import select_gates
from experiments.fsn_tcn.model import SegmentedTCN, TCNConfig, TemporalResidualTCN


class FSNTCNModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(218)
        self.features = torch.randn(7, 8)
        self.a_logits = torch.randn(7, 7)

    @staticmethod
    def base(trained=False, dropout=0):
        base = TemporalResidualTCN(TCNConfig(input_dim=8, width=6, layers=3, dropout=dropout))
        if trained:
            # Nonzero residual is essential: a zero-initialized head would make
            # every isolation test pass trivially without exercising the TCN.
            nn.init.normal_(base.output_projection.weight, std=.2)
            nn.init.normal_(base.output_projection.bias, std=.1)
        return base

    def test_new_reference_starts_exactly_at_a_and_singleton_stays_a(self):
        base = self.base(dropout=.3).train()
        torch.testing.assert_close(base(self.features, self.a_logits), self.a_logits, rtol=0, atol=0)
        trained = self.base(trained=True).train()
        singleton = self.a_logits[:1]
        self.assertIs(trained(self.features[:1], singleton), singleton)
        self.assertFalse(torch.equal(trained(self.features, self.a_logits), self.a_logits))

    def test_all_open_invokes_the_unchanged_base_and_all_closed_is_exact_a(self):
        for mode in ("train", "eval"):
            base = self.base(trained=True)
            getattr(base, mode)()
            wrapper = SegmentedTCN(base)
            torch.testing.assert_close(wrapper(self.features, self.a_logits, [True] * 6),
                                       base(self.features, self.a_logits), rtol=0, atol=0)
            self.assertIs(wrapper(self.features, self.a_logits, [False] * 6), self.a_logits)
            singleton = self.a_logits[:1]
            self.assertIs(wrapper(self.features[:1], singleton, []), singleton)

    def test_every_segment_uses_the_same_base_and_singletons_bypass_it(self):
        class CountingBase(nn.Module):
            def __init__(self):
                super().__init__()
                self.lengths = []

            def forward(self, features, a_logits):
                self.lengths.append(len(features))
                return a_logits + 1

        base = CountingBase()
        actual = SegmentedTCN(base)(self.features, self.a_logits,
                                    [True, False, False, True, True, False])
        self.assertEqual(base.lengths, [2, 3])
        expected = self.a_logits.clone()
        expected[:2] += 1
        expected[3:6] += 1
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_right_features_and_logits_cannot_change_left_across_cut_in_train_or_eval(self):
        gates = [True, True, False, True, True, True]
        for mode in ("train", "eval"):
            wrapper = SegmentedTCN(self.base(trained=True))
            getattr(wrapper, mode)()
            expected = wrapper(self.features, self.a_logits, gates)
            changed_features, changed_logits = self.features.clone(), self.a_logits.clone()
            changed_features[3:] = torch.randn_like(changed_features[3:]) * 100
            changed_logits[3:] = torch.randn_like(changed_logits[3:]) * 100
            actual = wrapper(changed_features, changed_logits, gates)
            torch.testing.assert_close(actual[:3], expected[:3], rtol=0, atol=0)
            self.assertFalse(torch.equal(actual[3:], expected[3:]))
            changed_left = self.features.clone()
            changed_left[:3] = torch.randn_like(changed_left[:3]) * 100
            changed_left_logits = self.a_logits.clone()
            changed_left_logits[:3] = torch.randn_like(changed_left_logits[:3]) * 100
            torch.testing.assert_close(wrapper(changed_left, changed_left_logits, gates)[3:],
                                       expected[3:], rtol=0, atol=0)
            # Information really crosses an OPEN edge in this nonzero model.
            all_open = wrapper(self.features, self.a_logits, [True] * 6)
            modified_open = wrapper(changed_features, changed_logits, [True] * 6)
            self.assertFalse(torch.equal(all_open[:3], modified_open[:3]))

    def test_gradient_does_not_cross_cut_but_model_and_left_features_receive_gradients(self):
        wrapper = SegmentedTCN(self.base(trained=True)).train()
        features = self.features.clone().requires_grad_()
        logits = self.a_logits.clone().requires_grad_()
        output = wrapper(features, logits, [True, True, False, True, True, True])
        F.cross_entropy(output[:3], torch.tensor([1, 2, 3])).backward()
        self.assertGreater(features.grad[:3].abs().sum().item(), 0)
        torch.testing.assert_close(features.grad[3:], torch.zeros_like(features.grad[3:]), rtol=0, atol=0)
        torch.testing.assert_close(logits.grad[3:], torch.zeros_like(logits.grad[3:]), rtol=0, atol=0)
        self.assertGreater(wrapper.base.output_projection.weight.grad.abs().sum().item(), 0)

    def test_zero_head_updates_first_then_internal_layers_receive_gradients(self):
        model = SegmentedTCN(self.base()).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        before = model.base.output_projection.weight.detach().clone()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(self.features, self.a_logits, [True] * 6)
            loss = F.cross_entropy(output, torch.tensor([0, 1, 2, 3, 4, 5, 6]))
            loss.backward()
            self.assertTrue(torch.isfinite(loss).item())
            self.assertGreater(model.base.output_projection.weight.grad.abs().sum().item(), 0)
            optimizer.step()
        self.assertFalse(torch.equal(before, model.base.output_projection.weight.detach()))
        self.assertGreater(model.base.input_projection.weight.grad.abs().sum().item(), 0)

    def test_no_batch_norm_and_cpu_bfloat16_forward(self):
        base = self.base(trained=True).eval()
        self.assertFalse(any(isinstance(layer, nn.modules.batchnorm._BatchNorm) for layer in base.modules()))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = SegmentedTCN(base)(self.features, self.a_logits, [True, False, True, True, False, True])
        self.assertEqual(output.shape, self.a_logits.shape)
        self.assertTrue(torch.isfinite(output).all().item())

    def test_invalid_config_inputs_masks_and_base_outputs_are_rejected(self):
        for kwargs in ({"input_dim": 0}, {"width": -1}, {"layers": True},
                       {"kernel_size": 2}, {"dropout": float("nan")}, {"dropout": 1}):
            with self.assertRaises(ValueError):
                TCNConfig(**({"input_dim": 8} | kwargs))
        wrapper = SegmentedTCN(self.base())
        for gates in ([1] * 6, [True] * 5, torch.ones(6), torch.ones(2, 3, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                wrapper(self.features, self.a_logits, gates)
        for features, logits in ((self.features[:0], self.a_logits[:0]),
                                 (self.features, self.a_logits[:2]),
                                 (self.features.long(), self.a_logits),
                                 (self.features * float("inf"), self.a_logits),
                                 (self.features, self.a_logits * float("nan"))):
            with self.assertRaises(ValueError):
                wrapper(features, logits, [True] * 6)
        with self.assertRaises(ValueError):
            wrapper.base(torch.randn(7, 9), self.a_logits)
        with self.assertRaises(ValueError):
            wrapper.base(self.features, torch.randn(7, 6))
        # Boolean tensors are supported without numeric-mask coercion.
        torch.testing.assert_close(wrapper(self.features, self.a_logits, torch.ones(6, dtype=torch.bool)),
                                   self.a_logits, rtol=0, atol=0)

        class InvalidBase(nn.Module):
            def forward(self, features, a_logits):
                return a_logits[:1]

        with self.assertRaises(ValueError):
            SegmentedTCN(InvalidBase())(self.features, self.a_logits, [True] * 6)


class FSNTCNGatesTest(unittest.TestCase):
    def setUp(self):
        self.eligible = [True, False, True, True, True, True, False, True]
        self.kinds = ["clinical"] * 4 + ["network"] * 4
        self.scores = [.99, .99, .3, .97, .8, .91, .99, .2]

    def test_all_strategies_preserve_structural_cuts_and_threshold_is_inclusive(self):
        expected = {"all": self.eligible, "source_rule": [True, False, True, True, False, False, False, False],
                    "learned": [True, False, False, True, False, True, False, False]}
        for strategy in ("all", "source_rule", "learned", "random"):
            gates = select_gates(strategy, self.eligible, self.kinds, self.scores)
            self.assertTrue(all(not flag or eligible for flag, eligible in zip(gates, self.eligible)))
            if strategy in expected:
                self.assertEqual(gates, expected[strategy])
        self.assertEqual(select_gates("learned", [True], ["clinical"], [.9]), [True])

    def test_random_matches_counts_per_source_and_is_reproducible_without_global_rng(self):
        learned = select_gates("learned", self.eligible, self.kinds, self.scores)
        random = select_gates("random", self.eligible, self.kinds, self.scores, seed=101, chain_id="record-1")
        self.assertEqual(random, select_gates("random", self.eligible, self.kinds, self.scores,
                                             seed=101, chain_id="record-1"))
        for kind in set(self.kinds):
            self.assertEqual(sum(flag for flag, source in zip(random, self.kinds) if source == kind),
                             sum(flag for flag, source in zip(learned, self.kinds) if source == kind))
        torch.manual_seed(9000)
        self.assertEqual(random, select_gates("random", self.eligible, self.kinds, self.scores,
                                             seed=101, chain_id="record-1"))

    def test_gate_validation_empty_chain_and_mixed_source(self):
        self.assertEqual(select_gates("all", [], []), [])
        self.assertEqual(select_gates("random", [], [], []), [])
        self.assertEqual(select_gates("source_rule", [True], ["mixed"]), [False])
        for kwargs in ({"strategy": "bad"}, {"eligible": [1] * 8}, {"source_kinds": ["clinical"]},
                       {"source_kinds": [""] * 8}, {"scores": None},
                       {"scores": [float("nan")] * 8}, {"scores": [1.1] * 8},
                       {"threshold": float("inf")}, {"threshold": True}, {"seed": True}):
            base = {"strategy": "learned", "eligible": self.eligible, "source_kinds": self.kinds,
                    "scores": self.scores}
            with self.assertRaises(ValueError):
                select_gates(**(base | kwargs))


if __name__ == "__main__":
    unittest.main()
