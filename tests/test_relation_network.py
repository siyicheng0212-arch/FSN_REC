"""Validate relation evidence contracts and meaningful gradient isolation."""

import unittest

import torch
import torch.nn.functional as F

from experiments.relation.network import RelationConfig, RelationNet


class RelationNetworkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(41)
        self.left = self.evidence()
        self.right = self.evidence(global_steps=6, local_steps=3)

    @staticmethod
    def evidence(batch=3, global_steps=4, local_steps=5, gradients=False):
        return {
            "global_tokens": torch.randn(batch, global_steps, 12, requires_grad=gradients),
            "local_tokens": torch.randn(batch, local_steps, 20, requires_grad=gradients),
            "global_positions": torch.linspace(0, 1, global_steps).expand(batch, -1).clone().requires_grad_(gradients),
            "local_positions": torch.linspace(0, 1, local_steps).expand(batch, -1).clone().requires_grad_(gradients),
        }

    @staticmethod
    def model(mode="dual", dropout=.1):
        return RelationNet(RelationConfig(global_dim=12, local_dim=20, dim=8, heads=2,
                                          endpoint_tokens=2, dropout=dropout, mode=mode))

    def test_binary_logit_shape_eval_determinism_and_variable_token_counts(self):
        for mode in ("mlp", "dual"):
            model = self.model(mode).eval()
            actual = model(self.left, self.right)
            self.assertEqual(actual.shape, (3,))
            self.assertTrue(torch.isfinite(actual).all().item())
            torch.testing.assert_close(actual, model(self.left, self.right), rtol=0, atol=0)
            # Fewer available tokens than endpoint_tokens remain legal.
            singleton = self.evidence(batch=1, global_steps=1, local_steps=1)
            self.assertEqual(model(singleton, singleton).shape, (1,))

    def test_gradients_reach_every_active_relation_parameter_but_not_visual_evidence(self):
        left, right = self.evidence(gradients=True), self.evidence(gradients=True)
        for mode in ("mlp", "dual"):
            model = self.model(mode, dropout=0)
            optimizer = torch.optim.SGD(model.parameters(), lr=.05)
            before = model.classifier[-1].weight.detach().clone()
            logits = model(left, right)
            loss = F.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0.]))
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, (mode, name))
                self.assertTrue(torch.isfinite(parameter.grad).all().item(), (mode, name))
                self.assertGreater(parameter.grad.abs().sum().item(), 0, (mode, name))
            for side in (left, right):
                for value in side.values():
                    self.assertIsNone(value.grad)
            optimizer.step()
            self.assertFalse(torch.equal(before, model.classifier[-1].weight.detach()))

    def test_direction_is_preserved_without_claiming_clinical_correctness(self):
        for mode in ("mlp", "dual"):
            model = self.model(mode).eval()
            forward = model(self.left, self.right)
            backward = model(self.right, self.left)
            self.assertFalse(torch.allclose(forward, backward))

    def test_dual_reads_ordered_sample_positions_while_scene_mlp_is_position_invariant(self):
        changed = dict(self.left)
        changed["global_positions"] = self.left["global_positions"] * .5
        for mode in ("mlp", "dual"):
            model = self.model(mode).eval()
            actual, alternative = model(self.left, self.right), model(changed, self.right)
            if mode == "mlp":
                torch.testing.assert_close(actual, alternative, rtol=0, atol=0)
            else:
                self.assertFalse(torch.allclose(actual, alternative))

    def test_batch_outputs_do_not_read_other_pairs(self):
        for mode in ("mlp", "dual"):
            model = self.model(mode).eval()
            batched = model(self.left, self.right)
            independent = torch.cat([model({k: v[i:i + 1] for k, v in self.left.items()},
                                           {k: v[i:i + 1] for k, v in self.right.items()})
                                     for i in range(3)])
            torch.testing.assert_close(batched, independent, rtol=1e-5, atol=1e-6)

    def test_missing_nonfinite_unordered_or_wrong_shape_evidence_is_rejected(self):
        model = self.model()
        bad_cases = []
        missing = dict(self.left)
        missing.pop("local_tokens")
        bad_cases.append(missing)
        for key in ("global_tokens", "local_tokens", "global_positions", "local_positions"):
            nonfinite = dict(self.left)
            nonfinite[key] = self.left[key].clone()
            nonfinite[key].view(-1)[0] = float("nan")
            bad_cases.append(nonfinite)
        bad_cases.extend([
            dict(self.left, global_positions=self.left["global_positions"].flip(1)),
            dict(self.left, local_positions=self.left["local_positions"] + .1),
            dict(self.left, local_tokens=torch.randn(3, 5, 19)),
            dict(self.left, local_tokens=torch.randn(2, 5, 20)),
            dict(self.left, global_positions=torch.ones(3, 3)),
            dict(self.left, global_tokens=torch.randn(3, 0, 12)),
        ])
        for bad in bad_cases:
            with self.assertRaises(ValueError):
                model(bad, self.right)
        with self.assertRaises(ValueError):
            model(self.left, self.evidence(batch=2))
        repeated = dict(self.left, global_positions=torch.zeros_like(self.left["global_positions"]))
        self.assertTrue(torch.isfinite(model(repeated, self.right)).all().item())

    def test_config_and_cpu_bfloat16(self):
        for kwargs in ({"mode": "unknown"}, {"dim": 7}, {"heads": 0},
                       {"endpoint_tokens": 0}, {"dropout": 1}, {"global_dim": -1}):
            with self.assertRaises(ValueError):
                RelationConfig(**kwargs)
        model = self.model().eval()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = model(self.left, self.right)
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertEqual(actual.shape, (3,))


if __name__ == "__main__":
    unittest.main()
