"""Operator and gradient tests; synthetic legacy-shaped fixtures are NOT a historical reproduction."""

import copy
import json
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from experiments.context_utility.model import ContextUtilityTCN
from experiments.fsn_tcn.model import TCNConfig, TemporalResidualTCN


class SyntheticLegacyFixture(nn.Module):
    """Test-only three-block scorer consuming visuals AND A predictions."""
    def __init__(self, *, groups=1, dilation=1, kernel=3, dropout=0, fail=False):
        super().__init__()
        self.input = nn.Conv1d(8 + 7, 8, 1)
        self.blocks = nn.ModuleList([
            nn.Conv1d(8, 8, kernel, padding=dilation * (kernel - 1) // 2,
                      dilation=dilation, groups=groups)
            for _ in range(3)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Conv1d(8, 7, 1)
        self.fail = fail

    def forward(self, features, a_logits):
        values = self.input(torch.cat((features, a_logits), -1).T.unsqueeze(0))
        for block in self.blocks:
            values = values + self.dropout(torch.tanh(block(values)))
        if self.fail:
            raise RuntimeError("fixture failure")
        return self.output(values).squeeze(0).T


class SingleConvFixture(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.temporal = conv

    def forward(self, features, a_logits):
        return self.temporal(features.T.unsqueeze(0)).squeeze(0).T


class UtilityModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(765)
        self.features = torch.randn(11, 8)
        self.logits = torch.randn(11, 7)

    @staticmethod
    def paths():
        return [f"blocks.{i}" for i in range(3)]

    def test_initialization_exactly_recovers_actual_base_including_groups_and_dilation(self):
        for dtype in (torch.float32, torch.float64):
            for groups, dilation, kernel in ((1, 1, 3), (2, 2, 3), (4, 3, 5)):
                with self.subTest(dtype=dtype, groups=groups, dilation=dilation, kernel=kernel):
                    base = SyntheticLegacyFixture(groups=groups, dilation=dilation, kernel=kernel).to(dtype).eval()
                    reference = copy.deepcopy(base)
                    model = ContextUtilityTCN(base, 8, self.paths(), dim=5).to(dtype).eval()
                    features, logits = self.features.to(dtype), self.logits.to(dtype)
                    torch.testing.assert_close(model(features, logits), reference(features, logits), rtol=0, atol=0)
                    torch.testing.assert_close(model(features, logits, gate_mode="unit"),
                                               reference(features, logits), rtol=0, atol=0)
                    self.assertGreater(model.last_gate_audit()["total_coefficients"], 0)
                    for group in model.last_gate_audit()["groups"]:
                        self.assertEqual(group["min"], 1)
                        self.assertEqual(group["max"], 1)

    def test_three_modes_retain_base_singleton_policy_and_plain_module_identity(self):
        for mode in ("plain", "scalar", "dynamic"):
            base = SyntheticLegacyFixture().eval()
            reference = copy.deepcopy(base)
            original_conv = base.blocks[0]
            wrapper = ContextUtilityTCN(base, 8, self.paths(), mode=mode, dim=5).eval()
            actual = wrapper(self.features[:1], self.logits[:1])
            torch.testing.assert_close(actual, reference(self.features[:1], self.logits[:1]), rtol=0, atol=0)
            self.assertFalse(torch.equal(actual, self.logits[:1]))
            if mode == "plain":
                self.assertIs(wrapper.base.blocks[0], original_conv)
                self.assertEqual(wrapper.added_parameter_count(), 0)
            elif mode == "scalar":
                self.assertEqual(wrapper.added_parameter_count(), 1)

    def test_new_reference_keeps_its_own_a_singleton_policy(self):
        base = TemporalResidualTCN(TCNConfig(input_dim=8, width=8, layers=4, dropout=0)).eval()
        nn.init.normal_(base.output_projection.weight, std=.2)
        reference = copy.deepcopy(base)
        paths = [f"blocks.{i}.temporal" for i in range(4)]
        wrapper = ContextUtilityTCN(base, 8, paths, dim=5).eval()
        torch.testing.assert_close(wrapper(self.features, self.logits), reference(self.features, self.logits),
                                   rtol=0, atol=0)
        singleton_logits = self.logits[:1]
        self.assertIs(wrapper(self.features[:1], singleton_logits), singleton_logits)
        self.assertEqual(wrapper.last_gate_audit()["total_coefficients"], 0)

    def test_center_and_bias_are_not_scaled_and_offcenter_terms_are(self):
        conv = nn.Conv1d(7, 7, 3, padding=2, dilation=2, groups=7).double()
        with torch.no_grad():
            conv.weight[:, 0, :] = torch.tensor([1., 3., 5.], dtype=torch.float64)
            conv.bias.fill_(7)
        model = ContextUtilityTCN(SingleConvFixture(conv), 7, ["temporal"], mode="scalar").double()
        with torch.no_grad():
            model.scalar_logit.fill_(torch.logit(torch.tensor(.75, dtype=torch.float64)))
        inputs = torch.arange(9., dtype=torch.float64).unsqueeze(1).expand(-1, 7)
        logits = torch.zeros(9, 7, dtype=torch.float64)
        padded = F.pad(inputs.T.unsqueeze(0), (2, 2))
        expected = 3 * inputs + 7 + 1.5 * (padded[:, :, :9].squeeze(0).T +
                                         5 * padded[:, :, 4:13].squeeze(0).T)
        torch.testing.assert_close(model(inputs, logits), expected, rtol=1e-14, atol=1e-14)
        self.assertEqual({group["offset"] for group in model.last_gate_audit()["groups"]}, {-2, 2})
        self.assertEqual(model.last_gate_audit()["total_coefficients"], 14)

    def test_gate_uses_ordered_visual_evidence_without_a_scores(self):
        model = ContextUtilityTCN(SyntheticLegacyFixture(), 8, self.paths(), dim=5).eval()
        with torch.no_grad():
            model.gate.network[-1].weight.normal_(std=.3)
        model(self.features, self.logits)
        before = model.last_coefficients()
        model(self.features, self.logits * 100 + 50)
        after = model.last_coefficients()
        for left, right in zip(before, after):
            torch.testing.assert_close(left["values"], right["values"], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(left["values"], right["values"])
                            for left, right in zip(before[::2], before[1::2])))

    def test_permuted_coefficients_keep_each_layer_and_offset_distribution(self):
        model = ContextUtilityTCN(SyntheticLegacyFixture(), 8, self.paths(), dim=5).eval()
        with torch.no_grad():
            model.gate.network[-1].weight.normal_(std=.3)
        model(self.features, self.logits)
        before = model.last_coefficients()
        rng_before = torch.random.get_rng_state().clone()
        model(self.features, self.logits, gate_mode="permuted", permutation_seed=202)
        after = model.last_coefficients()
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
        for original, permuted in zip(before, after):
            torch.testing.assert_close(original["values"].sort().values, permuted["values"].sort().values,
                                       rtol=0, atol=0)
        audit = model.last_gate_audit()
        self.assertGreater(audit["changed_positions_permuted"], 0)
        self.assertGreater(audit["shuffled_positions"], 0)
        self.assertNotIn("target", json.dumps(audit))
        again = model(self.features, self.logits, gate_mode="permuted", permutation_seed=202)
        same = model(self.features, self.logits, gate_mode="permuted", permutation_seed=202)
        torch.testing.assert_close(again, same, rtol=0, atol=0)
        # Zero-initialized identical values must be reported as ZERO changed
        # coefficients even if a permutation changes their source indices.
        initial = ContextUtilityTCN(SyntheticLegacyFixture(), 8, self.paths(), dim=5).eval()
        initial(self.features, self.logits, gate_mode="permuted", permutation_seed=202)
        self.assertEqual(initial.last_gate_audit()["changed_positions_permuted"], 0)

    def test_gate_last_layer_updates_first_then_projection_receives_gradients(self):
        for mode in ("dynamic", "scalar"):
            with self.subTest(mode=mode):
                model = ContextUtilityTCN(SyntheticLegacyFixture(), 8, self.paths(), mode=mode, dim=5).train()
                optimizer = torch.optim.SGD(model.parameters(), lr=.05)
                labels = torch.arange(11) % 7
                before = {name: value.clone() for name, value in model.named_parameters() if not name.startswith("base.")}
                for step in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(self.features, self.logits), labels)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss).item())
                    if mode == "dynamic":
                        self.assertGreater(model.gate.network[-1].weight.grad.abs().sum().item(), 0)
                        if step == 0:
                            self.assertEqual(model.gate.projection[0].weight.grad.abs().sum().item(), 0)
                        else:
                            self.assertGreater(model.gate.projection[0].weight.grad.abs().sum().item(), 0)
                    else:
                        self.assertGreater(model.scalar_logit.grad.abs().sum().item(), 0)
                    optimizer.step()
                self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.named_parameters()
                                    if name in before))

    def test_gate_initialization_does_not_consume_cpu_or_cuda_randomness(self):
        base = SyntheticLegacyFixture()
        cpu_state = torch.random.get_rng_state().clone()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        model = ContextUtilityTCN(base, 8, self.paths(), dim=5, gate_seed=999)
        torch.testing.assert_close(torch.random.get_rng_state(), cpu_state, rtol=0, atol=0)
        if cuda_states:
            for before, after in zip(cuda_states, torch.cuda.get_rng_state_all()):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertGreater(model.added_parameter_count(), 0)

    def test_dropout_randomness_and_train_output_are_preserved_at_initialization(self):
        base = SyntheticLegacyFixture(dropout=.4).train()
        reference = copy.deepcopy(base)
        model = ContextUtilityTCN(base, 8, self.paths(), dim=5).train()
        torch.manual_seed(18)
        expected = reference(self.features, self.logits)
        torch.manual_seed(18)
        actual = model(self.features, self.logits)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_checkpoint_roundtrip_and_deepcopy_share_only_copied_context(self):
        model = ContextUtilityTCN(SyntheticLegacyFixture(), 8, self.paths(), dim=5).eval()
        with torch.no_grad():
            model.gate.network[-1].weight.normal_(std=.3)
        restored = ContextUtilityTCN(SyntheticLegacyFixture(), 8, self.paths(), dim=5).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        cloned = copy.deepcopy(model)
        expected = model(self.features, self.logits)
        torch.testing.assert_close(restored(self.features, self.logits), expected, rtol=0, atol=0)
        torch.testing.assert_close(cloned(self.features, self.logits), expected, rtol=0, atol=0)
        self.assertIs(cloned.base.blocks[0].context, cloned._context)
        self.assertIsNot(cloned._context, model._context)

    def test_cpu_bfloat16_preserves_native_base(self):
        base = SyntheticLegacyFixture().eval()
        reference = copy.deepcopy(base)
        model = ContextUtilityTCN(base, 8, self.paths(), dim=5).eval()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(self.features, self.logits)
            expected = reference(self.features, self.logits)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_constructor_rejects_unsupported_semantics_instead_of_repairing(self):
        convs = [nn.Conv1d(8, 7, 1), nn.Conv1d(8, 7, 4, padding=2),
                 nn.Conv1d(8, 7, 3, stride=2, padding=1), nn.Conv1d(8, 7, 3, padding=2),
                 nn.Conv1d(8, 7, 3, padding=1, padding_mode="reflect")]
        for conv in convs:
            with self.assertRaises(ValueError):
                ContextUtilityTCN(SingleConvFixture(conv), 8, ["temporal"])
        weighted = nn.utils.weight_norm(nn.Conv1d(8, 7, 3, padding=1))
        with self.assertRaises(ValueError):
            ContextUtilityTCN(SingleConvFixture(weighted), 8, ["temporal"])
        for paths in (["blocks.99"], ["blocks.0", "blocks.0"], [], "blocks.0"):
            with self.assertRaises(ValueError):
                ContextUtilityTCN(SyntheticLegacyFixture(), 8, paths)
        for kwargs in ({"feature_dim": True}, {"dim": 0}, {"gate_seed": True}, {"mode": "other"}):
            with self.assertRaises(ValueError):
                ContextUtilityTCN(SyntheticLegacyFixture(), temporal_paths=self.paths(),
                                  **({"feature_dim": 8} | kwargs))

    def test_exception_cleans_active_context_and_diagnostic_state(self):
        base = SyntheticLegacyFixture(fail=True)
        model = ContextUtilityTCN(base, 8, self.paths(), dim=5)
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            model(self.features, self.logits)
        self.assertFalse(model._context.active)
        self.assertEqual(model._context.coefficients, {})
        self.assertEqual(model.last_gate_audit()["total_coefficients"], 0)
        base.fail = False
        self.assertTrue(torch.isfinite(model(self.features, self.logits)).all().item())
        for features, logits, kwargs in ((self.features[:0], self.logits[:0], {}),
                                         (self.features, self.logits[:1], {}),
                                         (self.features * float("nan"), self.logits, {}),
                                         (self.features, self.logits, {"gate_mode": "bad"}),
                                         (self.features, self.logits, {"permutation_seed": True})):
            with self.assertRaises(ValueError):
                model(features, logits, **kwargs)
            self.assertEqual(model.last_gate_audit()["total_coefficients"], 0)

    def test_selected_but_unused_temporal_module_is_rejected(self):
        class WrongPathFixture(SyntheticLegacyFixture):
            def forward(self, features, a_logits):
                return a_logits
        model = ContextUtilityTCN(WrongPathFixture(), 8, self.paths(), dim=5)
        with self.assertRaisesRegex(ValueError, "not called"):
            model(self.features, self.logits)


if __name__ == "__main__":
    unittest.main()
