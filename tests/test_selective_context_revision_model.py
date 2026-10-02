"""Behavioral parity and training tests for historical-TCN correction gate."""

import inspect
import unittest

import torch
from torch import nn

from experiments.selective_context.revision_model import RevisionGateTCN


class _HistoricalScorer(nn.Module):
    """Includes dropout, stateful call count, and non-A singleton behavior."""

    def __init__(self):
        super().__init__()
        self.temporal = nn.Conv1d(3, 7, 3, padding=1)
        self.dropout = nn.Dropout(p=0.5)
        self.calls = 0

    def forward(self, features, a_logits):
        self.calls += 1
        delta = self.temporal(self.dropout(features).T.unsqueeze(0)).squeeze(0).T
        return a_logits + delta


class RevisionGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)
        self.model = RevisionGateTCN(_HistoricalScorer(), feature_dim=3,
                                     gate_dim=8, freeze_base=True)
        self.features = torch.randn(4, 3)
        self.a_logits = torch.randn(4, 7)

    def test_unit_is_bitwise_historical_for_full_chain_and_singleton(self):
        for length in (4, 1):
            features, a_logits = self.features[:length], self.a_logits[:length]
            self.model.train()
            self.assertFalse(self.model.base_tcn.training)
            with torch.no_grad():
                reference = self.model.base_tcn(features, a_logits)
                before = self.model.base_tcn.calls
                output, details = self.model(features, a_logits,
                                             gate_mode="unit", return_details=True)
            self.assertEqual(self.model.base_tcn.calls, before + 1)
            self.assertTrue(torch.equal(output, reference))
            self.assertTrue(torch.equal(details["tcn_logits"], reference))
            self.assertTrue(torch.equal(details["alpha"], torch.ones(length)))

    def test_zero_is_bitwise_a_and_still_calls_full_old_base(self):
        before = self.model.base_tcn.calls
        result = self.model(self.features, self.a_logits, gate_mode="zero")
        self.assertTrue(torch.equal(result, self.a_logits))
        self.assertEqual(self.model.base_tcn.calls, before + 1)

    def test_gate_reads_per_clip_own_evidence_and_old_result(self):
        # The gate itself has no cross-clip operation.  The old TCN *does*
        # read neighbors, which is the intended full-chain behavior.
        self.model.eval()
        old = self.model.base_tcn(self.features, self.a_logits)
        alpha = self.model.gate(self.features, self.a_logits, old)
        mutated = self.features.clone()
        mutated[1:] += 40
        # Supply fixed old scores to isolate the gate from old TCN context.
        alpha_mutated = self.model.gate(mutated, self.a_logits, old)
        self.assertTrue(torch.equal(alpha[:1], alpha_mutated[:1]))

    def test_train_step_updates_gate_not_frozen_base(self):
        before = self.model.gate.network[-1].weight.detach().clone()
        old_weight = self.model.base_tcn.temporal.weight.detach().clone()
        optim = torch.optim.SGD(self.model.gate.parameters(), lr=0.1)
        self.model.train()
        result, details = self.model(self.features, self.a_logits, return_details=True)
        self.assertEqual(result.shape, (4, 7))
        self.assertTrue(bool(((details["alpha"] > 0) & (details["alpha"] < 1)).all()))
        loss = nn.functional.cross_entropy(result, torch.tensor([0, 1, 2, 3]))
        loss.backward()
        self.assertGreater(float(self.model.gate.network[-1].weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in self.model.base_tcn.parameters()))
        optim.step()
        self.assertFalse(torch.equal(before, self.model.gate.network[-1].weight))
        self.assertTrue(torch.equal(old_weight, self.model.base_tcn.temporal.weight))

    def test_no_source_relation_edge_label_interface(self):
        signature = set(inspect.signature(RevisionGateTCN.forward).parameters)
        self.assertEqual(signature, {"self", "features", "a_logits", "gate_mode", "return_details"})
        for argument in ("labels", "source", "edge_logits", "open_edges"):
            with self.assertRaises(TypeError):
                self.model(self.features, self.a_logits, **{argument: 0})

    def test_shared_scalar_and_class_ablation_counts(self):
        for variant, count in (("scalar", 1), ("class_conditioned", 7)):
            model = RevisionGateTCN(_HistoricalScorer(), 3, gate_dim=8, variant=variant)
            self.assertEqual(model.added_parameter_count(), count)
            values, audit = model(self.features, self.a_logits, return_details=True)
            self.assertEqual(values.shape, (4, 7))
            self.assertEqual(audit["alpha"].shape, (4,))
            # The baseline scorer is identical in every ablation mode.
            expected = model.base_tcn(self.features, self.a_logits)
            self.assertTrue(torch.equal(model(self.features, self.a_logits,
                                               gate_mode="unit"), expected))

    def test_logits_only_discards_visual_features_at_gate(self):
        model = RevisionGateTCN(_HistoricalScorer(), 3, gate_dim=8,
                                variant="logits_only").eval()
        self.assertIsNone(model.gate.visual)
        old = model.base_tcn(self.features, self.a_logits)
        alpha = model.gate(self.features, self.a_logits, old)
        alpha_other_visual = model.gate(self.features + 100, self.a_logits, old)
        self.assertTrue(torch.equal(alpha, alpha_other_visual))

    def test_class_condition_uses_a_prediction_not_ground_truth(self):
        model = RevisionGateTCN(_HistoricalScorer(), 3, gate_dim=8,
                                variant="class_conditioned").eval()
        with torch.no_grad():
            model.class_logits.copy_(torch.arange(7, dtype=torch.float32))
        _, audit = model(self.features, self.a_logits, return_details=True)
        expected = model.class_logits[self.a_logits.argmax(-1)].sigmoid()
        self.assertTrue(torch.equal(audit["alpha"], expected))


if __name__ == "__main__":
    unittest.main()
