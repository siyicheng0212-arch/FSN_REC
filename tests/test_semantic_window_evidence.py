"""CPU tests for the opt-in semantic corroboration model."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from experiments.semantic_window_evidence import (
    FSNSemanticWindowEvidence,
    SemanticWindowCorroborator,
    time_bin_margins,
)


class _FakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.new_fc = nn.Linear(3, 7, bias=False)
        self.calls = 0

    def forward(self, frames):
        self.calls += 1
        feature = frames.mean(dim=(-1, -2)).reshape(-1, 3)
        return self.new_fc(feature)


class _FakeCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_CNN = _FakeCNN()
        self.local_CNN = _FakeCNN()


class _FakeBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = _FakeCore()
        self.num_glance_segments = 8
        self.num_input_focus_segments = 36
        self.num_focus_segments = 12

    def forward(self, frames):
        batch = frames.shape[0]
        glance = torch.linspace(0, 35, 8, device=frames.device).round().long()
        focus = torch.tensor(
            [0, 2, 5, 8, 11, 14, 18, 21, 24, 28, 32, 35],
            device=frames.device,
        )
        global_votes = self.core.global_CNN(frames[:, glance]).view(
            batch, 8, 7
        )
        local_votes = self.core.local_CNN(frames[:, focus]).view(
            batch, 12, 7
        )
        logits = global_votes.mean(1) + local_votes.mean(1)
        flat_focus = focus[None].expand(batch, -1)
        flat_focus = flat_focus + (
            torch.arange(batch, device=frames.device) * 36
        )[:, None]
        eval_outputs = (logits,) + (None,) * 7 + (flat_focus.reshape(-1),)
        return {"logits": logits, "eval_outputs": eval_outputs}


class SemanticWindowEvidenceTest(unittest.TestCase):
    def test_time_bins_and_missing_counts(self):
        logits = torch.zeros(1, 4, 7)
        logits[0, :, 3] = torch.tensor((1.0, 3.0, -2.0, -4.0))
        positions = torch.tensor([[0.0, 0.2, 0.7, 1.0]])
        margins, counts = time_bin_margins(logits, positions)
        self.assertEqual(counts.tolist(), [[2, 0, 2]])
        self.assertTrue(torch.allclose(margins, torch.tensor([[2.0, 0.0, -3.0]])))

    def test_zero_initialization_and_repeated_frame_fallback(self):
        head = SemanticWindowCorroborator()
        global_logits = torch.zeros(1, 6, 7)
        local_logits = torch.zeros(1, 6, 7)
        global_logits[..., 3] = 2
        local_logits[..., 3] = 2
        positions = torch.tensor([[0.1, 0.2, 0.4, 0.5, 0.8, 0.9]])
        baseline = torch.zeros(1, 7)
        baseline[0, 4] = 2
        repeated = torch.zeros(1, 36, 3, 8, 8)
        output = head(
            global_logits, local_logits, positions, positions, baseline, repeated
        )
        self.assertTrue(torch.equal(output["correction"], torch.zeros_like(baseline)))
        self.assertFalse(bool(output["corroboration_gate"][0]))
        with torch.no_grad():
            head.alpha.fill_(1)
        output = head(
            global_logits, local_logits, positions, positions, baseline, repeated
        )
        self.assertTrue(torch.equal(output["correction"], torch.zeros_like(baseline)))
        distinct = torch.linspace(0, 0.2, 36).view(1, 36, 1, 1, 1)
        distinct = distinct.expand(1, 36, 3, 8, 8)
        output = head(
            global_logits, local_logits, positions, positions, baseline, distinct
        )
        self.assertTrue(bool(output["corroboration_gate"][0]))
        self.assertGreater(float(output["correction"][0, 3]), 0)
        self.assertLess(float(output["correction"][0, 4]), 0)
        self.assertTrue(torch.equal(output["correction"][0, :3], torch.zeros(3)))
        self.assertTrue(torch.equal(output["correction"][0, 5:], torch.zeros(2)))

    def test_wrapper_preserves_original_and_uses_no_extra_backbone_calls(self):
        torch.manual_seed(4)
        baseline = _FakeBaseline()
        model = FSNSemanticWindowEvidence(baseline)
        frames = torch.rand(2, 36, 3, 8, 8) * 0.2
        output = model(frames)
        self.assertTrue(torch.equal(output["logits"], output["baseline_logits"]))
        self.assertEqual(baseline.core.global_CNN.calls, 1)
        self.assertEqual(baseline.core.local_CNN.calls, 1)
        self.assertEqual(len(baseline.core.global_CNN.new_fc._forward_hooks), 0)
        self.assertEqual(len(baseline.core.local_CNN.new_fc._forward_hooks), 0)
        self.assertTrue(all(not p.requires_grad for p in baseline.parameters()))
        model.train()
        self.assertFalse(baseline.training)
        model.enabled = False
        self.assertTrue(torch.equal(model(frames)["logits"], model(frames)["baseline_logits"]))

    def test_pair_auxiliary_and_alpha_receive_gradients(self):
        torch.manual_seed(5)
        baseline = _FakeBaseline()
        model = FSNSemanticWindowEvidence(baseline)
        model.set_class_weights(torch.ones(7), torch.tensor([1.0, 2.0]))
        frames = torch.rand(2, 36, 3, 8, 8) * 0.2
        with torch.no_grad():
            baseline.core.global_CNN.new_fc.weight.zero_()
            baseline.core.local_CNN.new_fc.weight.zero_()
            baseline.core.global_CNN.new_fc.weight[3].fill_(2)
            baseline.core.local_CNN.new_fc.weight[3].fill_(2)
        output = model(frames)
        loss = model.compute_loss(output, torch.tensor([3, 4]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.evidence_module.alpha.grad)
        self.assertIsNotNone(model.evidence_module.calibrator[-1].weight.grad)
        self.assertTrue(torch.isfinite(model.evidence_module.alpha.grad))
        self.assertTrue(torch.isfinite(model.evidence_module.calibrator[-1].weight.grad).all())


if __name__ == "__main__":
    unittest.main()
