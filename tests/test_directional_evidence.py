"""Contracts for the opt-in within-clip directional-evidence candidate."""

import unittest

import torch
from torch import nn
import torch.nn.functional as F

from experiments.directional_evidence import (
    DirectionalClipEvidence,
    FSNDirectionalEvidence,
    summarize_window_activity,
)


class TinyBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 7)

    def forward(self, frames):
        return {"logits": self.head(frames.mean(dim=(1, 3, 4)))}

    def set_class_weights(self, weights):
        self.class_weights = weights

    def compute_loss(self, output, target):
        return F.cross_entropy(output["logits"], target)


class TrainingBranchBaseline(TinyBaseline):
    def forward(self, frames):
        logits = super().forward(frames)["logits"]
        return {
            "logits": logits,
            "random_branch": (logits,),
            "policy_branch": (logits,),
        }

    def compute_loss(self, output, target):
        return sum(F.cross_entropy(output[name][0], target)
                   for name in ("random_branch", "policy_branch"))


class DirectionalEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(min(6, torch.get_num_threads()))

    def test_initial_original_equivalence_and_trainable_margin(self):
        torch.manual_seed(17)
        model = FSNDirectionalEvidence(TinyBaseline())
        frames = torch.rand(2, 36, 3, 224, 224)
        output = model(frames)
        self.assertTrue(torch.equal(output["logits"], output["baseline_logits"]))
        self.assertTrue(torch.count_nonzero(output["directional_correction"]) == 0)
        self.assertEqual(tuple(output["directional_evidence"]["activity_window_probabilities"].shape), (2, 2, 9))
        model.compute_loss(output, torch.tensor([3, 4])).backward()
        grad = model.evidence_module.margin_head[-1].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_repeated_frames_leave_original_unchanged_after_nonzero_head(self):
        model = FSNDirectionalEvidence(TinyBaseline())
        nn.init.constant_(model.evidence_module.margin_head[-1].bias, 2.0)
        still = torch.rand(1, 1, 3, 224, 224).repeat(1, 36, 1, 1, 1)
        output = model(still)
        self.assertEqual(float(output["directional_evidence"]["temporal_reliability"][0]), 0.0)
        self.assertTrue(torch.equal(output["logits"], output["baseline_logits"]))

    def test_encoder_receives_gradient_after_zero_head_opens(self):
        torch.manual_seed(29)
        model = FSNDirectionalEvidence(TinyBaseline())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        frames = torch.rand(2, 36, 3, 224, 224)
        targets = torch.tensor([3, 4])
        model.compute_loss(model(frames), targets).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.compute_loss(model(frames), targets).backward()
        grad = model.evidence_module.encoder[0].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_correction_is_pair_specific_and_reaches_native_training_branches(self):
        model = FSNDirectionalEvidence(TrainingBranchBaseline())
        nn.init.constant_(model.evidence_module.margin_head[-1].bias, 2.0)
        output = model(torch.rand(1, 36, 3, 224, 224))
        correction = output["directional_correction"]
        self.assertTrue(torch.equal(correction[:, :3], torch.zeros_like(correction[:, :3])))
        self.assertTrue(torch.equal(correction[:, 5:], torch.zeros_like(correction[:, 5:])))
        self.assertTrue(torch.allclose(correction[:, 3], -correction[:, 4]))
        self.assertGreater(float(correction[:, 3].abs().sum()), 0.0)
        self.assertTrue(torch.equal(output["logits"], output["random_branch"][0]))
        self.assertTrue(torch.equal(output["logits"], output["policy_branch"][0]))

    def test_direction_and_cooccurrence_are_distinct(self):
        activity = torch.zeros(1, 2, 9)
        activity[0, 0, 2] = 1.0
        activity[0, 1, 3] = 1.0
        activity[0, 0, 5] = 1.0
        activity[0, 1, 5] = 1.0
        summary = summarize_window_activity(activity)
        self.assertGreater(float(summary[0, 5]), float(summary[0, 6]))
        self.assertGreater(float(summary[0, 4]), 0.0)
        with self.assertRaisesRegex(ValueError, "two activity streams"):
            summarize_window_activity(torch.zeros(1, 7, 9))

    def test_unordered_and_quality_gate_ablations_are_opt_in(self):
        with self.assertRaisesRegex(ValueError, "relation_mode"):
            DirectionalClipEvidence(relation_mode="invalid")
        model = FSNDirectionalEvidence(
            TinyBaseline(), relation_mode="unordered", quality_gate=False
        )
        nn.init.constant_(model.evidence_module.margin_head[-1].bias, 1.0)
        still = torch.rand(1, 1, 3, 224, 224).repeat(1, 36, 1, 1, 1)
        out = model(still)
        self.assertEqual(float(out["directional_evidence"]["temporal_reliability"][0]), 0.0)
        self.assertGreater(float(out["directional_correction"].abs().sum()), 0.0)
        self.assertEqual(model.evidence_module.relation_mode, "unordered")

    def test_batch_order_is_independent(self):
        model = FSNDirectionalEvidence(TinyBaseline()).eval()
        nn.init.constant_(model.evidence_module.margin_head[-1].bias, 1.0)
        frames = torch.rand(2, 36, 3, 224, 224)
        with torch.no_grad():
            forward = model(frames)["logits"]
            backward = model(frames.flip(0))["logits"]
        self.assertTrue(torch.allclose(forward.flip(0), backward, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
