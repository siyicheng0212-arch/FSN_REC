"""Safety checks for the opt-in within-clip evidence candidate."""

import unittest

import torch
from torch import nn
import torch.nn.functional as F

from experiments.clip_evidence import FSNClipEvidence, IntraClipEvidence


class TinyBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 7)
        self.class_weights = None

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
        return sum(
            F.cross_entropy(output[name][0], target)
            for name in ("random_branch", "policy_branch")
        )


class ClipEvidenceTest(unittest.TestCase):
    def test_initial_logits_equal_original_and_backward_reaches_new_head(self):
        torch.manual_seed(17)
        model = FSNClipEvidence(TinyBaseline())
        frames = torch.rand(2, 36, 3, 224, 224)
        out = model(frames)
        self.assertTrue(torch.equal(out["logits"], out["baseline_logits"]))
        self.assertTrue(torch.equal(
            out["iea_correction"], torch.zeros_like(out["iea_correction"])
        ))
        self.assertTrue(torch.allclose(
            out["iea_window_weights"].sum(dim=1), torch.ones(2, 7)
        ))
        model.compute_loss(out, torch.tensor([3, 4])).backward()
        self.assertIsNotNone(model.evidence_module.delta_head.weight.grad)
        self.assertGreater(
            float(model.evidence_module.delta_head.weight.grad.abs().sum()), 0.0
        )

    def test_repeated_frames_disable_motion_correction(self):
        torch.manual_seed(5)
        model = FSNClipEvidence(TinyBaseline())
        nn.init.constant_(model.evidence_module.delta_head.bias, 1.0)
        still = torch.rand(1, 1, 3, 224, 224).repeat(1, 36, 1, 1, 1)
        out = model(still)
        self.assertEqual(float(out["iea_motion_quality"][0]), 0.0)
        self.assertTrue(torch.equal(out["logits"], out["baseline_logits"]))

    def test_native_training_paths_receive_the_same_correction(self):
        model = FSNClipEvidence(TrainingBranchBaseline())
        nn.init.constant_(model.evidence_module.delta_head.bias, 0.3)
        out = model(torch.rand(1, 36, 3, 224, 224))
        self.assertTrue(torch.equal(out["logits"], out["policy_branch"][0]))
        self.assertTrue(torch.equal(out["logits"], out["random_branch"][0]))
        self.assertGreater(
            float((out["logits"] - out["baseline_logits"]).abs().sum()), 0.0
        )

    def test_batch_reordering_does_not_mix_clips(self):
        torch.manual_seed(3)
        model = FSNClipEvidence(TinyBaseline()).eval()
        nn.init.constant_(model.evidence_module.delta_head.bias, 0.25)
        frames = torch.rand(2, 36, 3, 224, 224)
        with torch.no_grad():
            first = model(frames)["logits"]
            reverse = model(frames.flip(0))["logits"]
        self.assertTrue(torch.allclose(first.flip(0), reverse, atol=1e-6))

    def test_rejects_wrong_frame_shape(self):
        with self.assertRaisesRegex(ValueError, "expected"):
            IntraClipEvidence()(torch.rand(1, 35, 3, 224, 224))


if __name__ == "__main__":
    unittest.main()
