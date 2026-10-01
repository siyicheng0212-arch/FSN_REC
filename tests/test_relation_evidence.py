"""The evidence reader preserves A and never leaks gradients or hooks."""
import unittest
from pathlib import Path
import tempfile

import torch
from torch import nn

from experiments.relation.evidence import FrozenOriginalEvidence, load_original


class DummyGlobal(nn.Module):
    def forward(self, values):
        features = values[:, None].expand(-1, 1280)
        return (None, None, None, features, None)


class DummyOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = "adafocus_original"
        self.num_glance_segments, self.num_input_focus_segments, self.num_focus_segments = 8, 36, 12
        self.head = nn.Linear(1280, 7)
        self.core = nn.Module()
        self.core.global_CNN = DummyGlobal()
        self.core.local_CNN = nn.Module()
        self.core.local_CNN.base_model = nn.Module()
        self.core.local_CNN.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fail = False

    @staticmethod
    def _take_uniform(frames, count):
        indices = torch.linspace(0, frames.shape[1] - 1, count).round().long()
        return frames[:, indices], (indices.float() / max(frames.shape[1] - 1, 1)).expand(frames.shape[0], -1)

    def forward(self, frames):
        b = frames.shape[0]
        features = self.core.global_CNN(frames.mean((1, 2, 3, 4)).repeat_interleave(8))[3]
        self.core.local_CNN.base_model.avgpool(torch.ones(b * 12, 2048, 1, 1))
        if self.fail:
            raise RuntimeError("deliberate failure")
        indices = torch.arange(0, 36, 3).expand(b, -1) + torch.arange(b).unsqueeze(1) * 36
        logits = self.head(features.reshape(b, 8, 1280).mean(1))
        return {"logits": logits, "eval_outputs": (logits,) * 8 + (indices.flatten(),)}


class RelationEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)

    def test_freeze_eval_logit_identity_positions_and_cleanup(self):
        visual = DummyOriginal().eval()
        frames = torch.rand(2, 36, 3, 224, 224, requires_grad=True)
        before = visual(frames)["logits"].detach()
        extractor = FrozenOriginalEvidence(visual).train()
        self.assertFalse(visual.training)
        result = extractor(frames)
        torch.testing.assert_close(before, result["logits"], rtol=0, atol=0)
        self.assertEqual(result["global_tokens"].shape, (2, 8, 1280))
        self.assertEqual(result["local_tokens"].shape, (2, 12, 2048))
        torch.testing.assert_close(result["local_positions"][0], torch.arange(0, 36, 3) / 35)
        torch.testing.assert_close(result["local_positions"][0], result["local_positions"][1])
        self.assertTrue(all(not x.requires_grad for x in result.values()))
        self.assertTrue(all(not p.requires_grad for p in visual.parameters()))
        for module in visual.modules():
            self.assertEqual(len(module._forward_hooks), 0)

    def test_invalid_inputs_modules_and_failure_do_not_leave_hooks(self):
        visual = DummyOriginal()
        extractor = FrozenOriginalEvidence(visual)
        for frames in (torch.zeros(1, 36, 3, 224, 224, dtype=torch.uint8),
                       torch.ones(1, 36, 3, 224, 224) * 2,
                       torch.ones(1, 36, 3, 12, 12)):
            with self.assertRaises(ValueError):
                extractor(frames)
        visual.fail = True
        with self.assertRaises(RuntimeError):
            extractor(torch.zeros(1, 36, 3, 224, 224))
        self.assertTrue(all(not module._forward_hooks for module in visual.modules()))
        for modification in ("name", "module", "path"):
            visual = DummyOriginal()
            if modification == "name":
                visual.model_name = "adafocus_context_aligned"
            elif modification == "module":
                visual.core.local_CNN.local_motion = nn.Identity()
            else:
                visual.core.local_CNN.return_feature_grid = True
            with self.assertRaises(ValueError):
                FrozenOriginalEvidence(visual)

    def test_real_untrained_original_forward_contract_and_exact_logits(self):
        # A contract smoke, not a trained model accuracy or CUDA test.
        from experiments.model_wrappers import AdaFocusFSN
        torch.manual_seed(42)
        visual = AdaFocusFSN(device=torch.device("cpu"), num_glance_segments=8,
                            num_input_focus_segments=36, num_focus_segments=12,
                            patch_size=128, mc_sample_times=128).eval()
        frames = torch.rand(1, 36, 3, 224, 224)
        with torch.no_grad():
            original = visual(frames)["logits"]
        result = FrozenOriginalEvidence(visual)(frames)
        torch.testing.assert_close(result["logits"], original, rtol=0, atol=0)
        self.assertEqual(result["global_tokens"].shape, (1, 8, 1280))
        self.assertEqual(result["local_tokens"].shape, (1, 12, 2048))
        self.assertTrue(torch.all(result["local_positions"][:, 1:] >= result["local_positions"][:, :-1]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            visual.set_class_weights(torch.ones(7))
            torch.save({"model": visual.state_dict(),
                        "args": {"variant": "original", "cache_dir": Path("/private/cache")}}, path)
            restored = load_original(path)
            torch.testing.assert_close(restored(frames)["logits"], original, rtol=0, atol=0)
            torch.testing.assert_close(restored.visual.class_weights, torch.ones(7))
            torch.save({"model": {}, "args": {"variant": "context_plain"}}, path)
            with self.assertRaisesRegex(ValueError, "canonical Original"):
                load_original(path)
            torch.save({"model": {}, "args": {"patch_size": 96}}, path)
            with self.assertRaisesRegex(ValueError, "sampling mismatch"):
                load_original(path)


if __name__ == "__main__":
    unittest.main()
