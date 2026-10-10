import unittest

import torch
from torch.nn import functional as F

from experiments.comparison.temporal_models import TemporalComparisonModel
from experiments.comparison.vision_models import VISION_BACKBONES, _replace_last_linear


class TemporalComparisonTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.global_tokens = torch.randn(2, 3, 8, 6)
        self.local_tokens = torch.randn(2, 3, 8, 4)
        self.a_logits = torch.randn(2, 3, 7)
        self.valid = torch.tensor([[True, True, True], [True, True, False]])

    def make_model(self, name):
        return TemporalComparisonModel(
            model_name=name,
            global_dim=6,
            local_dim=4,
            hidden_dim=8,
            dropout=0,
            tcn_dilations=(1, 2, 4),
            transformer_layers=2,
            transformer_heads=2,
            local_radius=1,
            max_sequence_length=16,
        )

    def test_all_temporal_heads_initially_preserve_A_logits(self):
        for name in ("mlp", "bigru", "tcn", "mstcnpp_adapted", "local_transformer"):
            with self.subTest(name=name):
                model = self.make_model(name).eval()
                output = model(
                    self.global_tokens, self.local_tokens, self.a_logits, self.valid
                )
                torch.testing.assert_close(output, self.a_logits, rtol=0, atol=0)

    def test_staged_tcn_returns_three_supervised_predictions(self):
        model = self.make_model("mstcnpp_adapted")
        outputs = model.forward_all(
            self.global_tokens, self.local_tokens, self.a_logits, self.valid
        )
        self.assertEqual(len(outputs), 3)
        for output in outputs:
            self.assertEqual(tuple(output.shape), (2, 3, 7))

    def test_padding_does_not_change_bigru_valid_predictions(self):
        model = self.make_model("bigru").eval()
        with torch.no_grad():
            model.classifier[-1].weight.fill_(0.02)
        short_global = self.global_tokens[:1, :2]
        short_local = self.local_tokens[:1, :2]
        short_logits = self.a_logits[:1, :2]
        short_valid = torch.ones(1, 2, dtype=torch.bool)
        alone = model(short_global, short_local, short_logits, short_valid)

        padded_global = torch.zeros(1, 5, 8, 6)
        padded_local = torch.zeros(1, 5, 8, 4)
        padded_logits = torch.zeros(1, 5, 7)
        padded_global[:, :2] = short_global
        padded_local[:, :2] = short_local
        padded_logits[:, :2] = short_logits
        padded_valid = torch.tensor([[True, True, False, False, False]])
        padded = model(padded_global, padded_local, padded_logits, padded_valid)
        torch.testing.assert_close(alone, padded[:, :2], rtol=1e-6, atol=1e-6)

    def test_clip_head_training_step_produces_finite_gradients(self):
        model = self.make_model("local_transformer")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        labels = torch.tensor([[0, 1, 2], [3, 4, -100]])
        output = model(
            self.global_tokens, self.local_tokens, self.a_logits, self.valid
        )
        loss = F.cross_entropy(output[self.valid], labels[self.valid])
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ))
        optimizer.step()


class VisionWrapperTest(unittest.TestCase):
    def test_declared_vision_backbones_cover_three_families(self):
        self.assertEqual(
            set(VISION_BACKBONES), {"mvit_v2_s", "swin3d_t", "r2plus1d_18"}
        )

    def test_classifier_replacement_preserves_sequential_head(self):
        module = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(32, 400))
        replacement, found = _replace_last_linear(module, 7)
        self.assertTrue(found)
        self.assertEqual(tuple(replacement[1].weight.shape), (7, 32))
        self.assertIsInstance(replacement[0], torch.nn.Dropout)


if __name__ == "__main__":
    unittest.main()
