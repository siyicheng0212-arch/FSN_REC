import unittest

import torch
from torch.nn import functional as F

from experiments.dual_timescale.model import (
    DualTimescaleModel,
    MultiScaleClipEncoder,
    TemporalContextHead,
    temporal_pyramid_pool,
)


class DualTimescaleTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.global_tokens = torch.randn(2, 3, 8, 6)
        self.local_tokens = torch.randn(2, 3, 8, 4)
        self.a_logits = torch.randn(2, 3, 7)
        self.valid = torch.tensor([[True, True, True], [True, True, False]])

    def test_temporal_pyramid_has_seven_ordered_bins(self):
        values = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8, 1)
        pooled = temporal_pyramid_pool(values, (1, 2, 4))
        self.assertEqual(tuple(pooled.shape), (1, 1, 7))
        torch.testing.assert_close(
            pooled.flatten(),
            torch.tensor([3.5, 1.5, 5.5, 0.5, 2.5, 4.5, 6.5]),
        )

    def test_clip_encoder_returns_embedding_and_zero_correction(self):
        encoder = MultiScaleClipEncoder(global_dim=6, local_dim=4, embedding_dim=8, dropout=0)
        embedding, correction = encoder(self.global_tokens, self.local_tokens, self.valid)
        self.assertEqual(tuple(embedding.shape), (2, 3, 8))
        self.assertEqual(tuple(correction.shape), (2, 3, 7))
        self.assertTrue(torch.equal(correction, torch.zeros_like(correction)))
        self.assertTrue(torch.equal(embedding[1, 2], torch.zeros_like(embedding[1, 2])))

    def test_all_learned_variants_initially_preserve_A_logits(self):
        for variant in ("M1", "M2", "M3"):
            with self.subTest(variant=variant):
                model = DualTimescaleModel(
                    variant=variant, global_dim=6, local_dim=4,
                    embedding_dim=8, hidden_dim=8, blocks=3, dropout=0,
                    dilations=(1, 2, 4),
                ).eval()
                output = model(self.global_tokens, self.local_tokens, self.a_logits, self.valid)
                self.assertTrue(torch.equal(output, self.a_logits))
                self.assertTrue(torch.equal(output[1, 2], self.a_logits[1, 2]))

    def test_context_padding_does_not_change_real_positions(self):
        model = TemporalContextHead(feature_dim=5, hidden_dim=8, blocks=3, dropout=0).eval()
        features = torch.randn(1, 2, 5)
        logits = torch.randn(1, 2, 7)
        valid = torch.ones(1, 2, dtype=torch.bool)
        alone = model(features, logits, valid)

        padded_features = torch.cat((features, torch.randn(1, 3, 5)), dim=1)
        padded_logits = torch.cat((logits, torch.randn(1, 3, 7)), dim=1)
        padded_valid = torch.tensor([[True, True, False, False, False]])
        padded = model(padded_features, padded_logits, padded_valid)
        torch.testing.assert_close(alone, padded[:, :2], atol=1e-6, rtol=1e-6)

    def test_clip_branch_can_train_with_seven_class_loss(self):
        model = DualTimescaleModel(
            variant="M1", global_dim=6, local_dim=4, embedding_dim=8,
            hidden_dim=8, blocks=3, dropout=0, dilations=(1, 2, 4),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        before = model(self.global_tokens, self.local_tokens, self.a_logits, self.valid).detach()
        labels = torch.tensor([[0, 1, 2], [2, 3, -100]])
        output = model(self.global_tokens, self.local_tokens, self.a_logits, self.valid)
        loss = F.cross_entropy(output[self.valid], labels[self.valid])
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        optimizer.step()
        after = model(self.global_tokens, self.local_tokens, self.a_logits, self.valid).detach()
        self.assertGreater(float((after - before).abs().max()), 0.0)

    def test_context_receptive_field_is_configured_explicitly(self):
        local = TemporalContextHead(feature_dim=5, hidden_dim=8, blocks=3,
                                    dropout=0, dilations=(1, 1, 1))
        historical = TemporalContextHead(feature_dim=5, hidden_dim=8, blocks=3,
                                         dropout=0, dilations=(1, 2, 4))
        self.assertEqual(local.receptive_field_clips, 7)
        self.assertEqual(historical.receptive_field_clips, 15)


if __name__ == "__main__":
    unittest.main()
