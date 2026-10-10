import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from experiments.comparison.temporal_models import (
    MarkovViterbiDecoder,
    TemporalComparisonModel,
    estimate_transition_log_probs,
)
from experiments.comparison.vision_models import (
    KineticsVideoClassifier, VISION_BACKBONES, _replace_last_linear,
)
from experiments.comparison.foundation_models import FrozenFoundationLinearProbe


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
        for name in ("mlp", "bigru", "tcn", "tcn_local", "mstcnpp_adapted", "local_transformer"):
            with self.subTest(name=name):
                model = self.make_model(name).eval()
                output = model(
                    self.global_tokens, self.local_tokens, self.a_logits, self.valid
                )
                torch.testing.assert_close(output, self.a_logits, rtol=0, atol=0)


    def test_tcn_and_local_tcn_have_matched_size_but_different_receptive_field(self):
        main_tcn = self.make_model("tcn")
        local_tcn = self.make_model("tcn_local")
        self.assertEqual(main_tcn.parameter_count, local_tcn.parameter_count)
        self.assertEqual(main_tcn.temporal_dilations, (1, 2, 4))
        self.assertEqual(local_tcn.temporal_dilations, (1, 1, 1))
        self.assertEqual(main_tcn.tcn.config.receptive_field_clips, 15)
        self.assertEqual(local_tcn.tcn.config.receptive_field_clips, 7)

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


    def test_transition_prior_uses_only_supplied_training_chains(self):
        train_chain = SimpleNamespace(labels=np.asarray([0, 1, 2, 2, 2], dtype=np.int64))
        log_probs = estimate_transition_log_probs([train_chain], smoothing=1.0)
        self.assertEqual(tuple(log_probs.shape), (7, 7))
        torch.testing.assert_close(
            torch.logsumexp(log_probs, dim=1),
            torch.zeros(7),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertGreater(float(log_probs[0, 1]), float(log_probs[0, 0]))
        self.assertGreater(float(log_probs[2, 2]), float(log_probs[2, 1]))

    def test_markov_decoder_reduces_to_independent_argmax_at_zero_strength(self):
        log_probs = torch.full((7, 7), -torch.log(torch.tensor(7.0)).item())
        decoder = MarkovViterbiDecoder(log_probs, transition_strength=0.0).eval()
        logits = torch.randn(2, 3, 7)
        valid = torch.tensor([[True, True, True], [True, True, False]])
        output = decoder(
            torch.empty(2, 3, 8, 6),
            torch.empty(2, 3, 8, 4),
            logits,
            valid,
        )
        torch.testing.assert_close(
            output.argmax(dim=-1)[valid],
            logits.argmax(dim=-1)[valid],
        )

    def test_markov_decoder_can_use_train_transition_order(self):
        transition = torch.full((7, 7), -10.0)
        transition[0, 1] = 0.0
        transition[1, 1] = 0.0
        transition = torch.log_softmax(transition, dim=-1)
        decoder = MarkovViterbiDecoder(transition, transition_strength=1.0).eval()
        logits = torch.full((1, 2, 7), -10.0)
        logits[0, 0, 0] = 2.0
        logits[0, 0, 1] = 1.9
        logits[0, 1, 0] = 2.0
        logits[0, 1, 1] = 1.9
        valid = torch.ones(1, 2, dtype=torch.bool)
        output = decoder(
            torch.empty(1, 2, 8, 6),
            torch.empty(1, 2, 8, 4),
            logits,
            valid,
        )
        self.assertEqual(output.argmax(dim=-1).tolist(), [[0, 1]])

    def test_all_learned_temporal_heads_have_finite_training_gradients(self):
        labels = torch.tensor([[0, 1, 2], [3, 4, -100]])
        for name in ("mlp", "bigru", "tcn", "tcn_local", "mstcnpp_adapted", "local_transformer"):
            with self.subTest(name=name):
                model = self.make_model(name)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
                stage_outputs = model.forward_all(
                    self.global_tokens, self.local_tokens, self.a_logits, self.valid
                )
                losses = [
                    F.cross_entropy(output[self.valid], labels[self.valid])
                    for output in stage_outputs
                ]
                loss = torch.stack(losses).mean()
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                self.assertTrue(all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                ))
                optimizer.step()



    def test_video_wrapper_uses_16_frames_and_registered_spatial_size(self):
        class CaptureCore(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(3, 7)
                self.last_shape = None

            def forward(self, video):
                self.last_shape = tuple(video.shape)
                pooled = video.mean(dim=(2, 3, 4))
                return self.fc(pooled)

        expected_sizes = {
            "mvit_v2_s": 224,
            "swin3d_t": 224,
            "r2plus1d_18": 112,
        }
        raw = torch.randint(0, 256, (1, 36, 3, 224, 224), dtype=torch.uint8)
        for name, size in expected_sizes.items():
            with self.subTest(name=name):
                core = CaptureCore()
                wrapper = KineticsVideoClassifier(name, core, "fc", "synthetic-test")
                output = wrapper(raw)
                self.assertEqual(tuple(output.shape), (1, 7))
                self.assertEqual(core.last_shape, (1, 3, 16, size, size))


class VisionWrapperTest(unittest.TestCase):
    def test_declared_vision_backbones_cover_kinetics_and_foundation_families(self):
        self.assertEqual(
            set(VISION_BACKBONES),
            {
                "mvit_v2_s",
                "swin3d_t",
                "r2plus1d_18",
                "vjepa2_vitl",
                "internvideo_next_base",
            },
        )

    def test_foundation_linear_probe_freezes_encoder_and_returns_seven_logits(self):
        class StubEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(1))

            def forward(self, nested_video):
                video = nested_video[0][0]
                pooled = video.mean(dim=(2, 3, 4)) * self.scale
                return [pooled.unsqueeze(1)]

        model = FrozenFoundationLinearProbe(
            name="vjepa2_vitl",
            core=StubEncoder(),
            feature_dim=3,
            model_kind="vjepa2",
            mean=(0.45, 0.45, 0.45),
            std=(0.225, 0.225, 0.225),
            input_size=(256, 256),
            crop_size=(256, 256),
            rescale_factor=1.0 / 255.0,
            audit={"test": True},
        )
        model.set_mode("linear_probe")
        raw = torch.randint(0, 256, (1, 36, 3, 224, 224), dtype=torch.uint8)
        logits = model(raw)
        self.assertEqual(tuple(logits.shape), (1, 7))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        logits.sum().backward()
        self.assertIsNotNone(model.head.weight.grad)
        self.assertIsNone(model.core.scale.grad)
        self.assertFalse(model.core.scale.requires_grad)

    def test_foundation_linear_probe_rejects_full_finetuning(self):
        class StubEncoder(torch.nn.Module):
            def forward(self, nested_video):
                return [nested_video[0][0].mean(dim=(2, 3, 4)).unsqueeze(1)]

        model = FrozenFoundationLinearProbe(
            name="vjepa2_vitl",
            core=StubEncoder(),
            feature_dim=3,
            model_kind="vjepa2",
            mean=(0.45, 0.45, 0.45),
            std=(0.225, 0.225, 0.225),
            input_size=(256, 256),
            crop_size=(256, 256),
            rescale_factor=1.0 / 255.0,
            audit={"test": True},
        )
        with self.assertRaisesRegex(ValueError, "linear_probe"):
            model.set_mode("finetune")


    def test_video_wrapper_samples_sixteen_frames_and_applies_model_crop(self):
        class DummyVideoCore(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(3, 7)
                self.seen_shape = None

            def forward(self, video):
                self.seen_shape = tuple(video.shape)
                pooled = video.mean(dim=(2, 3, 4))
                return self.head(pooled)

        core = DummyVideoCore()
        model = KineticsVideoClassifier(
            "r2plus1d_18", core, classifier_attribute="head", weight_url="test-only"
        ).eval()
        raw = torch.full((1, 36, 3, 224, 224), 127, dtype=torch.uint8)
        with torch.no_grad():
            output = model(raw)
        self.assertEqual(tuple(output.shape), (1, 7))
        self.assertEqual(core.seen_shape, (1, 3, 16, 112, 112))
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_classifier_replacement_preserves_sequential_head(self):
        module = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(32, 400))
        replacement, found = _replace_last_linear(module, 7)
        self.assertTrue(found)
        self.assertEqual(tuple(replacement[1].weight.shape), (7, 32))
        self.assertIsInstance(replacement[0], torch.nn.Dropout)


if __name__ == "__main__":
    unittest.main()
