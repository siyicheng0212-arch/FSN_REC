"""Mechanism tests for the opt-in layer-2 local evidence module."""

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models" / "Uni-AdaFocus-TSM-FSN"))

from archs.local_motion import LocalMotionEvidence  # noqa: E402


class LocalMotionEvidenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    @staticmethod
    def module(**kwargs):
        return LocalMotionEvidence(channels=12, bottleneck=6, **kwargs)

    def test_initial_identity_and_staged_gradient(self):
        module = self.module()
        features = torch.randn(6, 12, 5, 5, requires_grad=True)
        target = torch.randn_like(features)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        output = module(features, 3)
        self.assertTrue(torch.equal(output, features))
        (output * target).mean().backward()
        self.assertGreater(module.up.weight.grad.abs().sum().item(), 0)
        self.assertEqual(module.down.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        (module(features, 3) * target).mean().backward()
        self.assertGreater(module.down.weight.grad.abs().sum().item(), 0)
        self.assertGreater(module.evidence[0].weight.grad.abs().sum().item(), 0)

    def test_appearance_has_equal_capacity_and_uses_all_projection_channels(self):
        matching = self.module(mode="matching")
        appearance = self.module(mode="appearance")
        self.assertEqual(
            sum(parameter.numel() for parameter in matching.parameters()),
            sum(parameter.numel() for parameter in appearance.parameters()),
        )
        torch.nn.init.normal_(appearance.up.weight, std=0.05)
        x = torch.randn(6, 12, 4, 4)
        (appearance(x, 3) * torch.randn_like(x)).sum().backward()
        per_channel = appearance.down.weight.grad.abs().flatten(1).sum(1)
        self.assertTrue((per_channel > 0).all().item())

    def test_known_translation_and_invalid_border_mask(self):
        height, width = 5, 5
        module = LocalMotionEvidence(
            channels=height * width, bottleneck=height * width, temperature=0.01
        )
        source = torch.eye(height * width).reshape(height * width, height, width)
        target = torch.zeros_like(source)
        target[:, :, 1:] = source[:, :, :-1]
        projected = torch.stack((source, target)).unsqueeze(0)
        motion = module.estimate_motion(projected)
        self.assertTrue(torch.allclose(motion[0, 0, 0, :, :-1], torch.ones(height, width - 1), atol=1e-4))
        self.assertTrue(torch.allclose(motion[0, 0, 1, :, :-1], torch.zeros(height, width - 1), atol=1e-4))
        self.assertTrue((motion[0, 0, 0, :, -1] <= 0).all().item())
        self.assertTrue(torch.equal(motion[:, -1], torch.zeros_like(motion[:, -1])))
        self.assertTrue(torch.isfinite(motion).all().item())

    def test_no_cross_clip_matching(self):
        module = self.module()
        x = torch.randn(2, 3, 6, 4, 4)
        combined = module.estimate_motion(x)
        separate = torch.cat([module.estimate_motion(x[i:i + 1]) for i in range(2)])
        self.assertTrue(torch.equal(combined, separate))

    def test_temporal_shuffle_changes_matching_but_not_appearance(self):
        x = torch.randn(4, 12, 5, 5)
        order = torch.tensor([0, 2, 1, 3])
        inverse = torch.argsort(order)
        for mode in ("matching", "appearance"):
            module = self.module(mode=mode).eval()
            torch.nn.init.normal_(module.up.weight, std=0.05)
            output = module(x, 4)
            restored = module(x[order], 4)[inverse]
            if mode == "matching":
                self.assertFalse(torch.allclose(output, restored))
            else:
                self.assertTrue(torch.allclose(output, restored, atol=1e-6))

    def test_global_context_identity_then_gradient_and_spatial_sensitivity(self):
        module = self.module(context_mode="global", context_channels=8, context_grid=2)
        x = torch.randn(6, 12, 4, 4)
        context = torch.randn(2, 2, 8, 4, 4, requires_grad=True)
        self.assertTrue(torch.equal(module(x, 3, global_context=context), x))
        torch.nn.init.normal_(module.up.weight, std=0.05)
        torch.nn.init.normal_(module.context_gate.weight, std=0.05)
        output = module(x, 3, global_context=context)
        (output * torch.randn_like(output)).sum().backward()
        self.assertGreater(context.grad.abs().sum().item(), 0)
        self.assertGreater(module.context_projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(module.context_global.weight.grad.abs().sum().item(), 0)
        permuted = module(x, 3, global_context=context.flip(-1))
        self.assertFalse(torch.allclose(output, permuted))

    def test_identity_initialized_context_learns_after_residual_activation(self):
        module = self.module(context_mode="global", context_channels=8)
        x = torch.randn(6, 12, 4, 4)
        context = torch.randn(2, 2, 8, 4, 4)
        target = torch.randn_like(x)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.5)
        for step in range(3):
            optimizer.zero_grad()
            (module(x, 3, global_context=context) * target).mean().backward()
            if step == 0:
                self.assertGreater(module.up.weight.grad.abs().sum().item(), 0)
            elif step == 1:
                self.assertGreater(module.context_gate.weight.grad.abs().sum().item(), 0)
            else:
                self.assertGreater(module.context_projection.weight.grad.abs().sum().item(), 0)
            optimizer.step()

    def test_disable_is_exact_even_after_activation_and_without_context(self):
        module = self.module(context_mode="global", context_channels=8).eval()
        torch.nn.init.normal_(module.up.weight, std=0.1)
        x = torch.randn(6, 12, 4, 4)
        module.enabled = False
        output = module(x, 3)
        self.assertIs(output, x)
        self.assertEqual(module.last_diagnostics, {})

    def test_context_can_be_disabled_without_disabling_motion(self):
        module = self.module(context_mode="global", context_channels=8).eval()
        torch.nn.init.normal_(module.up.weight, std=0.1)
        torch.nn.init.normal_(module.context_gate.weight, std=0.1)
        x = torch.randn(6, 12, 4, 4)
        module.context_enabled = False
        local_only = module(x, 3)
        self.assertFalse(torch.equal(local_only, x))
        changed_context = module(x, 3, global_context=torch.randn(2, 2, 8, 4, 4))
        self.assertTrue(torch.equal(local_only, changed_context))
        self.assertNotIn("calibration_delta_rms", module.diagnostics())

    def test_positions_diagnostics_and_single_frame(self):
        module = self.module()
        x = torch.randn(6, 12, 4, 4)
        positions = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.4, 0.7]])
        module(x, 3, positions=positions)
        self.assertAlmostEqual(module.last_diagnostics["normalized_gap_mean"].item(), 0.4, places=6)
        self.assertAlmostEqual(module.last_diagnostics["duplicate_position_fraction"].item(), 0.25)
        self.assertTrue(all(not value.requires_grad for value in module.last_diagnostics.values()))
        self.assertTrue(torch.equal(module(x[:1], 1), x[:1]))
        self.assertEqual(module.last_diagnostics["matching_confidence"].item(), 0)

    def test_shape_and_position_errors(self):
        module = self.module()
        x = torch.randn(6, 12, 4, 4)
        with self.assertRaises(ValueError):
            module(x, 4)
        with self.assertRaises(ValueError):
            module(x[:, :8], 3)
        with self.assertRaises(ValueError):
            module(x, 3, positions=torch.zeros(2, 2))
        for bad in (
            torch.tensor([[0.0, 0.6, 0.2], [0.0, 0.5, 1.0]]),
            torch.tensor([[0.0, 0.5, 1.1], [0.0, 0.5, 1.0]]),
            torch.tensor([[0.0, float("nan"), 1.0], [0.0, 0.5, 1.0]]),
        ):
            with self.assertRaises((AssertionError, RuntimeError)):
                module(x, 3, positions=bad)
        with self.assertRaises(ValueError):
            self.module(context_mode="global", context_channels=8)(x, 3)

    def test_cpu_bfloat16_matching_stays_finite_and_restores_input_dtype(self):
        module = self.module()
        torch.nn.init.normal_(module.up.weight, std=0.05)
        x = torch.randn(6, 12, 5, 5).to(torch.bfloat16).requires_grad_()
        try:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                output = module(x, 3)
            output.float().square().mean().backward()
        except RuntimeError as error:
            if "not implemented" in str(error).lower():
                self.skipTest(str(error))
            raise
        self.assertEqual(output.dtype, x.dtype)
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertTrue(torch.isfinite(module.down.weight.grad).all().item())
        motion = module.estimate_motion(torch.randn(1, 3, 6, 5, 5).bfloat16())
        self.assertEqual(motion.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
