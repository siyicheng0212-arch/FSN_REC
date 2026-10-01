"""Mechanism and geometry checks for the new single-clip residual."""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models" / "Uni-AdaFocus-TSM-FSN"))
from archs.aligned_context import AlignedContextResidual, crop_affine_theta, pooled_grid_centers
from archs.uni_adafocus_tsm import get_patch_grid_scalexy


class AlignedContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(min(4, __import__("os").cpu_count() or 1))

    def setUp(self):
        torch.manual_seed(19)
        self.x = torch.randn(6, 12, 5, 6)
        self.context = torch.randn(2, 2, 8, 7, 7)
        self.kwargs = dict(global_context=self.context,
                           positions=torch.tensor([[.0, .5, 1.], [.1, .5, .9]]),
                           global_positions=torch.tensor([[.0, 1.], [.0, 1.]]),
                           crop_actions=torch.tensor([[.1, .2, .25, .25], [.5, .4, .25, .25]]))

    @staticmethod
    def module(mode="aligned"):
        return AlignedContextResidual(channels=12, dim=6, global_channels=8, grid=3, mode=mode)

    def test_identity_then_gradients_and_detached_global_input(self):
        module = self.module()
        context = self.context.clone().requires_grad_()
        kwargs = dict(self.kwargs, global_context=context)
        target = torch.randn_like(self.x)
        optimizer = torch.optim.SGD(module.parameters(), lr=.5)
        output = module(self.x, 3, **kwargs)
        self.assertTrue(torch.equal(output, self.x))
        (output * target).sum().backward()
        self.assertGreater(module.up.weight.grad.abs().sum().item(), 0)
        self.assertEqual(module.down.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        (module(self.x, 3, **kwargs) * target).sum().backward()
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertIsNone(context.grad)
        self.assertTrue(all(not value.requires_grad for value in module.diagnostics().values()))

    def test_all_modes_have_equal_active_capacity(self):
        counts = []
        for mode in ("plain", "aligned", "capacity"):
            module = self.module(mode)
            counts.append(sum(parameter.numel() for parameter in module.parameters()))
            torch.nn.init.normal_(module.up.weight, std=.05)
            output = module(self.x, 3, **self.kwargs)
            (output * torch.randn_like(output)).sum().backward()
            for name, parameter in module.named_parameters():
                self.assertIsNotNone(parameter.grad, (mode, name))
                self.assertGreater(parameter.grad.abs().sum().item(), 0, (mode, name))
        self.assertEqual(len(set(counts)), 1)
        default = AlignedContextResidual(mode="capacity")
        self.assertEqual(sum(parameter.numel() for parameter in default.parameters()), 164864)
        self.assertIsNone(default.global_projection)

    def test_disabling_both_priors_reproduces_plain_content_attention(self):
        aligned, plain = self.module(), self.module("plain")
        torch.nn.init.normal_(aligned.up.weight, std=.05)
        plain.load_state_dict(aligned.state_dict(), strict=True)
        aligned.time_prior_enabled = False
        aligned.spatial_prior_enabled = False
        torch.testing.assert_close(aligned(self.x, 3, **self.kwargs),
                                   plain(self.x, 3, **self.kwargs), rtol=0, atol=0)

    def test_temporal_correspondence_changes_aligned_but_not_plain(self):
        for mode in ("aligned", "plain"):
            module = self.module(mode)
            torch.nn.init.normal_(module.up.weight, std=.1)
            first = module(self.x, 3, **self.kwargs)
            second = module(self.x, 3, **dict(self.kwargs, global_context=self.context.flip(1)))
            # Plain attention treats time tokens as an unordered content set.
            if mode == "plain":
                torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
            else:
                self.assertFalse(torch.allclose(first, second))

    def test_geometry_affects_only_aligned_candidate(self):
        changed = dict(self.kwargs, crop_actions=1 - self.kwargs["crop_actions"])
        for mode in ("aligned", "plain", "capacity"):
            module = self.module(mode)
            torch.nn.init.normal_(module.up.weight, std=.1)
            module.time_prior_enabled = False
            first, second = module(self.x, 3, **self.kwargs), module(self.x, 3, **changed)
            if mode == "aligned":
                self.assertFalse(torch.allclose(first, second))
            else:
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_crop_sampling_geometry_and_actual_pool_bin_centers(self):
        image_size, patch_size = 224, 128
        action = self.kwargs["crop_actions"]
        theta = crop_affine_theta(action, image_size, patch_size, patch_size)
        pixels = torch.rand(2, 3, image_size, image_size)
        grid = F.affine_grid(theta, (2, 3, patch_size, patch_size), align_corners=False)
        expected = F.grid_sample(pixels, grid, align_corners=False)
        actual = get_patch_grid_scalexy(pixels, action, image_size, patch_size, patch_size)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        centers = pooled_grid_centers(7, 7, 3, torch.device("cpu"))
        torch.testing.assert_close(centers[0], torch.tensor([1.5 / 7, 1.5 / 7]))
        torch.testing.assert_close(centers[4], torch.tensor([.5, .5]))
        # Asymmetric y/x actions exercise the released action-axis convention.
        expected_x_center = (action[:, 1] * (224 - 128) + 64) / 224
        expected_y_center = (action[:, 0] * (224 - 128) + 64) / 224
        torch.testing.assert_close((theta[:, 0, 2] + 1) / 2, expected_x_center)
        torch.testing.assert_close((theta[:, 1, 2] + 1) / 2, expected_y_center)

    def test_no_cross_clip_and_capacity_never_reads_global(self):
        for mode in ("aligned", "capacity"):
            module = self.module(mode)
            torch.nn.init.normal_(module.up.weight, std=.1)
            combined = module(self.x, 3, **self.kwargs)
            separate = []
            for index in range(2):
                kwargs = {key: value[index:index + 1] for key, value in self.kwargs.items()}
                separate.append(module(self.x[index * 3:(index + 1) * 3], 3, **kwargs))
            torch.testing.assert_close(combined, torch.cat(separate), rtol=1e-5, atol=1e-6)
            if mode == "capacity":
                torch.testing.assert_close(combined, module(self.x, 3), rtol=0, atol=0)

    def test_disable_bypasses_everything_after_activation(self):
        module = self.module()
        torch.nn.init.normal_(module.up.weight, std=.1)
        module.enabled = False
        self.assertIs(module(self.x, 3), self.x)
        self.assertEqual(module.diagnostics(), {})

    def test_validation_and_bfloat16_finite(self):
        module = self.module()
        with self.assertRaises(ValueError):
            module(self.x, 4, **self.kwargs)
        with self.assertRaises(ValueError):
            module(self.x, 3, **dict(self.kwargs, global_positions=None))
        with self.assertRaises((RuntimeError, AssertionError)):
            module(self.x, 3, **dict(self.kwargs, crop_actions=torch.full((2, 4), float("nan"))))
        torch.nn.init.normal_(module.up.weight, std=.05)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = module(self.x.bfloat16(), 3, **self.kwargs)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output).all().item())
        output.float().square().mean().backward()
        self.assertTrue(torch.isfinite(module.down.weight.grad).all().item())


if __name__ == "__main__":
    unittest.main()
