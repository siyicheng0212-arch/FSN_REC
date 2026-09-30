"""Mechanism audit and trainer contract tests; no private data or GPU needed."""
import argparse
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from experiments.audit_local_motion import paired_forward, prediction_effect, model_from_saved_args
from experiments.model_wrappers import AdaFocusFSN
from experiments.train_adafocus import make_optimizer, parse_args, make_model


class StochasticAuditModel(nn.Module):
    def __init__(self, delta=0.):
        super().__init__()
        self.local_motion_module = SimpleNamespace(enabled=True, context_enabled=True, context_mode="global")
        self.delta = delta

    def forward(self, video):
        logits = torch.randn(len(video), 7)
        if self.local_motion_module.enabled and self.local_motion_module.context_enabled:
            logits[:, 0] += self.delta
        return {"logits": logits}

    def get_local_motion_diagnostics(self):
        return {"residual_rms": torch.tensor(self.delta)}


class LocalMotionTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_paired_rng_prevents_false_module_effect(self):
        model = StochasticAuditModel()
        video = torch.zeros(3, 1)
        torch.manual_seed(21)
        before = torch.random.get_rng_state()
        expected = model(video)["logits"]
        expected_after = torch.random.get_rng_state()
        torch.random.set_rng_state(before)
        on, off, _ = paired_forward(model, video, torch.device("cpu"))
        self.assertTrue(torch.equal(expected, on["logits"]))
        self.assertTrue(torch.equal(on["logits"], off["logits"]))
        self.assertTrue(torch.equal(expected_after, torch.random.get_rng_state()))
        self.assertTrue(model.local_motion_module.enabled)

    def test_context_only_intervention_and_state_restore(self):
        model = StochasticAuditModel(delta=2.)
        model.local_motion_module.enabled = False
        model.local_motion_module.context_enabled = False
        on, off, diagnostics = paired_forward(model, torch.zeros(2, 1), torch.device("cpu"), "context")
        delta = on["logits"] - off["logits"]
        torch.testing.assert_close(delta[:, 0], torch.full((2,), 2.))
        torch.testing.assert_close(delta[:, 1:], torch.zeros(2, 6))
        self.assertFalse(model.local_motion_module.enabled)
        self.assertFalse(model.local_motion_module.context_enabled)
        self.assertEqual(diagnostics["residual_rms"], 2.)

    def test_effect_counts_corrected_harmed_and_neutral(self):
        target = torch.tensor([0, 1, 2, 3])
        off = torch.full((4, 7), -1.)
        on = off.clone()
        off[torch.arange(4), torch.tensor([1, 1, 0, 3])] = 1.
        on[torch.arange(4), torch.tensor([0, 0, 1, 3])] = 1.
        result = prediction_effect(on, off, target)
        self.assertEqual((result["changed"], result["corrected"], result["harmed"], result["changed_both_wrong"]), (3, 1, 1, 1))

    def test_new_optimizer_group_has_complete_unique_coverage(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model = AdaFocusFSN(local_motion_mode="matching", local_motion_context="global")
            model.train()
        args = SimpleNamespace(lr=.002, weight_decay=5e-4, temporal_lr_ratio=.2,
                               stn_lr_ratio=.2, global_lr_ratio=.5, local_motion_lr_ratio=2.)
        optimizer, summary = make_optimizer(model, args)
        group = next(row for row in summary if row["name"] == "local_motion")
        self.assertEqual(group["initial_lr"], .004)
        ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})
        self.assertTrue(all(p.requires_grad for p in model.local_motion_module.parameters()))

    def test_saved_config_and_class_weight_checkpoint_restore(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model = model_from_saved_args({"variant": "local_motion_context", "local_motion_dim": 12,
                                          "local_motion_window": 5}, torch.device("cpu"))
            model.set_class_weights(torch.arange(1., 8.))
            state = model.state_dict()
            restored = model_from_saved_args({"variant": "local_motion_context", "local_motion_dim": 12,
                                             "local_motion_window": 5}, torch.device("cpu"))
        restored.set_class_weights(state["class_weights"])
        restored.load_state_dict(state, strict=True)
        self.assertEqual(restored.local_motion_module.window_size, 5)
        self.assertTrue(torch.equal(restored.class_weights, model.class_weights))

    def test_make_model_preserves_original_rng_and_shared_state(self):
        common = dict(seed=42, checkpoint=None, allow_random_init=True,
                      local_motion_dim=8, local_motion_window=3,
                      local_motion_temperature=.07, local_motion_context_grid=2)
        with contextlib.redirect_stdout(io.StringIO()):
            baseline, _ = make_model(argparse.Namespace(variant="original", **common), torch.device("cpu"))
            rng = torch.random.get_rng_state().clone()
            candidate, report = make_model(argparse.Namespace(variant="local_motion", **common), torch.device("cpu"))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for name, tensor in baseline.state_dict().items():
            self.assertTrue(torch.equal(tensor, candidate.state_dict()[name]), name)
        self.assertTrue(all(".local_motion." in key for key in report["new_module_keys"]))
        self.assertIsNone(candidate.core.local_CNN.local_adapter)
        self.assertIsNone(candidate.core.fsn_interaction)

    def test_cli_defaults_and_invalid_motion_settings(self):
        with patch("sys.argv", ["trainer", "--variant", "local_motion"]):
            args = parse_args()
        self.assertEqual(args.module_warmup_epochs, 0)
        self.assertEqual(args.local_motion_window, 3)
        for options in (["--local-motion-window", "4"], ["--local-motion-temperature", "nan"],
                        ["--module-warmup-epochs", "-1"], ["--local-motion-dim", "2"]):
            with patch("sys.argv", ["trainer", "--variant", "local_motion", *options]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): parse_args()


if __name__ == "__main__":
    unittest.main()
