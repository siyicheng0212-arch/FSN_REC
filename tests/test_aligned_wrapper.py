"""Integration checks for all four Original-derived context variants."""

import contextlib
import io
import unittest
from types import SimpleNamespace

import torch

from experiments.model_wrappers import AdaFocusFSN, build_model, load_shared_adafocus_weights


class AlignedWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(min(4, __import__("os").cpu_count() or 1))

    @staticmethod
    def make(mode):
        with contextlib.redirect_stdout(io.StringIO()):
            return AdaFocusFSN(context_mode=mode, context_dim=8)

    def test_shared_initialization_rng_and_logits_are_original(self):
        torch.manual_seed(11)
        original = self.make("none").eval()
        original_state = {key: value.clone() for key, value in original.state_dict().items()}
        original_rng = torch.random.get_rng_state().clone()
        frames = torch.rand(1, 8, 3, 224, 224)
        torch.manual_seed(103)
        with torch.no_grad():
            logits = original(frames)["logits"]
        for mode in ("plain", "aligned", "capacity"):
            torch.manual_seed(11)
            candidate = self.make(mode).eval()
            self.assertTrue(torch.equal(torch.random.get_rng_state(), original_rng))
            for key, value in original_state.items():
                torch.testing.assert_close(candidate.state_dict()[key], value, rtol=0, atol=0)
            report = load_shared_adafocus_weights(candidate, original_state)
            self.assertTrue(all(".aligned_context." in key for key in report["missing_keys"]))
            self.assertEqual(len(report["loaded_keys"]), len(original_state))
            torch.manual_seed(103)
            with torch.no_grad():
                observed = candidate(frames)["logits"]
            torch.testing.assert_close(observed, logits, rtol=0, atol=0)
            torch.nn.init.normal_(candidate.context_module.up.weight, std=.05)
            candidate.set_context_enabled(False)
            torch.manual_seed(103)
            with torch.no_grad():
                disabled = candidate(frames)["logits"]
            torch.testing.assert_close(disabled, logits, rtol=0, atol=0)

    def test_training_crop_branches_keep_each_geometry_and_clip_positions(self):
        model = self.make("aligned").train()
        seen = []
        handle = model.context_module.register_forward_pre_hook(
            lambda module, args, kwargs: seen.append({key: value.clone() for key, value in kwargs.items()}),
            with_kwargs=True,
        )
        try:
            output = model(torch.rand(2, 8, 3, 224, 224))
        finally:
            handle.remove()
        self.assertEqual(len(seen), 1)
        observed = seen[0]
        self.assertEqual(observed["global_context"].shape[:3], (4, 4, 1280))
        self.assertFalse(observed["global_context"].requires_grad)
        torch.testing.assert_close(observed["global_context"][:2], observed["global_context"][2:])
        torch.testing.assert_close(observed["positions"][:2], observed["positions"][2:])
        torch.testing.assert_close(observed["global_positions"][:2], observed["global_positions"][2:])
        torch.testing.assert_close(observed["crop_actions"][:2], output["random_branch"][5])
        torch.testing.assert_close(observed["crop_actions"][2:], output["policy_branch"][5])
        self.assertFalse(torch.equal(observed["crop_actions"][:2], observed["crop_actions"][2:]))

    def test_original_final_detach_and_local_auxiliary_training_boundary(self):
        model = self.make("aligned").train()
        kwargs = dict(global_context=torch.rand(1, 4, 1280, 7, 7, requires_grad=True),
                      positions=torch.linspace(0, 1, 4).unsqueeze(0),
                      global_positions=torch.linspace(0, 1, 4).unsqueeze(0),
                      crop_actions=torch.full((1, 4), .5))
        final_logits, local_aux = model.core.local_CNN(torch.rand(1, 12, 96, 96), **kwargs)
        final_logits.square().sum().backward(retain_graph=True)
        self.assertTrue(all(parameter.grad is None for parameter in model.context_module.parameters()))
        model.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(local_aux, torch.tensor([2])).backward()
        self.assertGreater(model.context_module.up.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(kwargs["global_context"].grad)

    def test_optimizer_groups_are_unique_complete_and_context_lr_is_respected(self):
        model = self.make("aligned")
        args = SimpleNamespace(global_lr_ratio=.5, stn_lr_ratio=.2, temporal_lr_ratio=.2, context_lr_ratio=2.)
        policies = model.core.get_optim_policies(args)
        ids = [id(parameter) for policy in policies for parameter in policy["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(parameter) for parameter in model.parameters()})
        policy = next(policy for policy in policies if policy["name"] == "aligned_context")
        self.assertEqual(policy["lr_mult"], 2.)
        self.assertEqual({id(parameter) for parameter in policy["params"]},
                         {id(parameter) for parameter in model.context_module.parameters()})
        with contextlib.redirect_stdout(io.StringIO()):
            model.core.local_CNN.train()
        local_ids = [id(parameter) for policy in model.core.local_CNN.get_optim_policies() for parameter in policy["params"]]
        self.assertEqual(len(local_ids), len(set(local_ids)))
        self.assertEqual(set(local_ids), {id(parameter) for parameter in model.core.local_CNN.parameters() if parameter.requires_grad})

    def test_other_modules_are_mutually_exclusive_and_build_names_work(self):
        for kwargs in (dict(modified=True), dict(local_motion_mode="matching")):
            with self.assertRaisesRegex(ValueError, "Original"):
                AdaFocusFSN(context_mode="aligned", **kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_model("adafocus_context_plain", torch.device("cpu"), context_dim=8)
        self.assertEqual(model.context_module.mode, "plain")
        self.assertIsNone(model.local_motion_module)
        self.assertIsNone(model.core.fsn_interaction)

    def test_shared_loading_rejects_missing_or_mismatched_original_tensor(self):
        original = self.make("none")
        candidate = self.make("aligned")
        state = dict(original.state_dict())
        key = "core.local_CNN.new_fc.weight"
        state.pop(key)
        with self.assertRaisesRegex(ValueError, "missing_shared"):
            load_shared_adafocus_weights(candidate, state)
        state = dict(original.state_dict())
        state[key] = state[key][:1]
        with self.assertRaisesRegex(ValueError, "incompatible"):
            load_shared_adafocus_weights(candidate, state)


if __name__ == "__main__":
    unittest.main()
