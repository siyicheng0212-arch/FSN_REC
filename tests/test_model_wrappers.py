import unittest
from types import SimpleNamespace

import torch

from experiments.model_wrappers import AdaFocusFSN, build_model, load_shared_adafocus_weights


class AdaFocusWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(min(6, __import__("os").cpu_count() or 1))

    def test_modified_model_is_output_equivalent_at_initialization(self):
        torch.manual_seed(11)
        original = build_model("adafocus_original", torch.device("cpu")).eval()
        state = {key: value.detach().clone() for key, value in original.state_dict().items()}
        torch.manual_seed(23)
        modified = build_model("adafocus_fsn", torch.device("cpu")).eval()
        report = load_shared_adafocus_weights(modified, state)
        frames = torch.rand(1, 8, 3, 224, 224)
        with torch.no_grad():
            original_logits = original(frames)["logits"]
            modified_logits = modified(frames)["logits"]
        self.assertLessEqual(float((original_logits - modified_logits).abs().max()), 1e-5)
        self.assertGreater(len(report["loaded_keys"]), 0)
        self.assertGreater(len(report["missing_keys"]), 0)
        self.assertEqual(modified.core.fsn_interaction.gamma.item(), 0.0)
        self.assertEqual(modified.core.fsn_interaction.beta.item(), 1.0)

    def test_optimizer_parameter_references_are_unique(self):
        model = build_model("adafocus_fsn", torch.device("cpu"))
        identifiers = [id(parameter) for parameter in model.parameters()]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_local_motion_is_opt_in_and_preserves_original_initial_outputs(self):
        torch.manual_seed(11)
        original = build_model("adafocus_original", torch.device("cpu")).eval()
        state = {key: value.detach().clone() for key, value in original.state_dict().items()}
        self.assertIsNone(original.local_motion_module)
        self.assertFalse(any("local_motion" in key for key in state))
        frames = torch.rand(1, 8, 3, 224, 224)
        torch.manual_seed(101)
        with torch.no_grad():
            original_logits = original(frames)["logits"]
        for name in ("adafocus_local_motion", "adafocus_local_motion_context", "adafocus_local_appearance"):
            with self.subTest(model=name):
                torch.manual_seed(11)
                model = build_model(name, torch.device("cpu")).eval()
                # Initializing only the added branch cannot alter shared heads
                # or policy tensors under the same seed.
                candidate_state = model.state_dict()
                for key, value in state.items():
                    torch.testing.assert_close(candidate_state[key], value, rtol=0, atol=0)
                report = load_shared_adafocus_weights(model, state)
                self.assertTrue(all("local_motion" in key for key in report["missing_keys"]))
                self.assertEqual(len(report["loaded_keys"]), len(state))
                self.assertIsNone(model.core.local_CNN.local_adapter)
                self.assertIsNone(model.core.fsn_interaction)
                torch.manual_seed(101)
                with torch.no_grad():
                    logits = model(frames)["logits"]
                torch.testing.assert_close(logits, original_logits, rtol=0, atol=1e-5)
                model.set_local_motion_enabled(False)
                torch.manual_seed(101)
                with torch.no_grad():
                    disabled_logits = model(frames)["logits"]
                torch.testing.assert_close(disabled_logits, original_logits, rtol=0, atol=1e-5)

    def test_motion_optimizer_policies_cover_parameters_once(self):
        model = build_model("adafocus_local_motion_context", torch.device("cpu"))
        args = SimpleNamespace(temporal_lr_ratio=.2, stn_lr_ratio=.2,
                               global_lr_ratio=.5, local_motion_lr_ratio=2.)
        policies = model.core.get_optim_policies(args)
        identifiers = [id(parameter) for policy in policies for parameter in policy["params"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(set(identifiers), {id(parameter) for parameter in model.parameters()})
        motion_group = next(policy for policy in policies if policy["name"] == "local_motion")
        self.assertEqual(motion_group["lr_mult"], 2.)
        self.assertEqual({id(parameter) for parameter in motion_group["params"]},
                         {id(parameter) for parameter in model.local_motion_module.parameters()})
        # The lower-level TSN API must include GroupNorm/direct parameters too.
        local_policies = model.core.local_CNN.get_optim_policies()
        local_ids = [id(parameter) for policy in local_policies for parameter in policy["params"]]
        self.assertEqual(len(local_ids), len(set(local_ids)))
        self.assertTrue({id(parameter) for parameter in model.local_motion_module.parameters()}.issubset(local_ids))

    def test_training_branches_get_same_clip_context_and_selected_positions(self):
        model = build_model("adafocus_local_motion_context", torch.device("cpu")).train()
        observed = []

        def inspect_inputs(module, args, kwargs):
            observed.append((kwargs["global_context"].detach().clone(), kwargs["positions"].detach().clone()))

        handle = model.local_motion_module.register_forward_pre_hook(inspect_inputs, with_kwargs=True)
        try:
            output = model(torch.rand(1, 8, 3, 224, 224))
        finally:
            handle.remove()
        self.assertEqual(len(observed), 1)
        context, positions = observed[0]
        self.assertEqual(context.shape[:3], (2, 4, 1280))
        self.assertEqual(positions.shape, (2, 4))
        torch.testing.assert_close(context[0], context[1])
        torch.testing.assert_close(positions[0], positions[1])
        self.assertTrue(bool((positions[:, 1:] >= positions[:, :-1]).all()))
        self.assertIn("policy_branch", output)

    def test_local_auxiliary_loss_reaches_motion_with_final_head_detached(self):
        model = build_model("adafocus_local_motion", torch.device("cpu")).train()
        local = model.core.local_CNN
        final_logits, auxiliary_logits = local(torch.rand(1, 12, 96, 96),
                                              positions=torch.linspace(0, 1, 4).unsqueeze(0))
        final_logits.square().sum().backward(retain_graph=True)
        self.assertTrue(all(parameter.grad is None for parameter in model.local_motion_module.parameters()))
        model.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(auxiliary_logits, torch.tensor([1])).backward()
        gradients = [parameter.grad for parameter in model.local_motion_module.parameters()]
        self.assertTrue(any(gradient is not None and bool(gradient.abs().sum() > 0) for gradient in gradients))

    def test_legacy_and_new_modules_cannot_be_combined(self):
        with self.assertRaisesRegex(ValueError, "Original"):
            AdaFocusFSN(modified=True, local_motion_mode="matching")


if __name__ == "__main__":
    unittest.main()
