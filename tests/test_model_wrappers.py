import unittest

import torch

from experiments.model_wrappers import build_model, load_shared_adafocus_weights


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


if __name__ == "__main__":
    unittest.main()
