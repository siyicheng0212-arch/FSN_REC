import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.relation.smoke import main, run_relation_smoke


def synthetic_evidence(seed):
    generator = torch.Generator().manual_seed(seed)
    return {"logits": torch.randn(2, 7, generator=generator),
            "global_tokens": torch.randn(2, 8, 1280, generator=generator, requires_grad=True),
            "local_tokens": torch.randn(2, 12, 2048, generator=generator, requires_grad=True),
            "global_positions": torch.linspace(0, 1, 8).expand(2, -1).clone(),
            "local_positions": torch.linspace(0, 1, 12).expand(2, -1).clone()}


class RelationSourceSmokeTests(unittest.TestCase):
    def test_optimizer_gradients_detach_and_decoder_fallback_on_cpu(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            left, right = synthetic_evidence(10), synthetic_evidence(11)
            report = run_relation_smoke(left, right)
        finally:
            torch.set_num_threads(previous)
        self.assertTrue(report["all_passed"])
        self.assertTrue(report["decoder_all_off_exact_visual_argmax"])
        self.assertEqual(report["modes"]["mlp"]["trainable_parameters"], 291521)
        self.assertEqual(report["modes"]["dual"]["trainable_parameters"], 391297)
        for mode in report["modes"].values():
            self.assertEqual(len(mode["losses"]), 3)
            self.assertTrue(mode["all_active_parameter_tensors_received_gradients"])
            self.assertTrue(mode["frozen_evidence_gradient_isolation"])
        self.assertIsNone(left["global_tokens"].grad)
        self.assertIsNone(right["local_tokens"].grad)

    def test_nonfinite_evidence_is_rejected(self):
        left, right = synthetic_evidence(10), synthetic_evidence(11)
        left["logits"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite evidence"):
            run_relation_smoke(left, right)

    def test_formal_cli_rejects_cpu_and_preserves_fresh_failure_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke.json"
            arguments = ["--manifest-dir", directory, "--cache-root", directory,
                         "--original-checkpoint", directory + "/unread.pt",
                         "--source-protocol-dir", directory, "--output", str(output),
                         "--device", "cpu"]
            with self.assertRaisesRegex(RuntimeError, "available CUDA"):
                main(arguments)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["all_passed"])
            with self.assertRaises(FileExistsError):
                main(arguments)


if __name__ == "__main__":
    unittest.main()
