"""Reject mismatched A provenance before allocating the full visual model."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.aligned_protocol import EXPECTED_COUNTS, EXPECTED_MANIFEST_SHA
from experiments.relation.evidence import load_original, validate_original_checkpoint_protocol


class OriginalCheckpointProtocolTests(unittest.TestCase):
    @staticmethod
    def checkpoint():
        return {"model": {}, "args": {"variant": "original"}, "epoch": 16,
                "split_audit": {"counts": dict(EXPECTED_COUNTS),
                                "manifest_sha256": dict(EXPECTED_MANIFEST_SHA)}}

    def test_historical_integer_finetune_and_new_explicit_phase_are_accepted(self):
        checkpoint = self.checkpoint()
        checked = validate_original_checkpoint_protocol(checkpoint)
        self.assertEqual(checked["counts"], {"train": 7372, "val": 823})
        self.assertEqual(checked["epoch"], 16)
        checkpoint["phase"] = "finetune"
        self.assertEqual(validate_original_checkpoint_protocol(checkpoint)["phase"], "finetune")
        checked["counts"]["train"] = 0
        self.assertEqual(checkpoint["split_audit"]["counts"]["train"], 7372)

    def test_internal_split_or_same_count_changed_manifest_is_rejected(self):
        checkpoint = self.checkpoint()
        checkpoint["split_audit"]["counts"] = {"train": 6586, "val": 786}
        with self.assertRaisesRegex(ValueError, "fixed full"):
            validate_original_checkpoint_protocol(checkpoint)
        checkpoint = self.checkpoint()
        checkpoint["split_audit"]["manifest_sha256"]["val"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "manifest SHA mismatch"):
            validate_original_checkpoint_protocol(checkpoint)

    def test_missing_audit_fields_fail_with_actionable_message(self):
        for checkpoint in ({"epoch": 16}, {"epoch": 16, "split_audit": None}):
            with self.assertRaisesRegex(ValueError, "locate the matching full-protocol"):
                validate_original_checkpoint_protocol(checkpoint)
        for key in ("counts", "manifest_sha256"):
            checkpoint = self.checkpoint()
            del checkpoint["split_audit"][key]
            with self.assertRaisesRegex(ValueError, "split_audit requires"):
                validate_original_checkpoint_protocol(checkpoint)

    def test_warmup_or_unknown_epoch_cannot_serve_as_formal_A(self):
        for epoch in ("head_warmup_5", "module_warmup_2", "finetune_16", 0, -1, True, 16.0, None):
            checkpoint = self.checkpoint()
            checkpoint["epoch"] = epoch
            with self.assertRaisesRegex(ValueError, "positive integer finetune"):
                validate_original_checkpoint_protocol(checkpoint)
        for phase in ("head_warmup", "module_warmup", None, "unknown"):
            checkpoint = self.checkpoint()
            checkpoint["phase"] = phase
            with self.assertRaisesRegex(ValueError, "not head/module warmup"):
                validate_original_checkpoint_protocol(checkpoint)

    def test_formal_loader_rejects_missing_provenance_before_model_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save({"model": {}, "args": {"variant": "original"}}, path)
            with patch("experiments.model_wrappers.AdaFocusFSN") as constructor:
                with self.assertRaisesRegex(ValueError, "split_audit"):
                    load_original(path, require_full_protocol=True)
                constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
