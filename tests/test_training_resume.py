"""Epoch-boundary resume integrity without CUDA or private video data."""

import random
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments.training_resume import (
    RESUME_SCHEMA,
    atomic_json_save,
    atomic_torch_save,
    capture_rng_state,
    load_epoch_checkpoint,
    protocol_fingerprint,
    restore_rng_state,
)


class TrainingResumeTest(unittest.TestCase):
    def test_protocol_rejects_different_seed_or_manifest(self):
        args = Namespace(
            variant="directional_evidence", seed=42, cache_dir=Path("/tmp/fsn-cache"),
            lr=0.002, epochs=100,
        )
        audit = {"manifest_sha256": {"train": "a" * 64, "val": "b" * 64}}
        fingerprint = protocol_fingerprint(args, audit, "c" * 64)
        args.seed = 2026
        self.assertNotEqual(fingerprint, protocol_fingerprint(args, audit, "c" * 64))
        args.seed = 42
        changed = {"manifest_sha256": {"train": "d" * 64, "val": "b" * 64}}
        self.assertNotEqual(fingerprint, protocol_fingerprint(args, changed, "c" * 64))
        self.assertNotEqual(
            protocol_fingerprint(args, audit, "c" * 64, "commit-a"),
            protocol_fingerprint(args, audit, "c" * 64, "commit-b"),
        )

    def test_atomic_save_keeps_previous_file_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            atomic_torch_save({"epoch": 1}, path)
            with patch("experiments.training_resume.torch.save", side_effect=RuntimeError("disk full")):
                with self.assertRaisesRegex(RuntimeError, "disk full"):
                    atomic_torch_save({"epoch": 2}, path)
            self.assertEqual(torch.load(path, weights_only=False)["epoch"], 1)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_result_json_is_replaced_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            atomic_json_save({"epoch": 1}, path)
            self.assertEqual(path.read_text(), '{\n  "epoch": 1\n}\n')
            atomic_json_save({"epoch": 2}, path)
            self.assertEqual(path.read_text(), '{\n  "epoch": 2\n}\n')

    def test_old_best_is_rejected_and_full_state_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            atomic_torch_save({"model": {}, "epoch": 1}, path)
            with self.assertRaisesRegex(ValueError, "older best.pt"):
                load_epoch_checkpoint(path, "protocol")
            state = {
                "schema_version": RESUME_SCHEMA,
                "protocol_fingerprint": "protocol",
                "phase": "finetune", "epoch": 4,
                "model": {}, "optimizer": {}, "scheduler": {"last_epoch": 4},
                "best_epoch": "finetune_2", "best_val_macro_f1": 0.7,
                "best_is_current": False, "stale": 2, "history": [],
                "rng": {}, "load_report": {}, "optimizer_groups": [],
            }
            atomic_torch_save(state, path)
            self.assertEqual(load_epoch_checkpoint(path, "protocol")["epoch"], 4)
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                load_epoch_checkpoint(path, "changed")

    def test_rng_optimizer_and_scheduler_continue_from_same_epoch(self):
        random.seed(9)
        np.random.seed(9)
        torch.manual_seed(9)
        sampler = torch.Generator().manual_seed(19)
        workers = {"train": torch.Generator().manual_seed(29)}
        augmentation = torch.Generator().manual_seed(39)
        model = nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        def one_epoch(network, opt, schedule, sample_gen, aug_gen):
            order = torch.randperm(4, generator=sample_gen)
            features = torch.rand(4, 3)[order]
            features = features * (1 + torch.rand((), generator=aug_gen))
            target = torch.tensor([0, 1, 1, 0])[order]
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(network(features), target)
            loss.backward()
            opt.step()
            schedule.step()
            return float(loss.detach())

        one_epoch(model, optimizer, scheduler, sampler, augmentation)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            atomic_torch_save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": capture_rng_state(sampler, workers, augmentation),
            }, path)
            uninterrupted_loss = one_epoch(model, optimizer, scheduler, sampler, augmentation)
            uninterrupted = {key: value.detach().clone() for key, value in model.state_dict().items()}

            saved = torch.load(path, weights_only=False)
            resumed = nn.Linear(3, 2)
            resumed_optimizer = torch.optim.SGD(resumed.parameters(), lr=0.1, momentum=0.9)
            resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_optimizer, T_max=5)
            resumed_sampler = torch.Generator()
            resumed_workers = {"train": torch.Generator()}
            resumed_aug = torch.Generator()
            resumed.load_state_dict(saved["model"])
            resumed_optimizer.load_state_dict(saved["optimizer"])
            resumed_scheduler.load_state_dict(saved["scheduler"])
            restore_rng_state(saved["rng"], resumed_sampler, resumed_workers, resumed_aug)
            resumed_loss = one_epoch(
                resumed, resumed_optimizer, resumed_scheduler, resumed_sampler, resumed_aug
            )
            self.assertEqual(uninterrupted_loss, resumed_loss)
            for key, value in uninterrupted.items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[key]))
            self.assertEqual(scheduler.state_dict(), resumed_scheduler.state_dict())


if __name__ == "__main__":
    unittest.main()
