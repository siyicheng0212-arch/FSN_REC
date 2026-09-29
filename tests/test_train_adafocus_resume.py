"""End-to-end two-phase resume on synthetic CPU data (no private clips)."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

from experiments import train_adafocus as trainer


class TinyDataset(Dataset):
    def __init__(self, manifest, cache_root, **_):
        self.split = Path(manifest).stem
        self.records = [SimpleNamespace(label_id=index % 7) for index in range(
            14 if self.split == "train" else 7
        )]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return {
            "video": torch.tensor([index / 14, (index % 7) / 7, 1.0]),
            "label": self.records[index].label_id,
            "clip_id": f"synthetic-{self.split}-{index}",
            "source": "synthetic",
            "duration": 1.0,
        }


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.new_fc = nn.Linear(3, 7)
        self.class_weights = None

    def forward(self, video):
        return {"logits": self.new_fc(video)}

    def set_class_weights(self, weights):
        self.class_weights = weights

    def compute_loss(self, output, target):
        return nn.functional.cross_entropy(
            output["logits"], target, weight=self.class_weights
        )


def fake_audit(*_args, **_kwargs):
    return {
        "counts": {"train": 14, "val": 7},
        "manifest_sha256": {"train": "a" * 64, "val": "b" * 64},
        "class_counts": {"train": {index: 2 for index in range(7)}},
    }


def fake_optimizer(model, args):
    groups = [{"params": list(model.parameters()), "lr": args.lr, "name": "tiny"}]
    return torch.optim.SGD(groups, lr=args.lr, momentum=0.9), [
        {"name": "tiny", "initial_lr": args.lr, "parameters": 28}
    ]


def make_args(output_dir):
    return Namespace(
        variant="original", seed=42, sampling="uniform",
        manifest_dir=Path("/tmp/synthetic-manifests"),
        cache_dir=Path("/tmp/synthetic-cache"), output_dir=Path(output_dir),
        checkpoint=None, allow_random_init=True, resume_from=None,
        head_warmup_epochs=2, head_warmup_lr=0.01,
        epochs=3, patience=10, batch_size=2, accumulation_steps=2,
        workers=0, lr=0.05, weight_decay=0.0,
        global_lr_ratio=0.5, stn_lr_ratio=0.2,
        temporal_lr_ratio=0.2, fsn_module_lr_ratio=1.0,
        class_weight_mode="none", clip_grad=20.0,
        pairwise_loss_weight=0.0, test_after_training=False,
        evidence_relation_mode="directed",
        disable_evidence_quality_gate=False,
        disable_evidence_ambiguity_gate=False,
    )


class TrainerIntegrationResumeTest(unittest.TestCase):
    def test_warmup_and_finetune_interruptions_match_uninterrupted_run(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(trainer, "FullClipDataset", TinyDataset),
                patch.object(trainer, "validate_splits", fake_audit),
                patch.object(trainer, "make_model", side_effect=lambda *_: (TinyModel(), {})),
                patch.object(trainer, "make_optimizer", fake_optimizer),
            ):
                complete = trainer.run(make_args(root / "full"), device_override=torch.device("cpu"))
                reference = torch.load(
                    root / "full/original/seed_42/last.pt", weights_only=False
                )
                for stop_phase, stop_after in (("head_warmup", 1), ("finetune", 1)):
                    with self.subTest(stop_phase=stop_phase):
                        args = make_args(root / f"interrupted-{stop_phase}")
                        original_train_epoch = trainer.train_epoch
                        calls = 0

                        def interrupted_epoch(*epoch_args, **epoch_kwargs):
                            nonlocal calls
                            phase = epoch_args[7]
                            if phase == stop_phase:
                                calls += 1
                                if calls > stop_after:
                                    raise RuntimeError("synthetic interruption")
                            return original_train_epoch(*epoch_args, **epoch_kwargs)

                        with patch.object(trainer, "train_epoch", side_effect=interrupted_epoch):
                            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                                trainer.run(args, device_override=torch.device("cpu"))
                        checkpoint = args.output_dir / "original/seed_42/last.pt"
                        self.assertTrue(checkpoint.exists())
                        if stop_phase == "head_warmup":
                            # Simulate power loss after last.pt was atomically
                            # saved but before best.pt was refreshed.
                            (checkpoint.parent / "best.pt").unlink()
                        args.resume_from = checkpoint
                        resumed = trainer.run(args, device_override=torch.device("cpu"))
                        self.assertTrue((checkpoint.parent / "best.pt").exists())
                        recovered = torch.load(checkpoint, weights_only=False)
                        self.assertEqual(resumed["resumed_from_epoch"], f"{stop_phase}_1")
                        self.assertEqual(resumed["best_epoch"], complete["best_epoch"])
                        self.assertEqual(resumed["best_val_macro_f1"], complete["best_val_macro_f1"])
                        self.assertEqual(recovered["scheduler"], reference["scheduler"])
                        for key, value in reference["model"].items():
                            self.assertTrue(torch.equal(value, recovered["model"][key]))
                        self.assertEqual(
                            [(row["phase"], row["epoch"], row["train_loss"])
                             for row in resumed["history"]],
                            [(row["phase"], row["epoch"], row["train_loss"])
                             for row in complete["history"]],
                        )

                stopped = make_args(root / "early-stop-before-result")
                stopped.patience = 1
                with patch.object(
                    trainer, "atomic_json_save", side_effect=RuntimeError("power loss")
                ):
                    with self.assertRaisesRegex(RuntimeError, "power loss"):
                        trainer.run(stopped, device_override=torch.device("cpu"))
                stopped.resume_from = stopped.output_dir / "original/seed_42/last.pt"
                saved = torch.load(stopped.resume_from, weights_only=False)
                self.assertEqual(saved["stale"], 1)
                self.assertEqual(saved["phase"], "finetune")
                resumed_stop = trainer.run(stopped, device_override=torch.device("cpu"))
                self.assertEqual(len(resumed_stop["history"]), len(saved["history"]))


if __name__ == "__main__":
    unittest.main()
