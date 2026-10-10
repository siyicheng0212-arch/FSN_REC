"""Offline checks for the pinned OpenGVLab VideoMAE V2 student adapter."""

from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest.mock import patch
import json
import os

import torch

from experiments.modern_baselines import videomae2 as baseline
from experiments.modern_baselines import train_videomae2 as trainer


class TinySevenClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(768, 7)


def distilled_state_for(model):
    state = {key: tensor.detach().half().clone()
             for key, tensor in model.state_dict().items() if not key.startswith("head.")}
    state["head.weight"] = torch.ones(710, 768, dtype=torch.float16)
    state["head.bias"] = torch.ones(710, dtype=torch.float16)
    return state


class OfflinePretrainingTest(unittest.TestCase):
    def test_exact_official_checkout_when_present(self):
        source = Path(os.environ.get("FSN_OFFICIAL_VIDEO_SOURCE",
                      Path(__file__).resolve().parents[2] / "reference_VideoMAEv2"))
        if not source.is_dir():
            self.skipTest("official source checkout is intentionally external to this repository")
        model_file = baseline.verify_official_source(source)
        self.assertEqual(baseline.sha256_file(model_file), baseline.OFFICIAL_MODEL_SHA256)

    def test_exact_published_student_container_when_present(self):
        path = Path(os.environ.get("FSN_OFFICIAL_VIDEO_CHECKPOINT",
                    Path(__file__).resolve().parents[2] / "reference_weights" / baseline.OFFICIAL_CHECKPOINT_NAME))
        if not path.is_file():
            self.skipTest("published weight file is intentionally external to this repository")
        audit = baseline.verify_official_checkpoint(path)
        self.assertEqual(audit["checkpoint_sha256"], baseline.OFFICIAL_CHECKPOINT_SHA256)
        state = baseline._checkpoint_state(path)
        self.assertEqual(len(state), 162)
        self.assertEqual(tuple(state["patch_embed.proj.weight"].shape), (768, 3, 2, 16, 16))
        self.assertEqual(tuple(state["head.weight"].shape), (710, 768))
        self.assertEqual(tuple(state["head.bias"].shape), (710,))
        self.assertFalse(any(key.startswith("pos_embed") for key in state))

    def test_checkpoint_hash_mismatch_fails_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / baseline.OFFICIAL_CHECKPOINT_NAME
            path.write_bytes(b"not the official weight")
            with patch.object(baseline.torch, "load") as load:
                with self.assertRaisesRegex(baseline.BaselinePreflightError, "SHA256 mismatch"):
                    baseline.verify_official_checkpoint(path)
                load.assert_not_called()

    def test_module_container_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / baseline.OFFICIAL_CHECKPOINT_NAME
            path.write_bytes(b"test checkpoint bytes")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(baseline, "OFFICIAL_CHECKPOINT_SHA256", digest), \
                    patch.object(baseline.torch, "load", return_value={"model": {}}) as load:
                with self.assertRaisesRegex(baseline.BaselinePreflightError, "module container"):
                    baseline._checkpoint_state(path)
                self.assertEqual(len(load.call_args.args), 1)
                self.assertFalse(isinstance(load.call_args.args[0], (str, Path)))
                self.assertTrue(load.call_args.kwargs["weights_only"])

    def test_source_executes_only_the_verified_bytes(self):
        safe = b"def vit_base_patch16_224():\n    return 'safe'\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "modeling_finetune.py"
            path.write_bytes(b"def vit_base_patch16_224():\n    return 'unsafe'\n")
            with patch.object(baseline.importlib.metadata, "version", return_value="0.4.12"), \
                    patch.object(baseline, "OFFICIAL_MODEL_SHA256", hashlib.sha256(safe).hexdigest()), \
                    patch.object(Path, "read_bytes", return_value=safe):
                constructor = baseline._official_constructor(path)
            self.assertEqual(constructor(), "safe")

    def test_all_backbone_tensors_load_and_seven_class_head_is_only_exception(self):
        model = TinySevenClass()
        old_head = model.head.weight.detach().clone()
        state = distilled_state_for(model)
        state["backbone.weight"].fill_(0.375)
        with patch.object(baseline, "EXPECTED_BACKBONE_PARAMETERS", 6):
            report = baseline.load_strict_pretrained_state(model, state)
        self.assertEqual(report["loaded_backbone_tensors"], 2)
        self.assertEqual(report["loaded_backbone_parameters"], 6)
        self.assertEqual(report["reinitialized_fsn_head_parameters"], 5383)
        self.assertTrue(torch.equal(model.backbone.weight, torch.full_like(model.backbone.weight, 0.375)))
        self.assertTrue(torch.equal(model.head.weight, old_head))

    def test_partial_or_corrupt_pretraining_is_rejected(self):
        model = TinySevenClass()
        cases = []
        absent = distilled_state_for(model)
        del absent["backbone.bias"]
        cases.append(absent)
        extra = distilled_state_for(model)
        extra["another.weight"] = torch.ones(1, dtype=torch.float16)
        cases.append(extra)
        wrong_shape = distilled_state_for(model)
        wrong_shape["backbone.weight"] = torch.ones(3, 2, dtype=torch.float16)
        cases.append(wrong_shape)
        bad_head = distilled_state_for(model)
        bad_head["head.weight"] = torch.ones(400, 768, dtype=torch.float16)
        cases.append(bad_head)
        nonfinite = distilled_state_for(model)
        nonfinite["backbone.bias"][0] = float("nan")
        cases.append(nonfinite)
        with patch.object(baseline, "EXPECTED_BACKBONE_PARAMETERS", 6):
            for state in cases:
                with self.subTest(kind=len(state), head=tuple(state["head.weight"].shape)):
                    with self.assertRaises(baseline.BaselinePreflightError):
                        baseline.load_strict_pretrained_state(model, state)

    def test_input_view_is_exact_fixed_selection_and_official_normalization(self):
        video = torch.zeros(36, 3, 224, 224, dtype=torch.uint8)
        for index in range(36):
            video[index, 0].fill_(index)
        result = baseline.preprocess_cached_video(video)
        self.assertEqual(tuple(result.shape), (3, 16, 224, 224))
        self.assertEqual(baseline.FRAME_INDICES,
                         (1, 3, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34))
        for selected, index in enumerate(baseline.FRAME_INDICES):
            self.assertAlmostEqual(result[0, selected, 0, 0].item(),
                                   (index / 255 - baseline.MEAN[0]) / baseline.STD[0],
                                   places=5)
        self.assertEqual(video[1, 0, 0, 0].item(), 1)
        with self.assertRaisesRegex(baseline.BaselinePreflightError, "uint8"):
            baseline.preprocess_cached_video(video.float())

    def test_only_frozen_manifest_hashes_can_start_data_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cache").mkdir()
            (root / "train.jsonl").write_text("{}\n")
            (root / "val.jsonl").write_text("{}\n")
            (root / "test.jsonl").write_text("intentionally invalid test\n")
            with patch.object(baseline, "ResolvedFullClipDataset") as dataset:
                with self.assertRaisesRegex(baseline.BaselinePreflightError, "train manifest SHA256"):
                    baseline.audit_frozen_data(root, root / "cache")
                dataset.assert_not_called()

    def test_actual_cuda_bf16_smoke_does_not_accept_cpu(self):
        class FrozenTrain:
            split = "train"

            def __len__(self):
                return 7372

        # A short fake dataset is rejected before any GPU/model activity.
        with self.assertRaises(baseline.BaselinePreflightError):
            baseline.smoke_bf16_batch(TinySevenClass(), FrozenTrain(),
                                      device=torch.device("cpu"), batch_size=4)

    def test_finetuned_checkpoint_requires_pretraining_provenance_and_real_steps(self):
        model = TinySevenClass()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fsn.pt"
            payload = {
                "schema": baseline.FINETUNED_SCHEMA,
                "training_completed": True,
                "optimizer_steps": 0,
                "finetune_optimizer_steps": 5,
                "pretrained_checkpoint_sha256": baseline.OFFICIAL_CHECKPOINT_SHA256,
                "official_source_commit": baseline.OFFICIAL_COMMIT,
                "data_protocol": baseline.DATA_PROTOCOL,
                "manifest_sha256": baseline.EXPECTED_MANIFEST_SHA,
                "class_names": list(baseline.DEFAULT_CLASS_NAMES),
                "model": model.state_dict(),
            }
            torch.save(payload, path)
            digest = baseline.sha256_file(path)
            with self.assertRaisesRegex(baseline.BaselinePreflightError, "official pretraining"):
                baseline.load_finetuned_fsn_checkpoint(model, path, digest)
            model._fsn_videomae2_pretraining = {"verified": True}
            with self.assertRaisesRegex(baseline.BaselinePreflightError, "optimizer steps"):
                baseline.load_finetuned_fsn_checkpoint(model, path, digest)
            payload["optimizer_steps"] = 10
            torch.save(payload, path)
            digest = baseline.sha256_file(path)
            result = baseline.load_finetuned_fsn_checkpoint(model, path, digest)
            self.assertEqual(result["optimizer_steps"], 10)
            self.assertEqual(result["finetune_optimizer_steps"], 5)
            self.assertEqual(model._fsn_finetuned_checkpoint_sha256, digest)


class TinyVideoClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = torch.nn.Module()
        self.patch_embed.proj = torch.nn.Linear(1, 768)
        self.blocks = torch.nn.ModuleList([torch.nn.Identity()])
        self.fc_norm = torch.nn.LayerNorm(768)
        self.head = torch.nn.Linear(768, 7)

    def forward(self, video):
        hidden = self.patch_embed.proj(video)
        return self.head(self.fc_norm(hidden))


class TinyTrainingDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 7

    def __getitem__(self, index):
        return {"video": torch.tensor([float(index + 1)]),
                "label": index, "clip_id": f"clip-{index}"}


class FixedTrainerTest(unittest.TestCase):
    def setUp(self):
        self.config = trainer.TrainConfig(batch_size=2, accumulation_steps=2,
                                          warmup_epochs=1, epochs=3, workers=0)

    def test_optimizer_groups_cover_all_parameters_and_switch_schedule(self):
        model = TinyVideoClassifier()
        optimizer, scheduler, summary = trainer.make_optimizer_scheduler(model, self.config)
        grouped_ids = [id(parameter) for group in optimizer.param_groups
                       for parameter in group["params"]]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(set(grouped_ids), {id(p) for p in model.parameters()})
        self.assertEqual(sum(group["parameters"] for group in summary),
                         sum(p.numel() for p in model.parameters()))
        head_lrs = [group["lr"] for group in optimizer.param_groups if group["head"]]
        backbone_lrs = [group["lr"] for group in optimizer.param_groups if not group["head"]]
        self.assertTrue(all(abs(value - self.config.head_warmup_lr) < 1e-12
                            for value in head_lrs))
        self.assertTrue(all(value == 0 for value in backbone_lrs))
        trainer._set_phase(model, "head_warmup", optimizer)
        self.assertTrue(all(group["weight_decay"] == 0 for group in optimizer.param_groups))
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()
        self.assertTrue(all(group["lr"] > 0 for group in optimizer.param_groups))
        self.assertEqual(scheduler.last_epoch, 1)
        trainer._set_phase(model, "finetune", optimizer)
        self.assertTrue(all(group["weight_decay"] == group["finetune_weight_decay"]
                            for group in optimizer.param_groups))

    def test_complete_epoch_has_exact_weight_mass_and_partial_final_window(self):
        model = TinyVideoClassifier()
        optimizer, scheduler, _ = trainer.make_optimizer_scheduler(model, self.config)
        trainer._set_phase(model, "finetune")
        generator = torch.Generator().manual_seed(42)
        loader = torch.utils.data.DataLoader(
            TinyTrainingDataset(), batch_size=self.config.batch_size,
            shuffle=True, generator=generator, drop_last=False)
        augmentation = torch.Generator().manual_seed(43)
        weights = torch.ones(7)
        report = trainer.train_epoch(model, loader, optimizer, weights,
                                     torch.device("cpu"), self.config, augmentation)
        self.assertEqual(report["samples_seen"], 7)
        self.assertEqual(report["optimizer_steps"], 2)
        self.assertEqual(report["effective_batch_min"], 3)
        self.assertEqual(report["effective_batch_max"], 4)
        self.assertGreater(report["first_step_gradients"]["head"], 0)
        self.assertGreater(report["first_step_gradients"]["patch_embed"], 0)

    def test_resume_checkpoint_restores_model_optimizer_scheduler_and_all_rng(self):
        trainer._seed_all(42)
        model = TinyVideoClassifier()
        optimizer, scheduler, _ = trainer.make_optimizer_scheduler(model, self.config)
        trainer._set_phase(model, "finetune")
        shuffle = torch.Generator().manual_seed(42)
        augmentation = torch.Generator().manual_seed(43)
        loader = torch.utils.data.DataLoader(
            TinyTrainingDataset(), batch_size=2, shuffle=True, generator=shuffle)
        trainer.train_epoch(model, loader, optimizer, torch.ones(7),
                            torch.device("cpu"), self.config, augmentation)
        scheduler.step()
        identity = {"identity_sha256": "a" * 64}
        saved = trainer._checkpoint(
            model, optimizer, scheduler, shuffle, augmentation, identity,
            completed_epochs=1, optimizer_steps=2, finetune_optimizer_steps=0,
            best_score=None, best_epoch=None, best_candidate=None,
            best_candidate_sha256=None, stale=0, history=[{"epoch": 1}],
            peak_cuda_bytes=0)
        expected_weight = model.head.weight.detach().clone()
        expected_shuffle = shuffle.get_state().clone()
        expected_augmentation = augmentation.get_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            trainer._atomic_torch_save(output / "last.pt", saved)
            with self.assertRaisesRegex(baseline.BaselinePreflightError, "protocol/epoch/history"):
                trainer._restore_checkpoint(
                    output, model, optimizer, scheduler, shuffle, augmentation,
                    {"identity_sha256": "b" * 64}, self.config)
            with torch.no_grad():
                model.head.weight.add_(10)
            shuffle.manual_seed(99)
            augmentation.manual_seed(99)
            scheduler.step()
            restored = trainer._restore_checkpoint(
                output, model, optimizer, scheduler, shuffle, augmentation,
                identity, self.config)
        self.assertEqual(restored["completed_epochs"], 1)
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertTrue(torch.equal(model.head.weight, expected_weight))
        self.assertTrue(torch.equal(shuffle.get_state(), expected_shuffle))
        self.assertTrue(torch.equal(augmentation.get_state(), expected_augmentation))
        self.assertTrue(optimizer.state)

    def test_failed_atomic_checkpoint_keeps_previous_complete_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "last.pt"
            target.write_bytes(b"previous complete epoch")
            with patch.object(trainer.torch, "save", side_effect=RuntimeError("disk full")):
                with self.assertRaisesRegex(RuntimeError, "disk full"):
                    trainer._atomic_torch_save(target, {"epoch": 2})
            self.assertEqual(target.read_bytes(), b"previous complete epoch")

    def test_lock_and_status_capture_concurrent_run_and_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with trainer._lock_run(output):
                with self.assertRaisesRegex(baseline.BaselinePreflightError, "another trainer"):
                    trainer._lock_run(output)
            trainer._status(output, "failed", exit_code=1, completed_epochs=3,
                            resume_available=True, error_type="RuntimeError")
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["exit_code"], 1)
            self.assertEqual(status["completed_epochs"], 3)
            self.assertTrue(status["resume_available"])
            self.assertFalse(status["test_read"])


class TinyValidationDataset(torch.utils.data.Dataset):
    split = "val"
    manifest_sha256 = baseline.EXPECTED_MANIFEST_SHA["val"]

    def __init__(self, duplicate=False):
        self.duplicate = duplicate

    def __len__(self):
        return 823

    def __getitem__(self, index):
        return {"video": torch.tensor([index % 7], dtype=torch.float32),
                "label": index % 7,
                "clip_id": f"clip-{0 if self.duplicate else index}",
                "source": "clinical", "duration": 2.0}


class IdentityClassify(torch.nn.Module):
    def forward(self, video):
        return torch.nn.functional.one_hot(video[:, 0].long(), 7).float()


class SingleViewEvaluationTest(unittest.TestCase):
    def test_untrained_seven_class_head_never_becomes_formal_baseline(self):
        with self.assertRaisesRegex(baseline.BaselinePreflightError, "FSN-trained"):
            baseline.evaluate_single_view(IdentityClassify(), TinyValidationDataset(),
                                          device=torch.device("cpu"))

    def test_complete_single_view_development_validation(self):
        model = IdentityClassify()
        model._fsn_finetuned_checkpoint_sha256 = "a" * 64
        report = baseline.evaluate_single_view(model, TinyValidationDataset(),
                                               device=torch.device("cpu"), batch_size=32)
        self.assertEqual(report["clips_evaluated"], 823)
        self.assertEqual(report["views_per_clip"], 1)
        self.assertEqual(report["metrics"]["all"]["accuracy"], 1.0)
        self.assertFalse(report["test_read"])

    def test_duplicate_clip_cannot_be_counted_twice(self):
        model = IdentityClassify()
        model._fsn_finetuned_checkpoint_sha256 = "a" * 64
        with self.assertRaisesRegex(baseline.BaselinePreflightError, "duplicate clip"):
            baseline.evaluate_single_view(model, TinyValidationDataset(duplicate=True),
                                          device=torch.device("cpu"), batch_size=16)


if __name__ == "__main__":
    unittest.main()
