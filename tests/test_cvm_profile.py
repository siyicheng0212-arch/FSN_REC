"""CPU deployment-path checks; these are not CUDA resource measurements."""

import argparse
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn

from cvm.models import ClinicalHierarchyModel, prediction_probabilities
from cvm.profile import (DeploymentWrapper, batch_sizes, declared_input_shape,
                         load_deployment_checkpoint, parameter_counts, parser,
                         profile_cuda_batch, run_profile, synthetic_normalized_input)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 8)

    def forward(self, video):
        return self.projection(video.mean(dim=(2, 3, 4)))


class CountedHead(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head
        self.calls = 0

    def forward(self, features):
        self.calls += 1
        return self.head(features)


def model_for(mode):
    torch.manual_seed(42)
    return ClinicalHierarchyModel(Backbone(), 8, mode=mode).eval()


class ProfileTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.inputs = torch.randn(3, 3, 2, 4, 4)
        self.config = {"backbone": "r2plus1d_18", "preprocessing": {
            "frames": 16, "crop_size": [112, 112], "mean": [.4, .5, .6], "std": [.2, .3, .4]}}

    def test_flat_deployment_matches_audit_forward_and_skips_extra_heads(self):
        for mode in ("flat", "capacity_control", "aux_flat"):
            model = model_for(mode)
            model.flat_head = CountedHead(model.flat_head)
            model.group_head = CountedHead(model.group_head)
            for index, group in enumerate(model.taxonomy.groups):
                if len(group) > 1:
                    model.conditional_heads[index] = CountedHead(model.conditional_heads[index])
            if model.capacity_head is not None:
                model.capacity_head = CountedHead(model.capacity_head)
            expected = prediction_probabilities(model(self.inputs), mode, model.taxonomy)
            for module in model.modules():
                if isinstance(module, CountedHead):
                    module.calls = 0
            deployment = DeploymentWrapper(model).eval()
            actual = deployment(self.inputs)
            self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(model.flat_head.calls, 1)
            self.assertEqual(model.group_head.calls, 0)
            for index, group in enumerate(model.taxonomy.groups):
                if len(group) > 1:
                    self.assertEqual(model.conditional_heads[index].calls, 0)
            if model.capacity_head is not None:
                self.assertEqual(model.capacity_head.calls, 0)
            names = [name for name, _ in deployment.named_parameters()]
            self.assertTrue(all(name.startswith(("backbone.", "flat_head.")) for name in names))

    def test_hierarchy_deployment_matches_both_decoders_and_omits_flat_head(self):
        model = model_for("hierarchy")
        model.flat_head = CountedHead(model.flat_head)
        outputs = model(self.inputs)
        model.flat_head.calls = 0
        for decoder in ("soft", "hard"):
            deployment = DeploymentWrapper(model, decoder).eval()
            expected = prediction_probabilities(outputs, "hierarchy", model.taxonomy, decoder)
            self.assertTrue(torch.equal(expected, deployment(self.inputs)))
            self.assertEqual(model.flat_head.calls, 0)
            self.assertFalse(any(name.startswith("flat_head.") for name, _ in deployment.named_parameters()))

    def test_counts_distinguish_stored_active_and_trainable_parameters(self):
        flat = model_for("flat")
        counts = parameter_counts(flat, DeploymentWrapper(flat))
        self.assertGreater(counts["stored_model_parameters"], counts["deployment_active_parameters"])
        self.assertEqual(counts["training_trainable_parameters"], counts["deployment_active_parameters"])
        auxiliary = model_for("aux_flat")
        aux_counts = parameter_counts(auxiliary, DeploymentWrapper(auxiliary))
        self.assertGreater(aux_counts["training_trainable_parameters"], aux_counts["deployment_active_parameters"])
        auxiliary.configure_trainable(backbone_trainable=False)
        frozen = parameter_counts(auxiliary, DeploymentWrapper(auxiliary))
        backbone_parameters = sum(parameter.numel() for parameter in auxiliary.backbone.parameters())
        self.assertEqual(frozen["training_trainable_parameters"], aux_counts["training_trainable_parameters"] - backbone_parameters)
        self.assertEqual(frozen["deployment_active_parameters"], aux_counts["deployment_active_parameters"])
        flat.configure_trainable(backbone_trainable=False)
        flat_frozen = parameter_counts(flat, DeploymentWrapper(flat))
        self.assertLess(flat_frozen["training_trainable_parameters"], flat_frozen["deployment_active_parameters"])

    def test_synthetic_input_shape_normalization_and_seed(self):
        first = synthetic_normalized_input(self.config, 2, torch.device("cpu"), seed=13)
        second = synthetic_normalized_input(self.config, 2, torch.device("cpu"), seed=13)
        self.assertEqual(first.shape, (2, 3, 16, 112, 112))
        self.assertTrue(torch.equal(first, second))
        mean = first.new_tensor(self.config["preprocessing"]["mean"]).view(1, 3, 1, 1, 1)
        std = first.new_tensor(self.config["preprocessing"]["std"]).view(1, 3, 1, 1, 1)
        rgb = first * std + mean
        self.assertTrue(bool((rgb >= 0).all() and (rgb <= 1).all()))
        self.config["preprocessing"]["frames"] = 32
        self.assertEqual(declared_input_shape(self.config, 1), (1, 3, 32, 112, 112))
        self.config["backbone"] = "mvit_v2_s"
        with self.assertRaises(ValueError):
            declared_input_shape(self.config, 1)

    def test_cpu_never_becomes_fake_cuda_measurement(self):
        with self.assertRaises(RuntimeError):
            profile_cuda_batch(DeploymentWrapper(model_for("flat")), self.inputs)

    def test_bad_decoder_and_batch_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            DeploymentWrapper(model_for("flat"), "hard")
        with self.assertRaises(ValueError):
            DeploymentWrapper(model_for("hierarchy"), "flat")
        for value in ("0", "1,1", "wrong", ""):
            with self.assertRaises(argparse.ArgumentTypeError):
                batch_sizes(value)
        self.assertEqual(batch_sizes("1,4"), [1, 4])
        args = parser().parse_args(["--checkpoint", "best.pt", "--output", "resources.json"])
        self.assertEqual((args.warmup, args.iterations), (20, 100))

    def test_existing_resource_file_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "resources.json"
            output.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_profile(argparse.Namespace(output=output))
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def test_checkpoint_source_hash_mismatch_fails_before_build(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save({"model": {}, "config": {"code_hash": "old"}}, path)
            with patch("cvm.profile.source_digest", return_value="current"):
                with self.assertRaises(RuntimeError):
                    load_deployment_checkpoint(path)

    def test_unselected_or_warmup_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            torch.save({"model": {}, "config": {"code_hash": "current"},
                        "stage": "warmup", "epoch": 1, "best_epoch": 1,
                        "selected_val_macro_f1": .8}, path)
            with patch("cvm.profile.source_digest", return_value="current"):
                with self.assertRaises(ValueError):
                    load_deployment_checkpoint(path)

    def test_selected_checkpoint_restores_only_declared_deployment(self):
        model = model_for("aux_flat")
        config = {**self.config, "code_hash": "current", "mode": "aux_flat", "main_decoder": "flat",
                  "backbone_training": "finetune", "parameters_trainable_finetune":
                  sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save({"model": model.state_dict(), "config": config, "stage": "finetune",
                        "epoch": 7, "best_epoch": 7, "selected_val_macro_f1": .8}, path)
            with patch("cvm.profile.source_digest", return_value="current"), \
                    patch("cvm.profile.model_arguments", return_value={}), \
                    patch("cvm.profile.build_model", return_value=model_for("aux_flat")):
                deployment, _, metadata = load_deployment_checkpoint(path)
            expected = prediction_probabilities(model(self.inputs), "aux_flat", model.taxonomy)
            self.assertTrue(torch.equal(expected, deployment(self.inputs)))
            self.assertEqual(metadata["checkpoint_epoch"], 7)
            self.assertEqual(len(metadata["checkpoint_sha256"]), 64)
            self.assertLess(metadata["parameters"]["deployment_active_parameters"],
                            metadata["parameters"]["training_trainable_parameters"])

    def test_frozen_selected_checkpoint_keeps_training_parameter_scope(self):
        model = model_for("flat")
        model.configure_trainable(backbone_trainable=False)
        config = {**self.config, "code_hash": "current", "mode": "flat", "main_decoder": "flat",
                  "backbone_training": "frozen", "parameters_trainable_finetune":
                  sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save({"model": model.state_dict(), "config": config, "stage": "frozen_features",
                        "epoch": 7, "best_epoch": 7, "selected_val_macro_f1": .8}, path)
            with patch("cvm.profile.source_digest", return_value="current"), \
                    patch("cvm.profile.model_arguments", return_value={}), \
                    patch("cvm.profile.build_model", return_value=model_for("flat")):
                _, _, metadata = load_deployment_checkpoint(path)
            self.assertEqual(metadata["parameters"]["training_trainable_parameters"],
                             sum(parameter.numel() for parameter in model.flat_head.parameters()))


if __name__ == "__main__":
    unittest.main()
