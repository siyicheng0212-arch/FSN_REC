"""Meaningful CPU checks without a pretrained download or CUDA dependency."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn

from cvm.models import (ClinicalHierarchyModel, build_model, compute_loss,
                        decode_flat, decode_hard, decode_oracle, decode_soft,
                        flat_oracle, loss_components, parameter_report,
                        preprocessing_spec, preprocessing_transform,
                        soft_leaf_probabilities)
from cvm.taxonomy import CLASS_NAMES, Taxonomy, clinical_taxonomy, get_taxonomy, matched_random_taxonomies


class ToyBackbone(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.linear = nn.Linear(3, dim)

    def forward(self, video):
        return self.linear(video.mean(dim=(2, 3, 4)))


def toy(mode="flat", taxonomy=None, seed=42):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ClinicalHierarchyModel(ToyBackbone(), 8, mode=mode, taxonomy=taxonomy)


class TaxonomyTests(unittest.TestCase):
    def test_canonical_partition_and_target_order(self):
        taxonomy = clinical_taxonomy()
        self.assertEqual(CLASS_NAMES, ("消毒", "进针", "运针", "扫散", "再灌注", "拔针", "固定"))
        self.assertEqual(taxonomy.groups, ((0,), (6,), (1, 2, 5), (3, 4)))
        labels = torch.arange(7)
        self.assertEqual(taxonomy.group_targets(labels).tolist(), [0, 2, 2, 3, 3, 2, 1])
        self.assertEqual(taxonomy.conditional_targets(labels).tolist(), [0, 0, 1, 0, 1, 2, 0])
        self.assertEqual(Taxonomy.from_dict(taxonomy.to_dict()), taxonomy)

    def test_invalid_partition_rejected(self):
        for groups in (((0,), (0, 1, 2, 3, 4, 5, 6)), ((0,), (1, 2, 3, 4, 5)), ((0,), (), (1, 2, 3, 4, 5, 6))):
            with self.assertRaises(ValueError):
                Taxonomy("invalid", groups)
        with self.assertRaises(ValueError):
            clinical_taxonomy().group_targets(torch.tensor([7]))

    def test_random_controls_preserve_singletons_and_are_fixed_distinct(self):
        controls = matched_random_taxonomies()
        self.assertEqual(controls, matched_random_taxonomies())
        signatures = {tuple(sorted(tuple(sorted(group)) for group in taxonomy.groups))
                      for taxonomy in (clinical_taxonomy(),) + controls}
        self.assertEqual(len(signatures), 4)
        for taxonomy in controls:
            self.assertEqual(taxonomy.groups[:2], ((0,), (6,)))
            self.assertEqual(taxonomy.group_sizes, (1, 1, 3, 2))
            self.assertEqual(taxonomy.auxiliary_rows, 9)
            self.assertEqual(get_taxonomy(taxonomy.name), taxonomy)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.video = torch.randn(7, 3, 2, 4, 4)
        self.targets = torch.arange(7)
        self.taxonomy = clinical_taxonomy()

    def test_same_seed_preserves_backbone_and_flat_initialization(self):
        baseline = toy("flat")
        for mode in ("aux_flat", "hierarchy", "capacity_control"):
            candidate = toy(mode)
            self.assertTrue(torch.equal(baseline(self.video)["features"], candidate(self.video)["features"]))
            self.assertTrue(torch.equal(baseline(self.video)["flat_logits"], candidate(self.video)["flat_logits"]))
        random_model = toy("aux_flat", matched_random_taxonomies()[0])
        self.assertTrue(torch.equal(baseline(self.video)["flat_logits"], random_model(self.video)["flat_logits"]))

    def test_head_freezes_and_backbone_warmup_restoration(self):
        for mode, active in (("flat", ("flat",)), ("capacity_control", ("flat", "capacity")),
                             ("aux_flat", ("flat", "group", "conditional")), ("hierarchy", ("group", "conditional"))):
            model = toy(mode)
            self.assertEqual(model.trained_heads, active)
            model.configure_trainable(backbone_trainable=False)
            self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.parameters()))
            model.configure_trainable()
            self.assertTrue(all(parameter.requires_grad for parameter in model.backbone.parameters()))
            for name, module in (("flat", model.flat_head), ("group", model.group_head), ("conditional", model.conditional_heads)):
                self.assertTrue(all(parameter.requires_grad == (name in active) for parameter in module.parameters()))
            outputs = model(self.video)
            loss = compute_loss(outputs, self.targets, mode, self.taxonomy)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(model.backbone.linear.weight.grad.abs().sum()), 0)
            self.assertEqual(sum(parameter.numel() for parameter in model.conditional_heads[0].parameters()), 0)
            self.assertEqual(sum(parameter.numel() for parameter in model.conditional_heads[1].parameters()), 0)
            if mode in ("flat", "capacity_control"):
                with self.assertRaises(ValueError):
                    decode_hard(outputs, self.taxonomy)
            if mode == "hierarchy":
                with self.assertRaises(ValueError):
                    decode_flat(outputs)

    def test_exact_active_capacity_match(self):
        control, auxiliary = toy("capacity_control"), toy("aux_flat")
        control_report, aux_report = parameter_report(control), parameter_report(auxiliary)
        self.assertEqual(control_report["trainable"], aux_report["trainable"])
        self.assertEqual(control_report["components"]["capacity"]["trainable"], 9 * (8 + 1))
        outputs = control(self.video)
        compute_loss(outputs, self.targets, "capacity_control", self.taxonomy).backward()
        for name, parameter in control.capacity_head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)

    def test_soft_probabilities_and_loss_factorization(self):
        outputs = toy("hierarchy")(self.video)
        probabilities = soft_leaf_probabilities(outputs, self.taxonomy)
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(7), atol=1e-6))
        selected_nll = -probabilities[torch.arange(7), self.targets].log()
        losses = loss_components(outputs, self.targets, self.taxonomy)
        self.assertTrue(torch.allclose(losses["hierarchy"], selected_nll.mean(), atol=1e-6))
        weights = torch.tensor([1., 4., 2., 3., 6., 2., 8.])
        weighted = loss_components(outputs, self.targets, self.taxonomy, weights)
        self.assertTrue(torch.allclose(weighted["hierarchy"], (selected_nll * weights).sum() / weights.sum(), atol=1e-6))
        # Singleton-only minibatch has exactly zero fine loss; no invented head.
        singleton_outputs = {key: value[[0, 6]] if isinstance(value, torch.Tensor) else
                             [part[[0, 6]] for part in value] if key == "conditional_logits" else value
                             for key, value in outputs.items()}
        singleton_loss = loss_components(singleton_outputs, torch.tensor([0, 6]), self.taxonomy)
        self.assertEqual(float(singleton_loss["conditional"]), 0.)

    def test_probabilities_are_float32_under_bfloat16_outputs(self):
        outputs = toy("hierarchy")(self.video)
        outputs["group_logits"] = outputs["group_logits"].to(torch.bfloat16)
        outputs["conditional_logits"] = [part.to(torch.bfloat16) for part in outputs["conditional_logits"]]
        probabilities = soft_leaf_probabilities(outputs, self.taxonomy)
        self.assertEqual(probabilities.dtype, torch.float32)
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(7), atol=1e-6))

    def test_oracle_and_hard_soft_counterexample(self):
        outputs = {"group_logits": torch.tensor([[1., -8., .9, .8]]),
                   "conditional_logits": [torch.zeros(1, 1), torch.zeros(1, 1),
                                          torch.tensor([[9., -9., -9.]]), torch.zeros(1, 2)],
                   "flat_logits": torch.tensor([[10., 1., 2., 3., 4., 5., 6.]]),
                   "trained_heads": ("flat", "group", "conditional")}
        self.assertEqual(decode_hard(outputs, self.taxonomy).item(), 0)
        self.assertEqual(decode_soft(outputs, self.taxonomy).item(), 0)
        # Make group2 the largest but its mass diffuse; singleton wins soft.
        outputs["group_logits"] = torch.tensor([[1., -8., 1.1, .8]])
        outputs["conditional_logits"][2] = torch.zeros(1, 3)
        self.assertEqual(decode_hard(outputs, self.taxonomy).item(), 1)
        self.assertEqual(decode_soft(outputs, self.taxonomy).item(), 0)
        target = torch.tensor([5])
        self.assertEqual(decode_oracle(outputs, target, self.taxonomy).item(), 1)
        self.assertEqual(flat_oracle(outputs, target, self.taxonomy).item(), 5)
        # Both diagnostic oracles remain inside true group, not equal to target.
        self.assertIn(decode_oracle(outputs, target, self.taxonomy).item(), (1, 2, 5))

    def test_no_metadata_or_labels_in_forward_interface(self):
        with self.assertRaises(TypeError):
            toy()(self.video, labels=self.targets)
        with self.assertRaises(ValueError):
            toy()(self.video[:, 0])

    def test_native_preprocessing_shapes_without_weights_download(self):
        clip = torch.rand(2, 3, 224, 224)
        self.assertEqual(preprocessing_transform("r2plus1d_18")(clip).shape, (3, 2, 112, 112))
        self.assertEqual(preprocessing_transform("mvit_v2_s")(clip).shape, (3, 2, 224, 224))
        self.assertEqual(preprocessing_spec("mvit_v2_s")["frames"], 16)
        self.assertEqual(preprocessing_spec("r2plus1d_18")["frames"], 16)
        self.assertEqual(preprocessing_spec("videomae_base16")["frames"], 16)
        self.assertEqual(preprocessing_spec("videomae_base16")["resize_size"], [224])
        self.assertEqual(preprocessing_spec("videomae_base16")["mean"], [.485, .456, .406])
        self.assertEqual(preprocessing_transform("videomae_base16")(clip).shape, (3, 2, 224, 224))

    def test_external_backbones_require_explicit_source_and_checkpoint(self):
        for backbone in ("videomamba_tiny16", "videomae_base16"):
            with self.assertRaises(ValueError):
                build_model(backbone, weights=None)
            with self.assertRaises(ValueError):
                build_model(backbone, external_repo="/unused/source", weights=None)

    def test_missing_pretrained_is_an_error_not_random_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("torch.hub.get_dir", return_value=directory):
                with self.assertRaises(FileNotFoundError):
                    build_model("r3d_18", weights="DEFAULT", allow_download=False)

    def test_corrupt_native_checkpoint_is_strictly_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pth"
            torch.save({"state_dict": {"unrelated.weight": torch.zeros(1)}}, path)
            with self.assertRaises((RuntimeError, ValueError)):
                build_model("r3d_18", weights_path=path)


if __name__ == "__main__":
    unittest.main()
