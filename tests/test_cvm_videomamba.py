"""CPU contract tests; these do not exercise official VideoMamba/CUDA kernels."""

import argparse
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from cvm import videomamba as vm


class FakeEncoder(nn.Module):
    num_features = 192

    def __init__(self):
        super().__init__()
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, 16, 192))
        self.projection = nn.Linear(3, 192)
        self.head = nn.Linear(192, 400)

    def forward_features(self, video):
        return self.projection(video.mean(dim=(2, 3, 4)))


class VideoMambaContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "videomamba_t16_k400_f16_res224.pth"
        self.original = FakeEncoder()
        torch.save(self.original.state_dict(), self.checkpoint)

    def tearDown(self):
        self.temporary.cleanup()

    def test_loads_every_original_tensor_then_discards_classifier(self):
        core = FakeEncoder()
        report = vm._load_checkpoint(core, self.checkpoint)
        self.assertTrue(torch.equal(core.projection.weight, self.original.projection.weight))
        self.assertIsInstance(core.head, nn.Identity)
        self.assertEqual(report["loaded_tensor_count"], len(self.original.state_dict()))
        self.assertTrue(report["strict"])
        self.assertFalse(report["publisher_checksum_verified"])
        self.assertEqual(len(report["sha256"]), 64)

    def test_container_and_uniform_wrapper_prefix(self):
        torch.save({"model": {"module." + key: value for key, value in self.original.state_dict().items()},
                    "epoch": 3}, self.checkpoint)
        report = vm._load_checkpoint(FakeEncoder(), self.checkpoint)
        self.assertEqual(report["checkpoint_container"], "model")

    def test_official_training_namespace_metadata_has_explicit_version_contract(self):
        torch.save({"model": self.original.state_dict(), "args": argparse.Namespace(num_frames=16)},
                   self.checkpoint)
        if hasattr(torch.serialization, "safe_globals"):
            self.assertTrue(vm._load_checkpoint(FakeEncoder(), self.checkpoint)["strict"])
        else:
            with self.assertRaisesRegex(ValueError, "safe_globals"):
                vm._load_checkpoint(FakeEncoder(), self.checkpoint)

    def test_missing_backbone_key_never_becomes_random_initialization(self):
        state = self.original.state_dict()
        state.pop("projection.weight")
        torch.save(state, self.checkpoint)
        core = FakeEncoder()
        old_weights = core.projection.weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "missing"):
            vm._load_checkpoint(core, self.checkpoint)
        self.assertTrue(torch.equal(old_weights, core.projection.weight))

    def test_wrong_temporal_checkpoint_and_wrong_dataset_head_fail(self):
        for key, shape in (("temporal_pos_embedding", (1, 8, 192)),
                           ("head.weight", (174, 192))):
            with self.subTest(key=key):
                state = dict(self.original.state_dict())
                state[key] = torch.zeros(shape)
                torch.save(state, self.checkpoint)
                with self.assertRaisesRegex(ValueError, "not exact"):
                    vm._load_checkpoint(FakeEncoder(), self.checkpoint)

    def test_extra_keys_and_nonfinite_weights_fail(self):
        state = dict(self.original.state_dict())
        state["unknown.weight"] = torch.zeros(1)
        torch.save(state, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            vm._load_checkpoint(FakeEncoder(), self.checkpoint)
        state.pop("unknown.weight")
        state["projection.weight"] = torch.full_like(state["projection.weight"], float("nan"))
        torch.save(state, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            vm._load_checkpoint(FakeEncoder(), self.checkpoint)

    def test_missing_file_fails_before_importing_external_code(self):
        with patch.object(vm, "_import_official") as importer:
            with self.assertRaises(FileNotFoundError):
                vm.build_videomamba_tiny16(repo_path=self.root, checkpoint_path=self.root / "missing.pth")
        importer.assert_not_called()

    def test_wrong_commit_model_blob_and_dirty_sources_fail(self):
        source = self.root / vm.OFFICIAL_MODEL_PATH
        source.parent.mkdir(parents=True)
        source.write_text("# synthetic test source\n")
        for answers in (["wrong_commit"], [vm.OFFICIAL_COMMIT, "wrong_blob"],
                        [vm.OFFICIAL_COMMIT, vm.OFFICIAL_MODEL_BLOB, " M mamba/setup.py"]):
            with self.subTest(answers=answers), patch.object(vm, "_git", side_effect=answers):
                with self.assertRaises(ValueError):
                    vm._verify_checkout(self.root)

    def test_builder_uses_pinned_contract_and_exposes_load_report(self):
        def construct(**kwargs):
            self.assertEqual(kwargs, {"pretrained": False, "num_classes": 400,
                                      "num_frames": 16, "img_size": 224, "kernel_size": 1})
            return FakeEncoder()
        module = SimpleNamespace(videomamba_tiny=construct)
        with patch.object(vm, "_verify_checkout", return_value={"commit": vm.OFFICIAL_COMMIT}), \
             patch.object(vm, "_import_official", return_value=module), \
             patch.object(vm, "_verify_loaded_dependencies", return_value={"contract_test": True}):
            encoder = vm.build_videomamba_tiny16(repo_path=self.root,
                                               checkpoint_path=self.checkpoint, seed=42)
        self.assertEqual(encoder.feature_dim, 192)
        self.assertEqual(encoder.load_report["external_source"]["commit"], vm.OFFICIAL_COMMIT)
        self.assertEqual(encoder.load_report["frames"], 16)
        video = torch.ones(1, 3, 16, 224, 224)
        features = encoder(video)
        self.assertEqual(tuple(features.shape), (1, 192))
        features.sum().backward()
        self.assertIsNotNone(encoder.encoder.projection.weight.grad)
        with self.assertRaisesRegex(ValueError, "16,224,224"):
            encoder(video[:, :, :8])

    def test_incompatible_loaded_dependency_is_rejected(self):
        bad_source = self.root / "bad_mamba.py"
        bad_source.write_text("# does not match pinned implementation\n")
        module = SimpleNamespace(__file__=str(bad_source))
        with patch.dict(vm.sys.modules, {"mamba_ssm.modules.mamba_simple": module}):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                vm._verify_loaded_dependencies(self.root)

    def test_file_import_registers_module_for_timm_decorators_and_restores_path(self):
        source = self.root / vm.OFFICIAL_MODEL_PATH
        source.parent.mkdir(parents=True)
        source.write_text("import sys\nregistered = sys.modules[__name__]\n")
        old_path = list(vm.sys.path)
        name = "_fsn_cvm_official_videomamba_" + vm.OFFICIAL_COMMIT
        with patch.dict(vm.sys.modules):
            module = vm._import_official(self.root)
            self.assertIs(module.registered, module)
            self.assertIs(vm.sys.modules[name], module)
        self.assertEqual(vm.sys.path, old_path)


if __name__ == "__main__":
    unittest.main()
