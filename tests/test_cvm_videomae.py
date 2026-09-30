"""CPU adapter contract tests; no official pretrained binary or CUDA is used."""

import argparse
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from cvm import videomae as mae


class FakeMeanPoolEncoder(nn.Module):
    """Small fixture preserves the official output/head and position contracts."""

    num_features = 768

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 768)
        self.fc_norm = nn.LayerNorm(768)
        self.head = nn.Linear(768, 400)
        # Official fixed sinusoidal positions are deliberately not a buffer.
        self.pos_embed = torch.zeros(1, 1568, 768)

    def forward_features(self, video):
        return self.fc_norm(self.projection(video.mean(dim=(2, 3, 4))))


class VideoMAEContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "videomae_vitb_k400_800e_finetuned.pth"
        self.original = FakeMeanPoolEncoder()
        torch.save(self.original.state_dict(), self.checkpoint)

    def tearDown(self):
        self.temporary.cleanup()

    def test_loads_original_normalization_and_head_before_removing_head(self):
        core = FakeMeanPoolEncoder()
        report = mae._load_checkpoint(core, self.checkpoint)
        for key in ("projection.weight", "fc_norm.weight", "fc_norm.bias"):
            self.assertTrue(torch.equal(core.state_dict()[key], self.original.state_dict()[key]))
        self.assertIsInstance(core.head, nn.Identity)
        self.assertEqual(report["loaded_tensor_count"], len(self.original.state_dict()))
        self.assertEqual(report["discarded_pretrained_classifier"], ["head.weight", "head.bias"])
        self.assertTrue(report["strict"])
        self.assertFalse(report["publisher_checksum_verified"])
        self.assertEqual(len(report["sha256"]), 64)

    def test_absent_fixed_position_key_does_not_certify_training_input(self):
        self.assertNotIn("pos_embed", self.original.state_dict())
        report = mae._load_checkpoint(FakeMeanPoolEncoder(), self.checkpoint)
        self.assertFalse(report["checkpoint_input_configuration_verified_from_metadata"])
        self.assertEqual(report["checked_training_args"], {})
        self.assertIn("do not prove", report["checkpoint_input_configuration_note"])

    def test_supported_container_uniform_prefix_and_verified_metadata(self):
        for container in ("model", "module", "state_dict"):
            with self.subTest(container=container):
                torch.save({container: {"module.encoder." + k: v for k, v in self.original.state_dict().items()},
                            "args": {"model": "vit_base_patch16_224", "num_frames": 16,
                                     "num_segments": 1, "tubelet_size": 2, "input_size": 224,
                                     "use_mean_pooling": True, "nb_classes": 400,
                                     "data_set": "Kinetics-400"}}, self.checkpoint)
                report = mae._load_checkpoint(FakeMeanPoolEncoder(), self.checkpoint)
                self.assertEqual(report["checkpoint_container"], container)
                self.assertTrue(report["checkpoint_input_configuration_verified_from_metadata"])

    def test_conflicting_training_metadata_rejected_without_parameter_mutation(self):
        for key, value in (("num_frames", 8), ("num_segments", True), ("tubelet_size", 1),
                           ("input_size", 320), ("use_mean_pooling", False),
                           ("data_set", "SSV2"), ("nb_classes", 710)):
            with self.subTest(key=key):
                torch.save({"model": self.original.state_dict(), "args": {key: value}}, self.checkpoint)
                core = FakeMeanPoolEncoder()
                before = core.projection.weight.detach().clone()
                with self.assertRaisesRegex(ValueError, "metadata"):
                    mae._load_checkpoint(core, self.checkpoint)
                self.assertTrue(torch.equal(before, core.projection.weight))
                self.assertIsInstance(core.head, nn.Linear)

    def test_original_namespace_metadata_has_explicit_safe_loading_contract(self):
        torch.save({"model": self.original.state_dict(), "args": argparse.Namespace(num_frames=16)},
                   self.checkpoint)
        if hasattr(torch.serialization, "safe_globals"):
            self.assertTrue(mae._load_checkpoint(FakeMeanPoolEncoder(), self.checkpoint)["strict"])
        else:
            with self.assertRaisesRegex(ValueError, "safe_globals"):
                mae._load_checkpoint(FakeMeanPoolEncoder(), self.checkpoint)

    def test_missing_encoder_or_normalization_tensor_never_falls_back_to_random(self):
        for key in ("projection.weight", "fc_norm.weight", "head.bias"):
            with self.subTest(key=key):
                state = dict(self.original.state_dict())
                state.pop(key)
                torch.save(state, self.checkpoint)
                core = FakeMeanPoolEncoder()
                before = core.projection.weight.detach().clone()
                with self.assertRaisesRegex(ValueError, "missing"):
                    mae._load_checkpoint(core, self.checkpoint)
                self.assertTrue(torch.equal(before, core.projection.weight))

    def test_wrong_dataset_head_or_pretraining_decoder_is_rejected(self):
        for shape in ((174, 768), (710, 768), (400, 384)):
            with self.subTest(shape=shape):
                state = dict(self.original.state_dict())
                state["head.weight"] = torch.zeros(shape)
                torch.save(state, self.checkpoint)
                with self.assertRaisesRegex(ValueError, "not exact"):
                    mae._load_checkpoint(FakeMeanPoolEncoder(), self.checkpoint)
        state = dict(self.original.state_dict())
        state["decoder.head.weight"] = torch.zeros(3, 3)
        torch.save(state, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            mae._load_checkpoint(FakeMeanPoolEncoder(), self.checkpoint)

    def test_nonfinite_weights_and_ambiguous_containers_are_rejected(self):
        state = dict(self.original.state_dict())
        state["projection.weight"] = torch.full_like(state["projection.weight"], float("nan"))
        torch.save(state, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            mae._load_checkpoint(FakeMeanPoolEncoder(), self.checkpoint)
        torch.save({"model": self.original.state_dict(), "module": self.original.state_dict()}, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            mae._load_checkpoint(FakeMeanPoolEncoder(), self.checkpoint)

    def test_missing_checkpoint_fails_before_importing_external_code(self):
        with patch.object(mae, "_import_official") as importer:
            with self.assertRaises(FileNotFoundError):
                mae.build_videomae_base16(repo_path=self.root, checkpoint_path=self.root / "missing.pth")
        importer.assert_not_called()

    def test_wrong_source_commit_blob_or_dirty_file_is_rejected(self):
        (self.root / mae.OFFICIAL_MODEL_PATH).write_text("# synthetic source fixture\n")
        for answers in (["wrong_commit"], [mae.OFFICIAL_COMMIT, "wrong_blob"],
                        [mae.OFFICIAL_COMMIT, mae.OFFICIAL_MODEL_BLOB, " M modeling_finetune.py"]):
            with self.subTest(answers=answers), patch.object(mae, "_git", side_effect=answers):
                with self.assertRaises(ValueError):
                    mae._verify_checkout(self.root)

    def test_builder_uses_native_all_frames_mean_pooling_and_preserves_caller_rng(self):
        def construct(**kwargs):
            self.assertEqual(kwargs, {"pretrained": False, "num_classes": 400, "all_frames": 16,
                                      "img_size": 224, "tubelet_size": 2, "use_mean_pooling": True,
                                      "use_learnable_pos_emb": False, "use_checkpoint": False,
                                      "drop_path_rate": .1, "init_scale": .001})
            return FakeMeanPoolEncoder()
        module = SimpleNamespace(vit_base_patch16_224=construct)
        before = torch.random.get_rng_state().clone()
        with patch.object(mae, "_verify_checkout", return_value={"commit": mae.OFFICIAL_COMMIT}), \
             patch.object(mae, "_import_official", return_value=module), \
             patch.object(mae, "_dependency_report", return_value={"contract_test": True}), \
             patch.object(torch.cuda, "manual_seed_all") as cuda_seed:
            encoder = mae.build_videomae_base16(repo_path=self.root, checkpoint_path=self.checkpoint, seed=42)
        cuda_seed.assert_not_called()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(encoder.feature_dim, 768)
        self.assertEqual(encoder.load_report["external_source"]["commit"], mae.OFFICIAL_COMMIT)
        video = torch.ones(1, 3, 16, 224, 224)
        features = encoder(video)
        self.assertEqual(tuple(features.shape), (1, 768))
        features.square().sum().backward()
        self.assertIsNotNone(encoder.encoder.projection.weight.grad)
        self.assertIsNotNone(encoder.encoder.fc_norm.weight.grad)
        with self.assertRaisesRegex(ValueError, "16,224,224"):
            encoder(video[:, :, :8])

    def test_incompatible_dependency_version_rejected(self):
        with patch.object(mae.metadata, "version", return_value="1.0.0"):
            with self.assertRaisesRegex(RuntimeError, "Unsupported"):
                mae._dependency_report()

    def test_import_registers_module_for_timm_and_restores_previous_on_error(self):
        source = self.root / mae.OFFICIAL_MODEL_PATH
        source.write_text("import sys\nregistered = sys.modules[__name__]\n")
        name = "_fsn_cvm_official_videomae_" + mae.OFFICIAL_COMMIT
        old_path = list(mae.sys.path)
        with patch.dict(mae.sys.modules):
            module = mae._import_official(self.root)
            self.assertIs(module.registered, module)
            self.assertIs(mae.sys.modules[name], module)
            source.write_text("raise ImportError('synthetic missing dependency')\n")
            with self.assertRaisesRegex(RuntimeError, "dependencies"):
                mae._import_official(self.root)
            self.assertIs(mae.sys.modules[name], module)
        self.assertEqual(mae.sys.path, old_path)


if __name__ == "__main__":
    unittest.main()
