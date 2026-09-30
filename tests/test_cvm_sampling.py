"""Tests for declared temporal and representation sensitivity treatments."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from cvm.models import ClinicalHierarchyModel, compute_loss
from cvm.train import (CachedProtocolDataset, Preprocessor, make_optimizer,
                       selected_frame_count, train_epoch)


class SamplingTests(unittest.TestCase):
    def test_32_frames_are_declared_only_for_convolutional_backbones(self):
        torch.set_num_threads(1)
        self.assertEqual(selected_frame_count("r2plus1d_18", 32), 32)
        with self.assertRaises(ValueError):
            selected_frame_count("videomae_base16", 32)
        with self.assertRaises(ValueError):
            selected_frame_count("mvit_v2_s", 32)
        with self.assertRaises(ValueError):
            selected_frame_count("r2plus1d_18", True)
        images = torch.zeros(1, 36, 3, 224, 224, dtype=torch.uint8)
        preprocess = Preprocessor("r2plus1d_18", "none", 32)
        self.assertEqual(preprocess.spec["frames"], 32)
        self.assertEqual(preprocess(images).shape, (1, 3, 32, 112, 112))
        self.assertEqual(Preprocessor("r2plus1d_18", "none").spec["frames"], 16)

    def test_repeat_fraction_uses_exact_selected_multiset_and_denominator(self):
        record = SimpleNamespace(clip_id="test-only")
        protocol = SimpleNamespace(records={"val": [record]}, cache_record=lambda value: value)
        with tempfile.TemporaryDirectory() as folder:
            array_path, metadata_path = Path(folder) / "cache.npy", Path(folder) / "meta.json"
            array = np.stack([np.full((2, 2, 3), index % 11, dtype=np.uint8) for index in range(36)])
            np.save(array_path, array)
            metadata_path.write_text(json.dumps({"pixel_repeat_fraction": 25 / 36}))
            for count in (16, 32):
                with patch("cvm.train.valid_cache", return_value=True), patch("cvm.train.cache_paths", return_value=(array_path, metadata_path)):
                    dataset = CachedProtocolDataset(protocol, "val", Path(folder), count)
                positions = torch.linspace(0, 35, count).round().long().tolist()
                unique = len({hashlib.sha256(array[index].tobytes()).digest() for index in positions})
                self.assertAlmostEqual(dataset.repeat_fractions[record.clip_id], 1 - unique / count)

    def test_frozen_feature_control_preserves_parameters_and_running_statistics(self):
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.BatchNorm1d(3)
                self.linear = nn.Linear(3, 4)

            def forward(self, video):
                return self.linear(self.norm(video.mean(dim=(2, 3, 4))))

        model = ClinicalHierarchyModel(Backbone(), 4, mode="flat")
        before = copy.deepcopy(model.backbone.state_dict())
        head_before = model.flat_head.weight.detach().clone()
        args = SimpleNamespace(backbone_training="frozen", optimizer="sgd", backbone_lr=.1,
                               head_lr=.1, warmup_lr=.1, weight_decay=0., momentum=0.)
        optimizer = make_optimizer(model, args, warmup=False)
        self.assertEqual([group["name"] for group in optimizer.param_groups], ["trained_heads"])
        batch = {"video": torch.randn(7, 3, 2, 2, 2), "label": torch.arange(7)}
        train_epoch(model, [batch], optimizer,
                    lambda output, target: compute_loss(output, target, model.mode, model.taxonomy),
                    lambda video, **kwargs: video, torch.device("cpu"), freeze_backbone=True)
        for key, value in model.backbone.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.assertFalse(torch.equal(model.flat_head.weight, head_before))


if __name__ == "__main__":
    unittest.main()
