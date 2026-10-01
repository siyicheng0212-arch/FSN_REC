"""Weighted accumulation, storage roles and no-test CLI checks."""
from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments.aligned_data import resolve_cache_record, ResolvedFullClipDataset, mapping_sha256
from experiments.full_data import cache_paths, request_digest
from experiments.pilot_data import ClipRecord, EXPECTED_LABELS
from experiments.run_aligned_context import object_sha256
from experiments.train_aligned_context import accumulation_loss, parse_args, video_to_device, make_datasets, _rng_state, _restore_rng


class AccumulationTest(unittest.TestCase):
    def test_weighted_and_unweighted_tail_match_full_window_gradients(self):
        for weights in (torch.tensor([1.,3.,.5]), torch.ones(3)):
            torch.manual_seed(3)
            model = nn.Linear(4,3).double()
            state = {k: v.clone() for k,v in model.state_dict().items()}
            x = torch.randn(7,4,dtype=torch.double)
            y = torch.tensor([0,1,2,1,1,0,2])
            weights = weights.double()
            full = F.cross_entropy(model(x),y,weight=weights) + .5*model(x).square().mean()
            full.backward()
            expected = [p.grad.clone() for p in model.parameters()]
            model.load_state_dict(state); model.zero_grad()
            mass = float(weights[y].sum())
            for indices in (slice(0,3),slice(3,6),slice(6,7)):
                a,b=x[indices],y[indices]
                logits=model(a)
                ce=F.cross_entropy(logits,b,weight=weights)
                penalty=.5*logits.square().mean()
                accumulation_loss(ce,penalty,len(b),len(y),float(weights[b].sum()),mass).backward()
            for wanted, actual in zip(expected,model.parameters()):
                torch.testing.assert_close(actual.grad,wanted,atol=1e-12,rtol=1e-12)

    def test_uint8_scales_only_once(self):
        raw=torch.tensor([0,255],dtype=torch.uint8)
        torch.testing.assert_close(video_to_device(raw,torch.device("cpu")),torch.tensor([0.,1.]))
        floats=torch.tensor([.2,.8])
        torch.testing.assert_close(video_to_device(floats,torch.device("cpu")),floats)

    def test_rng_replay_preserves_all_sampling_sources(self):
        before = _rng_state()
        expected = (random.random(), float(np.random.rand()), torch.rand(3))
        _restore_rng(before)
        actual = (random.random(), float(np.random.rand()), torch.rand(3))
        self.assertEqual(expected[:2], actual[:2])
        torch.testing.assert_close(expected[2], actual[2], atol=0, rtol=0)


class CacheRoleTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.row=ClipRecord("clip-a","val",0,EXPECTED_LABELS[0],"source","recording-a","/private/a.mp4",0.,1.,1.)

    def tearDown(self): self.temp.cleanup()

    def test_smoke_and_launcher_agree_on_unicode_cache_mapping_digest(self):
        mapping = [{"split": "val", "clip_id": "扫散_示例", "cache_split": "train"}]
        self.assertEqual(mapping_sha256(mapping), object_sha256(mapping))

    def cache(self,row, wrong_label=False):
        array,meta=cache_paths(self.root,row)
        array.parent.mkdir(parents=True,exist_ok=True)
        np.save(array,np.zeros((2,4,4,3),dtype=np.uint8))
        meta.write_text(json.dumps({"request_digest":request_digest(row,2,4),"clip_id":row.clip_id,
            "group_id":row.group_id,"label_id":1 if wrong_label else row.label_id}))

    def test_validation_role_can_read_existing_train_storage(self):
        self.cache(replace(self.row,split="train"))
        storage=resolve_cache_record(self.row,self.root,2,4)
        self.assertEqual(storage.split,"train")
        self.assertEqual(self.row.split,"val")
        manifest=self.root/"val.jsonl"
        manifest.write_text(json.dumps(self.row.to_manifest_dict())+"\n")
        dataset=ResolvedFullClipDataset(manifest,self.root,2,4)
        self.assertEqual(dataset.records[0].split,"val")
        self.assertEqual(dataset[0]["video"].dtype,torch.uint8)
        self.assertEqual(dataset.mapping[0]["cache_split"],"train")

    def test_corrupt_role_cache_cannot_hide_behind_fallback(self):
        self.cache(replace(self.row,split="train"))
        array,_=cache_paths(self.root,self.row); array.parent.mkdir(exist_ok=True)
        array.write_bytes(b"broken")
        with self.assertRaisesRegex(RuntimeError,"role cache"):
            resolve_cache_record(self.row,self.root,2,4)

    def test_wrong_label_is_rejected(self):
        self.cache(replace(self.row,split="train"),wrong_label=True)
        with self.assertRaisesRegex(RuntimeError,"label_id"):
            resolve_cache_record(self.row,self.root,2,4)

    def test_ambiguous_storage_requires_explicit_split(self):
        self.cache(self.row); self.cache(replace(self.row,split="train"))
        with self.assertRaisesRegex(RuntimeError,"ambiguous"):
            resolve_cache_record(self.row,self.root,2,4)
        self.assertEqual(resolve_cache_record(self.row,self.root,2,4,"train").split,"train")

    def test_test_cache_access_rejected(self):
        with self.assertRaises(ValueError):
            resolve_cache_record(replace(self.row,split="test"),self.root,2,4)
        with self.assertRaises(ValueError):
            resolve_cache_record(self.row,self.root,2,4,"test")

    def test_role_and_missing_group_fail_before_cache_loading(self):
        (self.root/"train.jsonl").write_text(json.dumps(self.row.to_manifest_dict())+"\n")
        args=type("Args",(),{"manifest_dir":self.root,"cache_dir":self.root})()
        with self.assertRaisesRegex(RuntimeError,"role mismatch"):
            make_datasets(args)


class CliTest(unittest.TestCase):
    def test_smoke_does_not_require_variant_or_output(self):
        args=parse_args(["--manifest-dir","a","--cache-dir","b","--checkpoint","c",
                         "--smoke-only","--smoke-report","d"])
        self.assertTrue(args.smoke_only)

    def test_no_random_initialization_or_test_flags(self):
        for flag in ("--allow-random-init","--test-after-training"):
            with self.assertRaises(SystemExit):
                parse_args(["--manifest-dir","a","--cache-dir","b","--checkpoint","c",flag])


if __name__=="__main__": unittest.main()
