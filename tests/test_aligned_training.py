"""Weighted accumulation, storage roles and no-test CLI checks."""
from dataclasses import replace
import json
import hashlib
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments.aligned_data import resolve_cache_record, ResolvedFullClipDataset, mapping_sha256, load_aligned_manifest, cache_split_hint
from experiments.full_data import cache_paths, request_digest
from experiments.pilot_data import ClipRecord, EXPECTED_LABELS
from experiments.run_aligned_context import object_sha256, audit_manifests, audit_cache
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

    def cache(self,row, wrong_label=False, frames=2, size=4):
        array,meta=cache_paths(self.root,row)
        array.parent.mkdir(parents=True,exist_ok=True)
        np.save(array,np.full((frames,size,size,3),row.label_id,dtype=np.uint8))
        meta.write_text(json.dumps({"request_digest":request_digest(row,frames,size),"clip_id":row.clip_id,
            "group_id":row.group_id,"label_id":1 if wrong_label else row.label_id}))

    def test_internal_validation_role_reads_original_train_cache_without_rewriting_manifest(self):
        original=replace(self.row,split="train")
        self.cache(original)
        raw={**original.to_manifest_dict(),"inner_split_role":"val"}
        path=self.root/"val.jsonl"
        path.write_text(json.dumps(raw)+"\n")
        before=path.read_bytes()
        records,rows=load_aligned_manifest(path,"val")
        self.assertEqual(records[0].split,"val")
        self.assertEqual(rows[0]["split"],"train")
        dataset=ResolvedFullClipDataset(path,self.root,2,4)
        self.assertEqual(dataset.records[0].split,"val")
        self.assertEqual(dataset.storage_records[0].split,"train")
        self.assertEqual(tuple(dataset[0]["video"].shape),(2,3,4,4))
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),hashlib.sha256(before).hexdigest())

    def test_invalid_internal_roles_and_source_test_cannot_be_reinterpreted(self):
        path=self.root/"val.jsonl"
        for invalid in (None,"","test","train",["val"]):
            with self.subTest(role=invalid):
                row={**replace(self.row,split="train").to_manifest_dict(),"inner_split_role":invalid}
                path.write_text(json.dumps(row)+"\n")
                with self.assertRaises(RuntimeError):
                    load_aligned_manifest(path,"val")
        path.write_text(json.dumps({**replace(self.row,split="test").to_manifest_dict(),"inner_split_role":"val"})+"\n")
        with self.assertRaisesRegex(RuntimeError,"source split"):
            load_aligned_manifest(path,"val")

    def test_internal_storage_hint_respects_explicit_override_and_rejects_unsafe_cache(self):
        row={**replace(self.row,split="train").to_manifest_dict(),"inner_split_role":"val"}
        self.assertEqual(cache_split_hint(row),"train")
        self.assertEqual(cache_split_hint({**row,"cache_split":"val"}),"val")
        # Optional null has the original loader's no-override semantics.
        self.assertEqual(cache_split_hint({**row,"cache_split":None}),"train")
        for invalid in ("test","",["train"]):
            with self.assertRaises(RuntimeError):
                cache_split_hint({**row,"cache_split":invalid})

    def test_internal_role_trainer_and_launcher_use_same_mapping_and_input_sha(self):
        directory=self.root/"manifests"; directory.mkdir()
        hashes={}; original_bytes={}
        for role in ("train","val"):
            rows=[]
            for label,name in EXPECTED_LABELS.items():
                original=replace(self.row,clip_id=f"clip-{role}-{label}",group_id=f"recording-{role}-{label}",
                                 split="train",label_id=label,normalized_label=name)
                self.cache(original,frames=36,size=224)
                rows.append({**original.to_manifest_dict(),"inner_split_role":role})
            path=directory/f"{role}.jsonl"
            path.write_text("".join(json.dumps(row)+"\n" for row in rows))
            original_bytes[role]=path.read_bytes()
            hashes[role]=hashlib.sha256(original_bytes[role]).hexdigest()
        args=type("Args",(),{"manifest_dir":directory,"cache_dir":self.root})()
        datasets,training_audit=make_datasets(args)
        records,_=audit_manifests(directory,hashes,{"train":7,"val":7})
        launcher_audit=audit_cache(records,self.root,directory)
        self.assertEqual(training_audit["manifest_sha256"],hashes)
        self.assertEqual(training_audit["cache_mapping_sha256"],launcher_audit["cache_mapping_sha256"])
        self.assertEqual(training_audit["counts"],{"train":7,"val":7})
        self.assertTrue(all(row.split=="val" for row in datasets["val"].records))
        self.assertTrue(all(row.split=="train" for row in datasets["val"].storage_records))
        self.assertEqual(int(datasets["val"][2]["video"][0,0,0,0]),2)
        for role in ("train","val"):
            self.assertEqual((directory/f"{role}.jsonl").read_bytes(),original_bytes[role])

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
