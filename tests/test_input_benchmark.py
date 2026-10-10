"""Synthetic CPU contracts, not real-feature CUDA or completed server experiments."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.anchor_reference.data import anchor_batch, train_replacements
from experiments.input_evidence.model import InputEvidenceTCN
from experiments.input_benchmark import model, metrics, run_suite, worker, report
from experiments.relation.data import sha256_file
from experiments.relation.tcn_baseline import TCNConfig, collate_segments
from experiments.paper_validation import perturb
from tests.test_anchor_reference import fixtures, inputs


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make(self, task="logits", config=None):
        return model.BenchmarkHead(config or TCNConfig(5, 7, 8, 3, 0), task)

    def test_registry_finite_and_copied(self):
        self.assertEqual(len(model.TASKS), 12)
        spec = model.specification("logits")
        spec["head"] = "other"
        self.assertEqual(model.TASKS["logits"]["head"], "native_tcn")
        with self.assertRaises(ValueError):
            self.make("new_unregistered_method")

    def test_all_models_support_padding_singletons_and_four_stage_layout(self):
        f, a, valid = inputs((7, 3, 1))
        with torch.no_grad():
            for task in model.TASKS:
                with self.subTest(task=task):
                    m = self.make(task).eval()
                    z = m(f, a, valid, all_stages=True)
                    count = 1 if m.spec["head"] in ("native_tcn", "center_only") else 4
                    self.assertEqual(z.shape, (count, 3, 7, 7))
                    self.assertTrue(torch.isfinite(z).all())
                    for i, length in enumerate((7, 3, 1)):
                        solo = m(f[i:i+1, :length], a[i:i+1, :length], valid[i:i+1, :length])
                        torch.testing.assert_close(z[-1, i, :length], solo[0], atol=2e-6, rtol=2e-6)

    def test_masks_have_no_score_path_but_raw_controls_do(self):
        f, a, valid = inputs((5, 1))
        changed = f.clone(); changed[..., -7:] += 3
        with torch.no_grad():
            for task in model.TASKS:
                with self.subTest(task=task):
                    m = self.make(task).eval()
                    first, second = m(f, a, valid), m(changed, changed[..., -7:], valid)
                    if m.spec["scores"] == "zero":
                        torch.testing.assert_close(first, second, atol=0, rtol=0)
                    else:
                        self.assertGreater(float((first-second).abs().max()), 0)

    def test_native_cells_match_existing_input_diagnostic_exactly(self):
        f, a, valid = inputs()
        for task in ("logits", "logits_aug", "visual", "visual_aug"):
            old = InputEvidenceTCN(TCNConfig(5, 7, 8, 3, 0), task).eval()
            actual = self.make(task).eval()
            torch.testing.assert_close(actual(f, a, valid), old(f, a, valid), atol=0, rtol=0)

    def test_center_only_is_native_single_clip_not_A_fallback(self):
        f, a, valid = inputs((7, 3))
        for task in ("clip_logits", "clip_visual"):
            m = self.make(task).eval()
            z = m(f, a, valid)
            for b, t in zip(*torch.where(valid)):
                single = m(f[b:b+1, t:t+1], a[b:b+1, t:t+1], torch.ones(1,1,dtype=torch.bool))
                torch.testing.assert_close(z[b, t], single[0,0], atol=2e-6, rtol=2e-6)
            self.assertFalse(torch.equal(z[valid], a[valid]))

    def test_center_only_neighbors_do_not_affect_current_clip(self):
        f, a, valid = inputs((7,))
        changed = f.clone(); changed[:, :3] *= 20; changed[:, 4:] *= 20
        for task in ("clip_logits", "clip_visual"):
            m = self.make(task).eval()
            torch.testing.assert_close(m(f,a,valid)[:,3], m(changed,changed[...,-7:],valid)[:,3], atol=0,rtol=0)

    def test_same_family_initialization_and_no_global_rng_consumption(self):
        families = [("logits", "visual", "logits_aug", "visual_aug", "clip_logits", "clip_visual"),
                    ("mstcnpp_logits", "mstcnpp_visual", "mstcnpp_logits_final_ce"),
                    ("asformer_logits", "asformer_visual", "asformer_logits_final_ce")]
        for tasks in families:
            first = self.make(tasks[0]).state_dict()
            for task in tasks:
                torch.manual_seed(11); before = torch.get_rng_state().clone()
                actual = self.make(task).state_dict()
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assertEqual(set(first), set(actual))
                self.assertTrue(all(torch.equal(first[n], actual[n]) for n in first))

    def test_mask_is_not_in_place_and_pairing_is_validated_first(self):
        f,a,valid=inputs(); before=f.clone()
        self.make("visual")(f,a,valid)
        self.assertTrue(torch.equal(f,before))
        for task in ("visual", "mstcnpp_visual", "asformer_visual"):
            with self.assertRaises(ValueError):
                self.make(task)(f,a+1,valid)
            bad=f.clone();bad[0,0,-1]=float("nan")
            with self.assertRaises(ValueError):
                self.make(task)(bad,bad[...,-7:],valid)

    def test_invalid_or_empty_mask_and_feature_dimension_rejected(self):
        f,a,v=inputs()
        for invalid in (torch.zeros_like(v), v.float(), v.flip(1)):
            with self.assertRaises(ValueError):
                self.make()(f,a,invalid)
        with self.assertRaises(ValueError):
            self.make()(f[...,:-1],a,v)

    def test_real_backbone_vector_dimensions_supported_without_A_model(self):
        for gd,ld in ((768,768),(1280,2048)):
            f=torch.randn(1,2,gd+ld+7);a=f[...,-7:];v=torch.ones(1,2,dtype=torch.bool)
            for task in ("logits","mstcnpp_logits","asformer_logits"):
                z=self.make(task,TCNConfig(gd,ld,64,3,.1)).eval()(f,a,v)
                self.assertEqual(z.shape,(1,2,7))

    def test_all_cells_three_CPU_updates_with_finite_gradients_and_real_changes(self):
        f,a,v=inputs((7,7));y=torch.arange(14).reshape(2,7)%7
        for task in model.TASKS:
            with self.subTest(task=task):
                m=self.make(task);opt=torch.optim.AdamW(m.parameters(),lr=.001)
                before={n:p.detach().clone() for n,p in m.named_parameters()}
                for step in range(3):
                    torch.manual_seed(42+step);opt.zero_grad()
                    loss,_,_=model.objective(m(f,a,v,all_stages=True),y,v,m.spec)
                    self.assertTrue(torch.isfinite(loss));loss.backward()
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
                    projection = m.core.base.stem[0] if m.spec["head"] in ("native_tcn","center_only") else \
                        m.core.PG.conv_1x1_in if m.spec["head"]=="mstcnpp" else m.core.encoder.conv_1x1
                    self.assertGreater(float(projection.weight.grad.abs().sum()),0)
                    if m.spec["scores"]=="zero":
                        self.assertEqual(float(projection.weight.grad[:,-7:].abs().sum()),0)
                    if m.spec["head"]=="center_only":
                        for block in m.core.base.blocks:
                            self.assertEqual(float(block.conv.weight.grad[:,:,[0,2]].abs().sum()),0)
                    opt.step()
                self.assertTrue(any(not torch.equal(p,before[n]) for n,p in m.named_parameters()))
                self.assertIsNone(f.grad);self.assertIsNone(a.grad)

    def test_counts_report_unused_center_parameters_not_equal_effective_capacity(self):
        plain=self.make().parameter_counts();center=self.make("clip_visual").parameter_counts()
        self.assertEqual(plain["parameters"],center["parameters"])
        self.assertGreater(center["unused_native_weight_elements"],plain["unused_native_weight_elements"])
        self.assertEqual(center["A_parameters_trained"],0)

    def test_missing_shared_checkpoint_weight_fails_strict_load(self):
        m=self.make();state=dict(m.core.state_dict());state.pop("base.classifier.bias")
        with self.assertRaises(RuntimeError):m.core.load_state_dict(state,strict=True)


class LossTests(unittest.TestCase):
    def test_final_ce_only_uses_last_stage(self):
        z=torch.randn(4,2,3,7,requires_grad=True);y=torch.arange(6).reshape(2,3);v=torch.ones(2,3,dtype=torch.bool)
        loss,ce,smooth=model.objective(z,y,v,{"loss":"final_ce"})
        torch.testing.assert_close(loss,torch.nn.functional.cross_entropy(z[-1][v],y[v]))
        self.assertEqual(float(smooth),0);loss.backward()
        self.assertEqual(float(z.grad[:-1].abs().sum()),0)

    def test_stage_loss_matches_explicit_valid_adjacent_formula(self):
        z=torch.randn(4,2,3,7);y=torch.arange(6).reshape(2,3);v=torch.tensor([[1,1,1],[1,0,0]],dtype=torch.bool)
        loss,ce,smooth=model.objective(z,y,v,{"loss":"stage_ce_smooth"})
        expected=sum(torch.nn.functional.cross_entropy(s[v],y[v]) for s in z)
        pairs=v[:,1:]&v[:,:-1]
        smoothing=sum(((s.log_softmax(-1)[:,1:]-s.log_softmax(-1)[:,:-1].detach()).square().clamp(0,16))[pairs].mean() for s in z)
        torch.testing.assert_close(ce,expected);torch.testing.assert_close(smooth,smoothing)
        torch.testing.assert_close(loss,expected+.15*smoothing)

    def test_anchor_loss_has_no_smoothing_or_donor_targets(self):
        z=torch.randn(4,2,3,7,requires_grad=True);y=torch.full((2,3),-100);y[0,1]=2;y[1,2]=3
        v=torch.ones(2,3,dtype=torch.bool)
        loss,_,smooth=model.objective(z,y,v,{"loss":"stage_ce_smooth"},[1,2]);loss.backward()
        self.assertEqual(float(smooth),0)
        self.assertEqual(float(z.grad[:,0,[0,2]].abs().sum()),0)
        bad=y.clone();bad[0,0]=4
        with self.assertRaises(ValueError):model.objective(z,bad,v,{"loss":"stage_ce_smooth"},[1,2])

    def test_singletons_have_zero_smoothing(self):
        z=torch.randn(4,2,1,7);y=torch.tensor([[1],[2]]);v=torch.ones(2,1,dtype=torch.bool)
        self.assertEqual(float(model.objective(z,y,v,{"loss":"stage_ce_smooth"})[2]),0)

    def test_bad_loss_anchor_and_class_space_rejected(self):
        z=torch.randn(1,2,3,7);y=torch.zeros(2,3,dtype=torch.long);v=torch.ones(2,3,dtype=torch.bool)
        with self.assertRaises(ValueError):model.objective(z,y,v,{"loss":"unknown"})
        y.fill_(-100);y[:,1]=1
        for pos in ([1.0,1.0],[-1,1],[3,1],[0,1]):
            with self.assertRaises(ValueError):model.objective(z,y,v,{"loss":"final_ce"},pos)
        y.fill_(7)
        with self.assertRaises(ValueError):model.objective(z,y,v,{"loss":"final_ce"})

    def test_augmentation_pairs_and_anchor_targets_are_shared_across_heads(self):
        segments,meta=fixtures();fixed=train_replacements(segments,meta)
        original,pos=anchor_batch(segments,fixed,range(4),False)
        changed,pos2=anchor_batch(segments,fixed,range(4),True)
        self.assertEqual(pos,pos2);self.assertTrue(torch.equal(original[2],changed[2]))
        torch.testing.assert_close(changed[0][...,-7:],changed[1],atol=0,rtol=0)
        for task in model.TASKS:
            m=model.BenchmarkHead(TCNConfig(5,7,8,3,0),task)
            loss,_,smooth=worker.loss_on_batch(m,changed,torch.device("cpu"),pos)
            self.assertTrue(torch.isfinite(loss));self.assertEqual(float(smooth),0)

    def test_CPU_fixture_is_not_accepted_as_real_CUDA_smoke(self):
        with self.assertRaises(RuntimeError):worker.smoke({},"videomae2","logits",{}, {},torch.device("cpu"))


class MetricTests(unittest.TestCase):
    def fixture(self):
        ids=list("abcdef")
        metadata=[{"clip_id":c,"split":"val","recording_key":"record"+str(i//2)} for i,c in enumerate(ids)]
        return np.array([0,0,1,1,2,2]),ids,metadata

    def test_paired_bootstrap_identical_models_zero_interval_and_no_id_leak(self):
        y,ids,meta=self.fixture();r=metrics.recording_interval(y,y,y,ids,meta,replicates=30)
        self.assertEqual(r["percentile_95_interval"],[0,0]);self.assertEqual(r["recording_count"],3)
        self.assertFalse(r["patient_independence_verified"]);worker.public_safe(r)

    def test_bootstrap_deterministic_and_keeps_all_seven_classes(self):
        y,ids,meta=self.fixture();wrong=y.copy();wrong[0]=1
        first=metrics.recording_interval(y,y,wrong,ids,meta,replicates=40)
        self.assertEqual(first,metrics.recording_interval(y,y,wrong,ids,meta,replicates=40))
        self.assertGreater(first["paired_macro_f1_difference"],0)
        self.assertAlmostEqual(metrics.confusion_f1(np.diag([2,2,2,0,0,0,0])),3/7)

    def test_insufficient_recordings_not_falsely_identifiable(self):
        y,ids,meta=self.fixture()
        for r in meta:r["recording_key"]="one"
        self.assertFalse(metrics.recording_interval(y,y,y,ids,meta)["identifiable"])

    def test_wrong_split_duplicate_id_and_noninteger_predictions_rejected(self):
        y,ids,meta=self.fixture();bad=copy.deepcopy(meta);bad[0]["split"]="train"
        for labels,keys,rows in ((y,ids,bad),(y,ids,meta+[meta[0]]),(y.astype(float),ids,meta),(y,ids[:-1],meta)):
            with self.assertRaises(ValueError):metrics.recording_interval(labels,y,y,keys,rows)


class PlanAndReuseTests(unittest.TestCase):
    def args(self,root,assets=None):
        file=root/"assets.json";file.write_text(json.dumps(assets or [{"backbone":"videomae2","index":str(root/"index"),
            "source_protocol_dir":str(root/"source"),"historical_run":str(root/"history")}]))
        return SimpleNamespace(assets=str(file),output=str(root/"new"),gpus="0,1,2",gpu_uuids="GPU-00000000,GPU-11111111,GPU-22222222",
            expected_commit="1"*40,python=sys.executable)

    def base(self,args):
        return {"backbone":"videomae2","output":args.output,"index":"/PRIVATE/index","source_protocol_dir":"/PRIVATE/source",
            "evidence":{"counts":{"train":7372,"val":823},"model_config":{"global_dim":5,"local_dim":7,"hidden_dim":8,"dropout":0},
                        "segments":{"train":{"count":4}},"input_sha256":{"synthetic":"a"*64,**{k:"b"*64 for k in
                            ("source_audit","edges.jsonl","chains_train.jsonl","chains_val.jsonl","manifest_metadata.jsonl")}}}}

    def test_read_only_plan_registers_two_backbones_and_no_GPU_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.args(root)
            rows=json.loads(Path(args.assets).read_text());rows.append({**rows[0],"backbone":"adafocus"})
            Path(args.assets).write_text(json.dumps(rows))
            def child(a):
                b=copy.deepcopy(self.base(args));b["backbone"]=a.backbone;return b
            with patch.object(run_suite.input_run,"plan",side_effect=child),patch.object(run_suite,"wait_released_gpu",side_effect=AssertionError("probed")):
                p=run_suite.plan(args)
            self.assertFalse(Path(args.output).exists());self.assertEqual(p["task_counts"]["registered"],24)
            self.assertEqual([j["backbone"] for j in p["train_waves"][0]], ["videomae2","adafocus","videomae2"])

    def test_one_to_four_card_finite_queue_mapping(self):
        jobs=[{"variant":v,"backbone":"videomae2"} for v in model.TASKS]
        for n in range(1,5):
            waves=run_suite.waves(jobs,list(range(n)),["GPU-"+str(i)*8 for i in range(n)])
            self.assertEqual([j["variant"] for w in waves for j in w],list(model.TASKS))
            self.assertTrue(all(j["gpu"]==i for w in waves for i,j in enumerate(w)))

    def test_two_backbone_source_protocol_mismatch_stops_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.args(root);rows=json.loads(Path(args.assets).read_text())
            rows.append({**rows[0],"backbone":"adafocus"});Path(args.assets).write_text(json.dumps(rows))
            def child(a):
                p=self.base(args);p["backbone"]=a.backbone
                if a.backbone=="adafocus":p["evidence"]["input_sha256"]["chains_val.jsonl"]="c"*64
                return p
            with patch.object(run_suite.input_run,"plan",side_effect=child),self.assertRaises(ValueError):run_suite.plan(args)

    def test_invalid_asset_and_GPU_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.args(root)
            for assets in ([],["not_mapping"],[{"backbone":"other"}],[{"backbone":"videomae2"}]*2,[{"backbone":"videomae2","unknown":1}]):
                Path(args.assets).write_text(json.dumps(assets))
                with self.assertRaises(ValueError):run_suite.plan(args)
            self.args(root);args.gpu_uuids="GPU-00000000,GPU-00000000,GPU-22222222"
            with self.assertRaises(ValueError):run_suite.plan(args)

    def old_fixture(self,root):
        args=self.args(root);p=self.base(args);p["verification_output"]=args.output
        run=root/"old";run.mkdir();d=run/"visual";d.mkdir()
        train={k:copy.deepcopy(model.CONFIG[k]) for k in run_suite.RECIPE_KEYS};train["loss"]="0.5_normal_clip_CE_plus_0.5_preserved_anchor_CE"
        old={"schema":"fsn-input-evidence-native-v1","source_commit":"2"*40,"training":train,"evidence":p["evidence"],
             "backbone":"videomae2","test_read":False,"source_sha256":{name:sha256_file(run_suite.ROOT/name) for name in
             ["experiments/anchor_reference/model.py","experiments/anchor_reference/data.py","experiments/anchor_reference/worker.py",
              "experiments/relation/tcn_baseline.py","experiments/input_evidence/model.py","experiments/input_evidence/worker.py"]}}
        ck={"schema":old["schema"],"source_commit":old["source_commit"],"training":train,"variant":"visual",
            "input_sha256":p["evidence"]["input_sha256"],"epoch":1,"macro_f1":.5,"accuracy":.6,
            "model":model.BenchmarkHead(TCNConfig(**p["evidence"]["model_config"]),"visual").core.state_dict()}
        torch.save(ck,d/"best.pt")
        result={"training_completed":True,"epochs_completed":60,"optimizer_steps":60,"best_epoch":1,"best_macro_f1":.5,
            "best_accuracy_same_F1_checkpoint":.6,"best_sha256":sha256_file(d/"best.pt")}
        for path,value in (("run_protocol.json",old),("suite_result.json",{"status":"complete","training_completed":True,"exit_codes":{"train":0}}),
            ("audit_seal.json",{}),("private_plans.json",{}),("visual/result.json",result),
            ("visual/history.json",[{"epoch":i+1,"cumulative_steps":i+1} for i in range(60)])):
            (run/path).write_text(json.dumps(value))
        return p,run

    def test_identical_completed_native_can_be_explicitly_reused_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,run=self.old_fixture(Path(tmp));before=sha256_file(run/"visual/best.pt")
            ref=run_suite.verify_reuse({"run":str(run),"variant":"visual"},"visual",p)
            self.assertFalse(ref["new_training"]);self.assertEqual(before,sha256_file(run/"visual/best.pt"))

    def test_partial_wrong_A_budget_loss_or_code_reuse_rejected(self):
        for change in ("partial","A","epochs","loss","code"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                p,run=self.old_fixture(Path(tmp));path=run/"run_protocol.json";old=json.loads(path.read_text())
                if change=="partial":
                    (run/"visual/history.json").write_text("[]")
                elif change=="A":old["evidence"]["input_sha256"]["synthetic"]="b"*64
                elif change=="epochs":old["training"]["epochs"]=30
                elif change=="loss":old["training"]["loss"]="changed"
                else:old["source_sha256"]={}
                path.write_text(json.dumps(old))
                with self.assertRaises(ValueError):run_suite.verify_reuse({"run":str(run),"variant":"visual"},"visual",p)

    def test_wrong_variant_and_mature_checkpoint_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,run=self.old_fixture(Path(tmp))
            for task,ref in (("mstcnpp_logits",{"run":str(run),"variant":"visual"}),
                             ("visual",{"run":str(run),"variant":"logits"})):
                with self.assertRaises(ValueError):run_suite.verify_reuse(ref,task,p)

    def test_default_plan_skips_explicit_reuse_in_training_but_not_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.args(root);rows=json.loads(Path(args.assets).read_text())
            rows[0]["reuse"]={"logits":{"run":"/PRIVATE/old","variant":"plain"}}
            Path(args.assets).write_text(json.dumps(rows))
            with patch.object(run_suite.input_run,"plan",return_value=self.base(args)),patch.object(run_suite,"verify_reuse",return_value={"new_training":False}):
                p=run_suite.plan(args)
            self.assertEqual(p["task_counts"],{"registered":12,"new_training":11,"reused":1})
            self.assertNotIn("logits",[j["variant"] for w in p["train_waves"] for j in w])
            self.assertIn("logits",[j["variant"] for w in p["smoke_waves"] for j in w])


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def completed_fixture(self,root):
        segments,metadata=fixtures()
        for r in metadata:r.update(split="val",duration=3.)
        child={"output":str(root),"evidence":{"counts":{"train":12,"val":12},"model_config":{
            "global_dim":5,"local_dim":7,"hidden_dim":8,"blocks":3,"dropout":0},"segments":{"train":{"count":4}},"input_sha256":{"synthetic":"a"*64}}}
        p={"output":str(root),"source_commit":"1"*40,"subplans":{"videomae2":child},"reuse":{"videomae2":{}},
           "task_counts":{"registered":12,"new_training":12,"reused":0}}
        seal={"sampling_hash":"a"*64,"train_replacement_hash":"b"*64,"validation_challenge_hash":"c"*64}
        a=torch.from_numpy(np.concatenate([s.a_logits for s in segments]));y=torch.from_numpy(np.concatenate([s.labels for s in segments]))
        data={"val":{"a":a,"t":a.clone(),"labels":y,"clip_ids":[c for s in segments for c in s.clip_ids]}}
        fixed={"validation":perturb.plans(segments,metadata)}
        for task in model.TASKS:
            m=worker.model(p,"videomae2",task,torch.device("cpu")).eval();z=perturb.infer(m,segments,"cpu");metric=worker.simple_metrics(data["val"],z)
            d=root/task;d.mkdir();identity=worker.checkpoint_identity(p,"videomae2",task,seal)
            torch.save({**identity,"model":m.core.state_dict(),"epoch":1,"macro_f1":metric["macro_f1"],"accuracy":metric["accuracy"]},d/"best.pt")
            result={**identity,**m.parameter_counts(),"training_completed":True,"test_read":False,"new_training":True,"epochs_completed":60,
                "optimizer_steps":60,"best_epoch":1,"best_macro_f1":metric["macro_f1"],"best_accuracy_same_F1_checkpoint":metric["accuracy"],
                "best_sha256":sha256_file(d/"best.pt"),"resources":{"elapsed_seconds":1.,"peak_allocated_bytes":0}}
            (d/"result.json").write_text(json.dumps(result));(d/"history.json").write_text(json.dumps([
                {"epoch":i+1,"cumulative_steps":i+1,"val_macro_f1":metric["macro_f1"]} for i in range(60)]))
        return p,{"train":segments,"val":segments},metadata,fixed,data,seal

    def test_all_twelve_CPU_checkpoint_replay_and_safe_complete_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,segs,meta,fixed,data,seal=self.completed_fixture(root)
            with patch.object(worker.common,"cache",return_value=(fixed,data,seal)):
                worker.evaluate(p,"videomae2",segs,meta,fixed,data,seal,torch.device("cpu"))
            public=json.loads((root/"evaluation/public_safe_summary.json").read_text());worker.public_safe(public)
            self.assertEqual(set(public["models"]),set(model.TASKS))
            for row in public["models"].values():
                self.assertEqual(len(row["metrics"]["per_class"]),7)
                for key in ("shuffle","cross_clinical"):
                    test=row["interventions"][key]
                    self.assertEqual(test["native_eligible_metrics"]["count"],test["eligible_metrics"]["count"])
            with self.assertRaises(FileExistsError):worker.evaluate(p,"videomae2",segs,meta,fixed,data,seal,torch.device("cpu"))
            report.aggregate(p)
            self.assertTrue((root/"aggregate_metrics.csv").exists());self.assertTrue((root/"videomae2_heads_curve.svg").exists())
            self.assertNotIn(str(root),(root/"REPORT.md").read_text())
            with self.assertRaises(FileExistsError):report.aggregate(p)

    def test_partial_history_or_wrong_task_spec_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,_,_,fixed,data,seal=self.completed_fixture(root)
            path=root/"logits/history.json";path.write_text("[]")
            with patch.object(worker.common,"cache",return_value=(fixed,data,seal)),self.assertRaises(RuntimeError):
                worker.validate_completed(p,"videomae2","logits",torch.device("cpu"))
            path=root/"visual/result.json";r=json.loads(path.read_text());r["task_specification"]["scores"]="raw";path.write_text(json.dumps(r))
            with patch.object(worker.common,"cache",return_value=(fixed,data,seal)),self.assertRaises(RuntimeError):
                worker.validate_completed(p,"videomae2","visual",torch.device("cpu"))

    def test_curve_rejects_partial_nonfinite_and_escapes_titles(self):
        self.assertIn("&lt;x&gt;",report.curve_svg({"a":[.5]*60},"<x>"))
        for series in ({"a":[.5]*30},{"a":[float("nan")]*60},{"a":[1.1]*60}):
            with self.assertRaises(ValueError):report.curve_svg(series,"test")


class LauncherTests(unittest.TestCase):
    def fake(self,root,failure=None,bad_smoke=False,launch_failure=False):
        features=root/"features";features.mkdir();index=features/"index";index.write_text("synthetic");source=root/"source";source.mkdir()
        jobs=[{"backbone":"videomae2","variant":v} for v in model.TASKS]
        waves=run_suite.waves(jobs,[0,1,2],["GPU-"+str(i)*8 for i in range(3)])
        out=root/"new";p={"output":str(out),"python":sys.executable,"source_commit":"1"*40,
            "subplans":{"videomae2":{"index":str(index),"source_protocol_dir":str(source),"output":str(out/"videomae2")}},
            "reuse":{"videomae2":{}},"smoke_waves":waves,"train_waves":waves,"task_counts":{"registered":12,"new_training":12,"reused":0},
            "evaluation_slots":[{"backbone":"videomae2","variant":"all","gpu":0,"gpu_uuid":"GPU-00000000"}]}
        started,waited=[],[]
        class Child:
            def __init__(self,command,**kwargs):
                stage="unit_tests" if "unittest" in command else command[command.index("--stage")+1]
                task="all" if "--variant" not in command else command[command.index("--variant")+1]
                if launch_failure and stage=="train" and task=="logits_aug":raise RuntimeError("synthetic launch failed")
                self.pid=100+len(started);self.stage=stage;self.task=task;started.append((stage,task))
                if stage in ("smoke","train"):
                    uuid=next(j["gpu_uuid"] for w in waves for j in w if j["variant"]==task)
                elif stage=="evaluate":uuid="GPU-00000000"
                else:uuid=""
                assert kwargs["env"]["CUDA_VISIBLE_DEVICES"]==uuid
            def wait(self):
                waited.append((self.stage,self.task))
                if (self.stage,self.task)==failure:return 1
                if self.stage=="smoke":
                    d=out/"videomae2/smoke";d.mkdir(exist_ok=True)
                    (d/(self.task+".json")).write_text(json.dumps({"all_passed":not bad_smoke,"task_specification":model.TASKS[self.task]}))
                if self.stage=="aggregate":
                    (out/"public_safe_summary.json").write_text(json.dumps({"training_completed":True,"backbones":{"videomae2":{}}}))
                return 0
        with patch.dict(os.environ,{},clear=True),patch("builtins.print"),patch.object(run_suite,"unchanged"),patch.object(run_suite,"wait_released_gpu"), \
             patch.object(run_suite.subprocess,"Popen",Child),patch.object(worker,"validate_completed"):
            if failure or bad_smoke or launch_failure:
                with self.assertRaises(RuntimeError):run_suite.execute(p)
            else:run_suite.execute(p)
        return p,started,waited

    def test_all_smokes_before_any_training_and_complete_result_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,started,waited=self.fake(Path(tmp));last=max(i for i,j in enumerate(started) if j[0]=="smoke")
            self.assertEqual(sum(s=="smoke" for s,v in started),12)
            self.assertGreater(min(i for i,j in enumerate(started) if j[0]=="train"),last)
            self.assertEqual(started,waited);self.assertTrue(json.loads((Path(p["output"])/"suite_result.json").read_text())["training_completed"])

    def test_failed_wave_waits_peers_and_stops_later_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,started,waited=self.fake(Path(tmp),failure=("train","logits"))
            self.assertEqual([v for s,v in started if s=="train"],list(model.TASKS)[:3])
            self.assertIn(("train","visual"),waited);self.assertFalse(any(s=="evaluate" for s,v in started))
            self.assertEqual(json.loads((Path(p["output"])/"status.json").read_text())["status"],"failed")

    def test_launch_failure_waits_existing_peer_and_does_not_kill_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,started,waited=self.fake(Path(tmp),launch_failure=True)
            self.assertEqual([v for s,v in started if s=="train"],["logits"])
            self.assertIn(("train","logits"),waited)

    def test_smoke_failure_blocks_all_formal_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,started,_=self.fake(Path(tmp),bad_smoke=True)
            self.assertFalse(any(s=="train" for s,v in started))

    def test_outer_CUDA_mapping_rejected_before_writes(self):
        with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"0"}),self.assertRaises(RuntimeError):run_suite.execute({})

    def test_existing_output_is_not_overwritten_or_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"new").mkdir();sentinel=root/"new/old.txt";sentinel.write_text("preserve")
            with self.assertRaises(FileExistsError):self.fake(root)
            self.assertEqual(sentinel.read_text(),"preserve")


if __name__=="__main__":unittest.main()
