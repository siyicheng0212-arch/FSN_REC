import copy
from pathlib import Path
import tempfile
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from experiments.anchor_reference.data import anchor_view, sampling_plan, train_replacements, plan_hash
from experiments.anchor_reference.model import AnchorReferenceTCN, VARIANTS, neighbor_indices
from experiments.anchor_reference.run_suite import waves
from experiments.anchor_reference.worker import loss_on_batch, native_oracle, public_safe
from experiments.anchor_reference import run_suite
from experiments.relation.tcn_baseline import TCNConfig, Segment, collate_segments


def inputs(lengths=(6,3,1),seed=31):
    generator=torch.Generator().manual_seed(seed)
    f=torch.randn(len(lengths),max(lengths),19,generator=generator)
    a=f[...,-7:].clone()
    valid=torch.arange(max(lengths))[None]<torch.tensor(lengths)[:,None]
    return f,a,valid


def fixtures():
    segments=[];metadata=[]
    for split,record,collection,kind in [("train","r1","lishui","clinical"),
        ("train","r2","lishui","clinical"),("train","r3","menzhen","clinical"),
        ("train","r4","FSN","network")]:
        ids=[record+str(j) for j in range(3)]
        f=np.random.default_rng(len(segments)).normal(size=(3,19)).astype(np.float32)
        a=f[:,-7:].copy();labels=np.array([0,1,2])
        segments.append(Segment(ids,f,a,labels,[kind]*3))
        metadata.extend({"clip_id":c,"split":split,"recording_key":record,
                         "source_collection":collection,"source_kind":kind,"label_id":int(labels[j])}
                        for j,c in enumerate(ids))
    return segments,metadata


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def model(self,v="reference",dtype=torch.float32):
        return AnchorReferenceTCN(TCNConfig(5,7,8,3,0),v,projection_dim=4,gate_hidden_dim=5).to(dtype)

    def test_shared_initial_weights(self):
        expected=self.model("plain").base.state_dict()
        for v in VARIANTS:
            actual=self.model(v).base.state_dict()
            self.assertEqual(set(expected),set(actual))
            for k in expected:self.assertTrue(torch.equal(expected[k],actual[k]),(v,k))

    def test_constructor_does_not_consume_global_rng(self):
        torch.manual_seed(28);before=torch.get_rng_state().clone()
        self.model("reference")
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_initial_all_variants_replay_native_classifier(self):
        f,a,valid=inputs();batch=(f,a,None,valid,None)
        for v in VARIANTS:
            m=self.model(v).eval()
            with torch.no_grad():z,d=m(f,a,valid,return_details=True)
            torch.testing.assert_close(z,native_oracle(m,batch,"cpu"),atol=2e-6,rtol=2e-6)
            for k,c in d["coefficients"].items():self.assertTrue(torch.equal(c,torch.ones_like(c)),k)

    def test_forced_unit_after_nontrivial_gate(self):
        f,a,valid=inputs();m=self.model().eval()
        with torch.no_grad():m.reference.score[-1].weight.normal_();m.reference.score[-1].bias.fill_(-2)
        unit=m(f,a,valid,mode="unit")
        torch.testing.assert_close(unit,native_oracle(m,(f,a,None,valid,None),"cpu"),atol=2e-6,rtol=2e-6)
        self.assertFalse(torch.allclose(unit,m(f,a,valid),atol=1e-8,rtol=1e-8))

    def test_double_precision_kernel_layout(self):
        f,a,valid=inputs((11,7,2));m=self.model(dtype=torch.float64).eval()
        f,a=f.double(),a.double()
        torch.testing.assert_close(m(f,a,valid),native_oracle(m,(f,a,None,valid,None),"cpu"),atol=1e-12,rtol=1e-12)

    def test_zero_neighbors_equals_singleton_classification(self):
        f,a,valid=inputs((7,3,1));m=self.model().eval();center=m(f,a,valid,mode="zero")
        for b,t in zip(*torch.where(valid)):
            single=m(f[b:b+1,t:t+1],a[b:b+1,t:t+1],torch.ones(1,1,dtype=torch.bool))
            torch.testing.assert_close(center[b,t],single[0,0],atol=2e-6,rtol=2e-6)

    def test_singleton_keeps_native_center_not_forced_A(self):
        f,a,valid=inputs((1,));m=self.model().eval();z=m(f,a,valid)
        torch.testing.assert_close(z,native_oracle(m,(f,a,None,valid,None),"cpu"))
        self.assertFalse(torch.equal(z,a))

    def test_padding_cannot_change_valid_outputs(self):
        f,a,valid=inputs((7,3));m=self.model().eval()
        with torch.no_grad():m.reference.score[-1].weight.fill_(.3)
        expected=m(f,a,valid);other=f.clone();other[~valid]*=10000
        actual=m(other,other[...,-7:],valid)
        torch.testing.assert_close(expected[valid],actual[valid],atol=1e-6,rtol=1e-6)

    def test_directed_kernel_span_and_edge_masks(self):
        f,a,valid=inputs((7,3,1));m=self.model().eval();_,d=m(f,a,valid,return_details=True)
        self.assertEqual(set(d["coefficients"]),{f"layer{i}_offset{o:+d}" for i in range(3) for o in (-2**i,2**i)})
        for i in range(3):
            for off in (-2**i,2**i):
                ids,inside=neighbor_indices(7,off,f.device)
                self.assertTrue(torch.equal(d["edge_masks"][f"layer{i}_offset{off:+d}"],valid & valid[:,ids] & inside[None]))

    def test_gate_has_finite_gradient_and_projection_learns_later(self):
        f,a,valid=inputs((7,7));y=torch.arange(14).reshape(2,7)%7
        m=self.model();optimizer=torch.optim.AdamW(m.parameters(),lr=.01)
        before=m.reference.score[-1].weight.detach().clone()
        first=None
        for step in range(3):
            optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(m(f,a,valid)[valid],y[valid]);loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(m.reference.score[-1].weight.grad.abs().max().item(),0)
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
            if step==0:first=m.reference.project[0].weight.grad.abs().max().item()
            else:self.assertGreater(m.reference.project[0].weight.grad.abs().max().item(),0)
            optimizer.step()
        self.assertEqual(first,0)
        self.assertFalse(torch.equal(before,m.reference.score[-1].weight))
        self.assertIsNone(f.grad)

    def test_gate_is_inside_each_temporal_block(self):
        f,a,valid=inputs((7,));m=self.model().eval()
        with torch.no_grad():m.reference.score[-1].bias.fill_(-1)
        z,d=m(f,a,valid,return_details=True)
        unit=m(f,a,valid,mode="unit")
        self.assertFalse(torch.allclose(z,unit))
        # It is not multiplication of the final output by a common scalar:
        self.assertFalse(torch.allclose(z,unit*(2*torch.tensor(-1.).sigmoid()),atol=1e-6,rtol=1e-6))
        self.assertEqual(len(d["coefficients"]),6)

    def test_scalar_is_one_shared_parameter(self):
        m=self.model("scalar_aug")
        self.assertEqual(m.parameter_counts()["reference_parameters"],1)

    def test_override_real_positions_changes_outputs(self):
        f,a,valid=inputs((9,));m=self.model().eval()
        with torch.no_grad():m.reference.score[-1].weight.normal_()
        z,d=m(f,a,valid,return_details=True)
        flipped={k:torch.flip(v,[-1]) for k,v in d["coefficients"].items()}
        p=m(f,a,valid,overrides=flipped)
        self.assertGreater(float((z-p).abs().max()),1e-7)

    def test_missing_override_is_not_accepted(self):
        f,a,valid=inputs();m=self.model()
        with self.assertRaises(ValueError):m(f,a,valid,overrides={})

    def test_mismatched_A_logits_rejected(self):
        f,a,valid=inputs()
        with self.assertRaises(ValueError):self.model()(f,a+1,valid)

    def test_invalid_masks_rejected(self):
        f,a,valid=inputs();valid[1]=torch.tensor([True,False,True,False,False,False])
        with self.assertRaises(ValueError):self.model()(f,a,valid)

    def test_nonfinite_rejected(self):
        f,a,valid=inputs();f[0,0,0]=float("nan")
        with self.assertRaises(ValueError):self.model()(f,a,valid)

    def test_unsupported_conv_rejected(self):
        f,a,valid=inputs();m=self.model();m.base.blocks[0].conv=torch.nn.Conv1d(8,8,1)
        with self.assertRaises(RuntimeError):m(f,a,valid)

    def test_strict_checkpoint_load(self):
        m=self.model();n=self.model();n.load_state_dict(m.state_dict(),strict=True)
        bad=dict(m.state_dict());bad.pop("base.classifier.bias")
        with self.assertRaises(RuntimeError):n.load_state_dict(bad,strict=True)

    def test_native_output_does_not_add_A_logits(self):
        f,a,valid=inputs();m=self.model().eval()
        with torch.no_grad():m.base.classifier.weight.zero_();m.base.classifier.bias.zero_()
        self.assertTrue(torch.equal(m(f,a,valid),torch.zeros_like(a)))


class DataTests(unittest.TestCase):
    def test_train_plan_is_label_free_and_reproducible(self):
        s,meta=fixtures();p=train_replacements(s,meta)
        changed=copy.deepcopy(meta)
        for r in changed:r["label_id"]=(r["label_id"]+4)%7
        self.assertEqual(p,train_replacements(s,changed))
        self.assertEqual(plan_hash(p),plan_hash(train_replacements(s,meta)))

    def test_donors_are_train_different_record_and_prefer_collection(self):
        s,meta=fixtures();lookup={r["clip_id"]:r for r in meta};p=train_replacements(s,meta)
        for r in p["rows"]:
            a=lookup[s[r["segment"]].clip_ids[r["anchor"]]]
            for pos,ds,dj in r["swaps"]:
                donor=lookup[s[ds].clip_ids[dj]]
                self.assertEqual(donor["split"],"train")
                self.assertNotEqual(a["recording_key"],donor["recording_key"])
                if r["same_collection"]:self.assertEqual(a["source_collection"],donor["source_collection"])

    def test_val_clip_in_training_is_rejected(self):
        s,meta=fixtures();meta[0]["split"]="val"
        with self.assertRaises(ValueError):train_replacements(s,meta)

    def test_only_anchor_has_truth_and_donor_scores_are_paired(self):
        s,meta=fixtures();r=train_replacements(s,meta)["rows"][0];view=anchor_view(s,r,True)
        j=r["anchor"];original=s[r["segment"]]
        np.testing.assert_array_equal(view.features[j],original.features[j])
        np.testing.assert_array_equal(view.a_logits[j],original.a_logits[j])
        self.assertEqual(view.labels[j],original.labels[j]);self.assertEqual(sum(view.labels>=0),1)
        for pos,ds,dj in r["swaps"]:
            np.testing.assert_array_equal(view.features[pos],s[ds].features[dj])
            np.testing.assert_array_equal(view.a_logits[pos],s[ds].a_logits[dj])

    def test_normal_anchor_view_preserves_neighbors(self):
        s,meta=fixtures();r=train_replacements(s,meta)["rows"][0];view=anchor_view(s,r,False)
        np.testing.assert_array_equal(view.features,s[r["segment"]].features)
        self.assertEqual(sum(view.labels>=0),1)

    def test_anchor_cannot_be_replaced(self):
        s,meta=fixtures();r=train_replacements(s,meta)["rows"][0];r["swaps"].append([r["anchor"],1,1])
        with self.assertRaises(ValueError):anchor_view(s,r,True)

    def test_shared_sampler_does_not_use_torch_model_rng(self):
        a=sampling_plan(17,18);torch.manual_seed(192);_ = torch.rand(20)
        self.assertEqual(a,sampling_plan(17,18));self.assertEqual(len(a["normal_orders"]),60)
        for row in a["normal_orders"]:self.assertEqual(sorted(row),list(range(17)))

    def test_anchor_loss_does_not_supervise_donors(self):
        s,meta=fixtures();r=train_replacements(s,meta)["rows"][0]
        batch=collate_segments([anchor_view(s,r,True)])
        m=AnchorReferenceTCN(TCNConfig(5,7,8,3,0),"plain")
        loss=loss_on_batch(m,batch,"cpu",[r["anchor"]]);self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(ValueError):loss_on_batch(m,batch,"cpu")

    def test_no_eligible_donor_rejected(self):
        s,meta=fixtures()
        with self.assertRaises(ValueError):train_replacements(s[:1],meta[:3])


class SafetyTests(unittest.TestCase):
    def test_four_gpu_mapping_and_fifth_control_barrier(self):
        w=waves(VARIANTS,[0,1,2,3],["u0","u1","u2","u3"])
        self.assertEqual([r["variant"] for r in w[0]],list(VARIANTS[:4]))
        self.assertEqual(w[1][0],{"variant":"scalar_aug","gpu":0,"gpu_uuid":"u0"})

    def test_two_gpu_waves(self):
        w=waves(VARIANTS,[0,1],["u0","u1"])
        self.assertEqual([len(x) for x in w],[2,2,1])

    def test_three_gpu_waves(self):
        w=waves(VARIANTS,[0,1,2],["u0","u1","u2"])
        self.assertEqual([len(x) for x in w],[3,2])
        self.assertEqual([r["variant"] for r in w[0]],["plain","plain_aug","reference"])
        self.assertEqual(w[1],[{"variant":"reference_aug","gpu":0,"gpu_uuid":"u0"},
                              {"variant":"scalar_aug","gpu":1,"gpu_uuid":"u1"}])

    def test_single_gpu_queue(self):self.assertEqual(len(waves(VARIANTS,[0],["u"])),5)

    def test_duplicate_devices_rejected(self):
        for gp,uu in [([0,0],["u0","u1"]),([0,1],["u0","u0"]),
                      ([0,1,2],["u0","u1"]),([-1],["u0"]),
                      ([0,1,2,3,4],["u0","u1","u2","u3","u4"])]:
            with self.assertRaises(ValueError):waves(VARIANTS,gp,uu)

    def test_registered_controls_cannot_be_removed(self):
        with self.assertRaises(ValueError):waves(VARIANTS[:4],[0],["u"])

    def test_public_summary_blocks_private_content(self):
        for x in [{"clip_id":"c"},{"text":"/root/private/data"},{"recording_key":"id"}]:
            with self.assertRaises(RuntimeError):public_safe(x)
        self.assertEqual(public_safe({"macro_f1":.9}),{"macro_f1":.9})


class LauncherTests(unittest.TestCase):
    def prepare_plan_arguments(self,root):
        source=root/"source";source.mkdir();base=root/"historical";base.mkdir()
        inp=root/"features";inp.mkdir();index=inp/"index.jsonl";index.write_text("synthetic-test-only")
        for n in run_suite.historical_video.BASE_FILES:
            f=base/n;f.parent.mkdir(exist_ok=True);f.write_text("synthetic-test-only")
        return SimpleNamespace(index=str(index),source_protocol_dir=str(source),historical_run=str(base),
            output=str(root/"new_private"),expected_commit="1"*40,gpus="0,1,2,3",
            gpu_uuids=",".join("GPU-"+str(i)*16 for i in range(4)),python=sys.executable,backbone="videomae2")

    def test_read_only_plan_never_queries_GPU_or_creates_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.prepare_plan_arguments(Path(tmp))
            info={"model_config":{"global_dim":5,"local_dim":7,"hidden_dim":8,"blocks":3,"dropout":.1}}
            with patch.object(run_suite,"repository_state",return_value={"commit":"1"*40,"clean":True}), \
                 patch.object(run_suite,"evidence",return_value=({},[],info)), \
                 patch.object(run_suite.historical_video,"plan",return_value={"base_result":{"best_val_macro_f1":.7}}), \
                 patch.object(run_suite,"code_hashes",return_value={}), \
                 patch.object(run_suite,"wait_released_gpu",side_effect=AssertionError("GPU queried")):
                p=run_suite.plan(args)
            self.assertTrue(p["plan_only_does_not_probe_GPU_or_create_output"])
            self.assertFalse(Path(args.output).exists())

    def fake_execution(self,root,failure,gpu_count=4):
        args=self.prepare_plan_arguments(root)
        p={"output":args.output,"index":args.index,"source_protocol_dir":args.source_protocol_dir,
            "source_commit":"1"*40,"python":sys.executable,"waves":run_suite.waves(VARIANTS,list(range(gpu_count)),
            ["GPU-"+str(i)*16 for i in range(gpu_count)]),"variants":list(VARIANTS)}
        launched=[];waited=[]
        class Child:
            def __init__(self,command,**kwargs):
                self.pid=100+len(launched)
                stage="unit_tests" if "unittest" in command else command[command.index("--stage")+1]
                variant=command[command.index("--variant")+1] if "--variant" in command else "all"
                self.stage,self.variant=stage,variant;launched.append((stage,variant))
            def wait(self):
                waited.append((self.stage,self.variant));out=Path(p["output"])
                if (self.stage,self.variant)==failure:return 1
                if self.stage=="smoke":
                    d=out/"smoke";d.mkdir(exist_ok=True);(d/(self.variant+".json")).write_text(json.dumps({"all_passed":True}))
                if self.stage=="train":
                    d=out/self.variant;d.mkdir(exist_ok=True);(d/"result.json").write_text(json.dumps({"training_completed":True,"epochs_completed":60}))
                if self.stage=="evaluate":
                    d=out/"evaluation";d.mkdir();(d/"public_safe_summary.json").write_text(json.dumps({"training_completed":True,"models":dict.fromkeys(VARIANTS,{})}))
                return 0
        with patch.dict("os.environ",{},clear=True),patch.object(run_suite,"unchanged"), \
             patch.object(run_suite,"wait_released_gpu"),patch.object(run_suite.subprocess,"Popen",Child):
            if failure:
                with self.assertRaises(RuntimeError):run_suite.execute(p)
            else:run_suite.execute(p)
        return p,launched,waited

    def test_all_smokes_precede_training_and_scalar_waits_for_first_wave(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,started,waited=self.fake_execution(Path(tmp),None)
            smoke=[i for i,r in enumerate(started) if r[0]=="smoke"]
            train=[i for i,r in enumerate(started) if r[0]=="train"]
            self.assertLess(max(smoke),min(train))
            self.assertEqual([r[1] for r in waited if r[0]=="train"],list(VARIANTS))
            self.assertEqual(json.loads((Path(p["output"])/"suite_result.json").read_text())["status"],"complete")

    def test_train_failure_waits_for_all_peers_and_never_starts_scalar(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,started,waited=self.fake_execution(Path(tmp),("train","plain"))
            self.assertEqual([r[1] for r in waited if r[0]=="train"],list(VARIANTS[:4]))
            self.assertNotIn(("train","scalar_aug"),started)
            self.assertNotIn(("evaluate","all"),started)
            self.assertFalse(json.loads((Path(p["output"])/"status.json").read_text())["training_completed"])

    def test_three_gpu_queue_completes_all_five_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,started,waited=self.fake_execution(Path(tmp),None,3)
            self.assertEqual([r[1] for r in waited if r[0]=="train"],list(VARIANTS))
            self.assertLess(max(i for i,r in enumerate(started) if r[0]=="smoke"),
                            min(i for i,r in enumerate(started) if r[0]=="train"))
            self.assertIn(("evaluate","all"),waited)

    def test_three_gpu_failure_stops_second_wave_after_peers_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,started,waited=self.fake_execution(Path(tmp),("train","plain_aug"),3)
            self.assertEqual([r[1] for r in waited if r[0]=="train"],list(VARIANTS[:3]))
            self.assertNotIn(("train","reference_aug"),started)
            self.assertNotIn(("train","scalar_aug"),started)

    def test_failed_smoke_prevents_every_formal_training_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,started,waited=self.fake_execution(Path(tmp),("smoke","reference"))
            self.assertEqual(len([r for r in waited if r[0]=="smoke"]),4)
            self.assertFalse(any(r[0]=="train" for r in started))

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.prepare_plan_arguments(root);out=Path(args.output);out.mkdir()
            marker=out/"original";marker.write_text("keep")
            p={"output":str(out),"index":args.index,"source_protocol_dir":args.source_protocol_dir,
                "waves":run_suite.waves(VARIANTS,[0],["GPU-00000000"])}
            with patch.dict("os.environ",{},clear=True),patch.object(run_suite,"unchanged"),patch.object(run_suite,"wait_released_gpu"):
                with self.assertRaises(FileExistsError):run_suite.execute(p)
            self.assertEqual(marker.read_text(),"keep")

    def test_outer_cuda_mapping_rejected_without_any_launch(self):
        with patch.dict("os.environ",{"CUDA_VISIBLE_DEVICES":"0"}),patch.object(run_suite,"wait_released_gpu") as guard:
            with self.assertRaises(RuntimeError):run_suite.execute({})
            guard.assert_not_called()


if __name__=="__main__":unittest.main()
