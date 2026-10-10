"""Synthetic CPU tests for protocol seals, fixed samplers and paired diagnostics."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments.hospital_temporal import features, run, worker
from experiments.hospital_temporal.model import TemporalVariant, VARIANTS, temporal_contribution
from experiments.metrics import compute_classification_metrics
from experiments.relation.tcn_baseline import Segment
from tests.test_hospital_temporal import row


def json_file(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def records_and_features():
    records = [row("synthetic-train-a"),row("synthetic-train-b",1,2,label_id=1),
        row("synthetic-train-c",record_id="other-train",video_path="other-video",group_id="other-group",label_id=2),
        row("synthetic-val-a",split="val",source_collection="menzhen",record_id="val-record",
            video_path="val-video",group_id="val-group",label_id=3),
        row("synthetic-val-b",1,2,split="val",source_collection="menzhen",record_id="val-record",
            video_path="val-video",group_id="val-group",label_id=4),
        row("synthetic-val-c",split="val",source_collection="menzhen",record_id="other-val-record",
            video_path="other-val-video",group_id="other-val-group",label_id=5)]
    torch.manual_seed(91)
    x = torch.randn(6,1543)
    y = torch.tensor([r["label_id"] for r in records])
    return records,x,y


def test_realistic_segment_partition_and_no_cross_split():
    records,x,y = records_and_features()
    segments = worker.make_segments(x,y,records)
    assert [len(s.clip_ids) for s in segments["train"]]==[1,2]
    assert sorted(len(s.clip_ids) for s in segments["val"])==[1,2]
    for split,items in segments.items():
        assert {clip for s in items for clip in s.clip_ids}=={r["clip_id"] for r in records if r["split"]==split}
        for s in items:
            assert np.array_equal(s.features[:,-7:],s.a_logits)


def test_sampler_is_fixed_and_does_not_consume_global_rng():
    torch.manual_seed(9)
    state = torch.random.get_rng_state().clone()
    a = worker.sampling_plan(13)
    _ = TemporalVariant("relative_tcn")
    b = worker.sampling_plan(13)
    assert a==b and torch.equal(state,torch.random.get_rng_state())
    assert len(a)==100 and all(sorted(v)==list(range(13)) for v in a)


def test_nonlinear_relative_operator_is_not_just_linear_kernel_change():
    conv = torch.nn.Conv1d(1,1,3,padding=1,bias=False)
    with torch.no_grad(): conv.weight.fill_(1)
    valid = torch.ones(1,3,dtype=torch.bool)
    h = torch.tensor([[[1.,3.,2.]]])
    actual = temporal_contribution(h,valid,conv,1,relative=True)
    expected = h.clone()
    expected[0,0,0]+=F.gelu(torch.tensor(2.))
    expected[0,0,1]+=F.gelu(torch.tensor(-2.))+F.gelu(torch.tensor(-1.))
    expected[0,0,2]+=F.gelu(torch.tensor(1.))
    torch.testing.assert_close(actual,expected)
    doubled = temporal_contribution(2*h,valid,conv,1,relative=True)
    assert not torch.allclose(doubled,2*actual)


class AnchorAudit(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.views=[]
    def forward(self,x,a,valid,mode="native"):
        assert torch.equal(x[...,-7:],a)
        self.views.append(x.cpu().clone())
        return a


@pytest.mark.parametrize("condition",["shuffle_neighbors","cross_record_neighbors"])
def test_challenges_anchor_and_donor_pair_preservation(condition):
    records,x,y=records_and_features()
    records=[r for r in records if r["split"]=="val"]
    val_x=x[3:]
    items=[Segment([r["clip_id"] for r in records[:2]],val_x[:2].numpy(),val_x[:2,-7:].numpy(),y[3:5].numpy(),["clinical"]*2),
           Segment([records[2]["clip_id"]],val_x[2:].numpy(),val_x[2:,-7:].numpy(),y[5:].numpy(),["clinical"])]
    before=copy.deepcopy(items)
    model=AnchorAudit()
    score,changed=worker.challenge(model,items,records,torch.device("cpu"),condition)
    assert torch.equal(score,val_x[:,-7:])
    assert changed==(2 if condition=="cross_record_neighbors" else 0)
    # Two-clip chains have one neighbor: shuffling it cannot change positions.
    if condition=="cross_record_neighbors":
        assert torch.equal(model.views[0][0,0],val_x[0])
        assert torch.equal(model.views[0][0,1],val_x[2])
        assert torch.equal(model.views[1][0,1],val_x[1])
        assert torch.equal(model.views[1][0,0],val_x[2])
    assert all(np.array_equal(a.features,b.features) for a,b in zip(before,items))


def test_prediction_coverage_and_singleton_summary():
    records,x,y=records_and_features()
    segments=worker.make_segments(x,y,records)
    subset=[r for r in records if r["split"]=="val"]
    model=TemporalVariant("plain_tcn").eval()
    scores=worker.predictions(model,segments["val"],subset,torch.device("cpu"))
    assert torch.equal(scores,x[3:,-7:])
    with pytest.raises(ValueError):
        worker.predictions(model,segments["val"]*2,subset,torch.device("cpu"))
    with pytest.raises(ValueError):
        worker.predictions(model,segments["val"][:1],subset,torch.device("cpu"))
    metrics=worker.summary(scores,subset,scores)
    assert metrics["paired_vs_A"]["changed_count"]==0
    assert len(metrics["metrics"]["all"]["per_class"])==7


def completed_fixture(tmp_path,monkeypatch):
    a=tmp_path/"A"
    a.mkdir(); (a/"train").mkdir(); (a/"train/best.pt").write_bytes(b"synthetic weight identifier only")
    digest=features.sha(a/"train/best.pt")
    json_file(a/"suite_result.json",{"status":"complete","training_completed":True,"evaluation_completed":True})
    json_file(a/"train/result.json",{"training_completed":True,"optimizer_steps":features.TOTAL_UPDATES,
        "selection":features.SELECTION,"best_checkpoint_sha256":digest})
    json_file(a/"evaluation/result.json",{"dev_best_replayed":True,"best_checkpoint_sha256":digest})
    (a/"launcher_logs").mkdir()
    for name in ("unit_tests","seal_data","train_cache","val_cache","smoke","train","evaluation"):
        (a/f"launcher_logs/{name}.exit_code").write_text("0\n")
    monkeypatch.setattr(features,"audit",lambda path: ({},{}))
    return a,digest


def test_completed_A_requires_full_budget_replay_hash_and_every_exit(tmp_path,monkeypatch):
    a,digest=completed_fixture(tmp_path,monkeypatch)
    assert features.completed_A(a)[1]==digest
    (a/"launcher_logs/train.exit_code").write_text("1\n")
    with pytest.raises(ValueError): features.completed_A(a)
    (a/"launcher_logs/train.exit_code").write_text("0\n")
    json_file(a/"evaluation/result.json",{"dev_best_replayed":False,"best_checkpoint_sha256":digest})
    with pytest.raises(ValueError): features.completed_A(a)
    json_file(a/"evaluation/result.json",{"dev_best_replayed":True,"best_checkpoint_sha256":digest})
    (a/"train/best.pt").write_bytes(b"changed weight")
    with pytest.raises(ValueError): features.completed_A(a)


def test_frozen_feature_seal_rejects_order_label_nonfinite_and_old_A(tmp_path,monkeypatch):
    records,x,y=records_and_features()
    a=tmp_path/"A"; directory=tmp_path/"features";directory.mkdir()
    metrics={s:compute_classification_metrics(x[[i for i,r in enumerate(records) if r["split"]==s],-7:],
        y[[i for i,r in enumerate(records) if r["split"]==s]],[r for r in records if r["split"]==s])
        for s in ("train","val")}
    monkeypatch.setattr(features,"EXPECTED_COUNTS",{"train":3,"val":3})
    monkeypatch.setattr(features,"completed_A",lambda path: ({"best_metrics":metrics["val"]},"a"*64))
    monkeypatch.setattr(features,"manifest_rows",lambda path,s: ([r for r in records if r["split"]==s],{}))
    original_sha=features.sha
    monkeypatch.setattr(features,"sha",lambda path: "c"*64 if Path(path).name=="protocol.json" else original_sha(path))
    payload={"schema":features.SCHEMA,"features":x,"labels":y,"clip_ids":[r["clip_id"] for r in records]}
    seal={"schema":features.SCHEMA,"status":"complete","counts":{"train":3,"val":3},"A_frozen":True,
        "every_clip_independent_logit_replay":True,"R_D_enabled":False,"A_checkpoint_sha256":"a"*64,
        "protocol_sha256":"c"*64,"A_metrics":metrics}
    def save(value,header):
        torch.save(value,directory/"private_features.pt")
        json_file(directory/"seal.json",{**header,"features_sha256":original_sha(directory/"private_features.pt")})
    save(payload,seal)
    assert torch.equal(features.load(a,directory)[0],x)
    for key in ("clip_ids","labels","features"):
        altered=copy.deepcopy(payload)
        if key=="clip_ids": altered[key]=list(reversed(altered[key]))
        elif key=="labels": altered[key][0]=6
        else: altered[key][0,0]=float("nan")
        save(altered,seal)
        with pytest.raises(ValueError): features.load(a,directory)
    save(payload,{**seal,"A_checkpoint_sha256":"d"*64})
    with pytest.raises(ValueError): features.load(a,directory)


def test_unified_small_cpu_train_evaluate_replay_and_aggregate_privacy(tmp_path,monkeypatch):
    records,x,y=records_and_features()
    extra=[row("synthetic-extra-a",record_id="extra-train",video_path="extra-video",group_id="extra-group",label_id=0),
           row("synthetic-extra-b",1,2,record_id="extra-train",video_path="extra-video",group_id="extra-group",label_id=1)]
    records=records[:3]+extra+records[3:]
    x=torch.cat((x[:3],torch.randn(2,1543),x[3:]))
    y=torch.tensor([r["label_id"] for r in records])
    split_records={s:[r for r in records if r["split"]==s] for s in ("train","val")}
    segments=worker.make_segments(x,y,records)
    configuration=copy.deepcopy(worker.CONFIG)
    configuration["training"]["epochs"]=2
    configuration["training"]["batch_segments"]=1
    monkeypatch.setattr(worker,"CONFIG",configuration)
    seal={"A_checkpoint_sha256":"a"*64,"features_sha256":"b"*64,"protocol_sha256":"c"*64}
    monkeypatch.setattr(worker,"load",lambda *args: (x,y,records,seal))
    for variant in VARIANTS:
        output=tmp_path/variant;output.mkdir()
        result=worker.train(segments,split_records,variant,output,torch.device("cpu"),seal)
        assert result["epochs_completed"]==2 and result["optimizer_steps"]==6
        assert all(v>0 for v in json.loads((output/"history.json").read_text())[-1]["gradient_max"].values())
    out=worker.evaluate_all(tmp_path/"A",tmp_path/"features",tmp_path,torch.device("cpu"))
    assert len(out["variants"])==6 and out["independent_test_evaluated"] is False
    encoded=json.dumps(out)
    assert not any(k in encoded for k in ("synthetic-train","synthetic-val","clip_id","record_id","/root/","/Users/"))
    record=json.loads((tmp_path/"relative_tcn/result.json").read_text())
    record["optimizer_steps"]=1
    json_file(tmp_path/"relative_tcn/result.json",record)
    with pytest.raises(ValueError): worker.evaluate_all(tmp_path/"A",tmp_path/"features",tmp_path,torch.device("cpu"))


def test_fresh_output_does_not_cover_input_or_old_output(tmp_path):
    for name in ("A","source","old"): (tmp_path/name).mkdir()
    for target in (tmp_path/"old",tmp_path/"A/new",tmp_path):
        with pytest.raises(ValueError): run.fresh_output(target,tmp_path/"A",tmp_path/"source")
    assert run.fresh_output(tmp_path/"new",tmp_path/"A",tmp_path/"source")==tmp_path/"new"


def test_readonly_plan_does_not_probe_gpu_or_create_output(tmp_path,monkeypatch):
    for name in ("A","source"): (tmp_path/name).mkdir()
    monkeypatch.setattr(run,"verify_code",lambda expected: expected)
    monkeypatch.setattr(run.visual,"verify_official_source",lambda path: None)
    monkeypatch.setattr(run.visual,"verify_official_checkpoint",lambda path: {})
    monkeypatch.setattr(run,"audit",lambda path: ({"protocol_sha256":"c"*64,"manifest_sha256":{},
        "splits":{"train":{"segments":3062}}},{}))
    monkeypatch.setattr(run,"dependency_state",lambda path: {"ready":False,"phase":"train_cache"})
    monkeypatch.setattr(run,"available_gpu",lambda: pytest.fail("read-only must not probe CUDA/GPU"))
    args=argparse.Namespace(expected_commit="d"*40,output=tmp_path/"new",a_run=tmp_path/"A",
        official_source=tmp_path/"source",python=Path(sys.executable),pretrained=tmp_path/"weights",
        gpu_uuid="GPU-synthetic",execute=False,wait_for_A=False)
    result=run.plan(args)
    assert result["A_dependency"]["ready"] is False and not args.output.exists()
    assert result["steps_per_variant"]==19200
    with pytest.raises(ValueError): run.execute(args,result)
    assert not args.output.exists()


def test_wait_stops_on_dependency_failure_no_gpu_claim(tmp_path,monkeypatch):
    args=argparse.Namespace(a_run=tmp_path/"A",wait_for_A=True,wait_hours=1)
    states=iter([{"ready":False,"phase":"train_cache"}])
    def dependency(path):
        try: return next(states)
        except StopIteration: raise ValueError("synthetic A failure")
    monkeypatch.setattr(run,"dependency_state",dependency)
    monkeypatch.setattr(run.time,"sleep",lambda delay: None)
    monkeypatch.setattr(run,"available_gpu",lambda: pytest.fail("waiting must not claim GPU"))
    with pytest.raises(ValueError): run.await_dependency(args,tmp_path)
    status=json.loads((tmp_path/"status.json").read_text())
    assert status["GPU_claimed"] is False and not status["training_completed"]
