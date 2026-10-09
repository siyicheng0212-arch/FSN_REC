"""Strict best-checkpoint replay, stratification and fixed interventions."""
import argparse
import json
import math
from pathlib import Path
import statistics
import numpy as np
import torch
from experiments.context_evidence import worker as context
from experiments.direction_gate.worker import native_outputs, simple_metrics
from experiments.relation.data import read_jsonl, sha256_file
from experiments.relation.evaluate_suite import _paired
from experiments.relation.tcn_baseline import Segment
from experiments.videomae_tcn.evidence import prepare
from experiments.videomae_tcn.metrics import summarize
from experiments.videomae_tcn.io import write_json
from experiments.videomae_tcn.worker import _torch_save
from experiments.temporal_heads.worker import make_model as head_model
from . import legacy
from .perturb import infer, intervene, plans, singleton_fallback
from .run import unchanged


def safe_write(out, report):
    serialized=json.dumps(report)
    if any(s in serialized for s in ('"clip_id"','"recording_key"','/root/','/Users/','"feature_path"')):
        raise RuntimeError("private data in aggregate")
    write_json(out/"public_safe_summary.json",report)


def load_completed(entry,info,device):
    directory=Path(entry["directory"]);v=entry["variant"]
    p=json.loads((directory/"run_protocol.json").read_text())
    r=json.loads((directory/v/"result.json").read_text())
    h=json.loads((directory/v/"history.json").read_text())
    ck=torch.load(directory/v/"best.pt",map_location="cpu",weights_only=True)
    if not r["training_completed"] or r["epochs_completed"]!=30 or r["optimizer_steps"]!=5910 or len(h)!=30 or \
            sha256_file(directory/v/"best.pt")!=r["best_sha256"] or ck["source_commit"]!=p["source_commit"] or \
            ck["input_sha256"]!=info["input_sha256"] or ck["training"]!=p["training"] or ck["variant"]!=v or \
            ck["epoch"]!=r["best_epoch"] or abs(ck["macro_f1"]-r["best_macro_f1"])>1e-12 or p["evidence"]!=info:
        raise RuntimeError("strict completed checkpoint/protocol/history mismatch")
    if entry["kind"]=="context":
        if ck["base_sha256"]!=p["base_sha256"]:raise RuntimeError("base checkpoint changed")
        model=context.make_model(p,ck["statistics"],v,device)
    else:model=head_model(v,device)
    model.load_state_dict(ck["model"],strict=True);model.eval()
    return model,r,h


def private_rows(segments,logits):
    rows=[];i=0
    for s in segments:
        for j,c in enumerate(s.clip_ids):
            rows.append({"clip_id":c,"label_id":int(s.labels[j]),"a_prediction":int(s.a_logits[j].argmax()),"tcn_prediction":int(logits[i].argmax())});i+=1
    return rows


def condition_report(data,native,changed,eligible):
    result={"metrics":simple_metrics(data,changed),"eligible_anchor_count":len(eligible),
        "changed_predictions":int((changed.argmax(-1)!=native.argmax(-1)).sum()),
        "logit_mean_abs_difference":float((changed-native).abs().mean()),
        "logit_max_abs_difference":float((changed-native).abs().max())}
    if eligible:
        sub={k:data[k][eligible] for k in ("a","labels")}
        result["eligible_only_metrics"]=simple_metrics(sub,changed[eligible])
        result["eligible_native_metrics"]=simple_metrics(sub,native[eligible])
    return result


def audit(p,out,segments,metadata,device):
    plan=plans(segments["val"],metadata)
    write_json(out/"private_intervention_plan.json",plan)
    flat=[Segment([c],s.features[j:j+1],s.a_logits[j:j+1],s.labels[j:j+1],[s.source_kinds[j]])
          for s in segments["val"] for j,c in enumerate(s.clip_ids)]
    data={"a":torch.from_numpy(np.concatenate([s.a_logits for s in segments["val"]])),
          "labels":torch.from_numpy(np.concatenate([s.labels for s in segments["val"]]))}
    report={"validation_role":"val823_development_not_independent_test","test_read":False,
        "A_frozen":True,"counts":p["evidence"]["counts"],"source_commit":p["source_commit"],
        "independent_test_performed":False,"A":simple_metrics(data,data["a"]),"models":{},
        "perturbation_protocol":{"seed":42,"anchor_evidence_and_label_unchanged":True,
            "donor_feature_and_A_logits_paired":True,"no_action_labels_in_sampling":True,
            "shuffle_all_nonanchor_slots_deranged_within_chain":True,"cross_all_nonanchor_slots_different_clinical_record":True,
            "shuffle_eligible":len(plan["shuffle"]),"cross_eligible":len(plan["cross_clinical"]),
            "shuffle_excluded":plan["shuffle_excluded"],"cross_excluded":plan["cross_excluded"],
            "synthetic_not_real_edit_boundaries":True},
        "stability_scope":"fixed_A42_and_fixed_initial_TCN42_only","private_predictions_exported":False}
    predictions={}
    for entry in p["entries"]:
        model,result,history=load_completed(entry,p["evidence"],device)
        logits=infer(model,segments["val"],device)
        native=simple_metrics(data,logits)
        if abs(native["macro_f1"]-result["best_macro_f1"])>1e-12:raise RuntimeError("best F1 not reproduced")
        rows=private_rows(segments["val"],logits)
        if entry["kind"]=="context":
            historical=read_jsonl(Path(entry["directory"])/"evaluation"/(entry["variant"]+"_private_val_predictions.jsonl"))
            observed={r["clip_id"]:r for r in rows}
            if len(observed)!=823 or any(observed.get(r["clip_id"])!=r for r in historical) or len(historical)!=823:
                raise RuntimeError("paired seed per-clip historical prediction replay failed")
        center=infer(model,flat,device)
        interventions={"no_neighbors":condition_report(data,logits,center,list(range(823)))}
        challenged={}
        for name in ("shuffle","cross_clinical"):
            changed,eligible=intervene(model,segments["val"],logits,plan[name],device)
            interventions[name]=condition_report(data,logits,changed,eligible);challenged[name]=changed
        name=entry["name"];predictions[name]=logits
        report["models"][name]={"seed":entry["seed"],"metrics":native,"aggregate":summarize(rows,metadata,segments["val"]),
            "training_result":result,"interventions":interventions,"best_replay_exact":True,
            "gradient_records_finite":all(math.isfinite(float(x["train_loss"])) for x in history)}
        if entry["variant"]=="legacy_continue":
            protected=singleton_fallback(logits,data["a"],segments["val"])
            protected_rows=private_rows(segments["val"],protected)
            key=f"seed{entry['seed']}_ordinary_singleton_A"
            predictions[key]=protected
            report["models"][key]={"seed":entry["seed"],"metrics":simple_metrics(data,protected),
                "aggregate":summarize(protected_rows,metadata,segments["val"]),"no_retraining":True,
                "checkpoint_identical_to_ordinary":True,"singleton_exact_A":True,
                "interventions":{"no_neighbors":condition_report(data,protected,data["a"],list(range(823))),
                    **{n:condition_report(data,protected,singleton_fallback(z,data["a"],segments["val"]),
                        [sum(len(s.clip_ids) for s in segments["val"][:r["segment"]])+r["anchor"] for r in plan[n]]) for n,z in challenged.items()}}}
        print(json.dumps({"model":name,"F1":native["macro_f1"],"best_replay":True}),flush=True)
        del model
    for seed in (42,10,1217):
        names=[f"seed{seed}_legacy_continue",f"seed{seed}_ordinary_singleton_A",f"seed{seed}_context_delta_continue"]
        for name in names:
            y=data["labels"].numpy();z=predictions[name].argmax(-1).numpy()
            report["models"][name]["paired_vs_same_seed_ordinary"]=_paired(y,z,predictions[names[0]].argmax(-1).numpy())
            report["models"][name]["paired_vs_same_seed_singleton_A"]=_paired(y,z,predictions[names[1]].argmax(-1).numpy())
    report["three_seed_summary"]={}
    for suffix in ("legacy_continue","ordinary_singleton_A","context_delta_continue"):
        values=[report["models"][f"seed{s}_{suffix}"]["metrics"]["macro_f1"] for s in (42,10,1217)]
        report["three_seed_summary"][suffix]={"values":values,"mean":statistics.mean(values),"sample_std":statistics.stdev(values)}
    report["context_minus_singleton_protection_F1"]=[report["models"][f"seed{s}_context_delta_continue"]["metrics"]["macro_f1"]-
        report["models"][f"seed{s}_ordinary_singleton_A"]["metrics"]["macro_f1"] for s in (42,10,1217)]
    _torch_save(out/"private_replay_logits.pt",predictions)
    write_json(out/"audit_seal.json",{"interventions_sha256":sha256_file(out/"private_intervention_plan.json"),"logits_sha256":sha256_file(out/"private_replay_logits.pt")})
    safe_write(out,report)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stage",required=True);ap.add_argument("--run-protocol",required=True)
    a=ap.parse_args();p=json.loads(Path(a.run_protocol).read_text());unchanged(p)
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("one UUID-mapped CUDA device required")
    device=torch.device("cuda:0");out=Path(p["output"])
    if p["mode"]=="audit":
        segments,metadata,info=prepare(p["index"],p["source_protocol_dir"])
        if info!=p["evidence"]:raise RuntimeError("evidence changed")
        audit(p,out,segments,metadata,device)
    else:
        segments,metadata,info=legacy.prepare(p["index"],p["source_protocol_dir"],p["base_run"])
        if info!=p["evidence"]:raise RuntimeError("legacy evidence changed")
        if a.stage=="legacy_audit":legacy.audit(p,out,segments,metadata,device)
        else:
            data,plans_=context.cache(out)
            if a.stage=="legacy_smoke":legacy.smoke(p,out,segments,data,plans_,device)
            elif a.stage.startswith("train_"):context.train(p,out,segments,data,plans_,a.stage[6:],device)
            elif a.stage=="legacy_evaluate":
                context.evaluate(p,out,segments,metadata,data,plans_,device)
                report=json.loads((out/"evaluation/public_safe_summary.json").read_text())
                # Generic metric helper has historical VideoMAE display keys, not model inputs.
                text=json.dumps(report).replace('"VideoMAE_only"','"AdaFocus_A"').replace('"VideoMAE_TCN"','"AdaFocus_temporal"')
                report=json.loads(text);report["backbone"]="verified_historical_AdaFocus_A"
                safe_write(out,report)
            else:raise ValueError("unknown stage")
    unchanged(p)


if __name__=="__main__":main()
