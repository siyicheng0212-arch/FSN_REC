"""One sealed stage. Private clip identities, plans and weights never leave output."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F

from experiments.direction_gate.worker import base_model, native_outputs, simple_metrics, retention, gradient_norms
from experiments.paper_validation import legacy, perturb
from experiments.relation.data import read_jsonl, sha256_file
from experiments.relation.evaluate_suite import _paired
from experiments.relation.tcn_baseline import TCNConfig, Segment, collate_segments
from experiments.videomae_tcn.io import atomic_json, write_json
from experiments.videomae_tcn.metrics import summarize
from experiments.videomae_tcn.worker import _torch_save
from .data import anchor_batch, plan_hash, sampling_plan, train_replacements
from .model import AUGMENTED, AnchorReferenceTCN
from .run_suite import CONFIG, evidence, unchanged


def model(p,variant,device):
    return AnchorReferenceTCN(TCNConfig(**p["evidence"]["model_config"]),variant,seed=42,
        projection_dim=CONFIG["projection_dim"],gate_hidden_dim=CONFIG["gate_hidden_dim"]).to(device)


def check_signal(data, logits):
    y=data["labels"]; pred=logits.argmax(-1); a=data["a"].argmax(-1)
    return {"count":len(y),"accuracy":float((pred==y).float().mean()),
        "A_accuracy":float((a==y).float().mean()),"A_vs_model_disagreements":int((a!=pred).sum()),
        "A_correct_model_wrong":int(((a==y)&(pred!=y)).sum()),
        "A_wrong_model_correct":int(((a!=y)&(pred==y)).sum())}


def audit(p,out,segments,metadata):
    info=p["evidence"]
    if p["backbone"]=="adafocus":
        old=legacy.historical_model(p["historical_run"],torch.device("cpu"))
        historical_path=Path(p["historical_run"])/"tcn/private_val_predictions.jsonl"
    else:
        old=base_model({"base_run":p["historical_run"],"base_result":json.loads(
            (Path(p["historical_run"])/"tcn/result.json").read_text())},info,torch.device("cpu"))
        historical_path=Path(p["historical_run"])/"evaluation/private_val_predictions.jsonl"
    data={s:native_outputs(old,segments[s],torch.device("cpu")) for s in ("train","val")}
    legacy.exact_replay(data["val"],read_jsonl(historical_path))
    if not math.isclose(simple_metrics(data["val"],data["val"]["t"])["macro_f1"],
                        p["historical_TCN_F1_reference_only"],abs_tol=1e-12):
        raise RuntimeError("historical best-F1 replay mismatch")
    train=train_replacements(segments["train"],metadata)
    sampling=sampling_plan(len(segments["train"]),len(train["rows"]),CONFIG["epochs"])
    validation=perturb.plans(segments["val"],metadata)
    fixed={"train_replacements":train,"sampling":sampling,"validation":validation}
    write_json(out/"private_plans.json",fixed)
    _torch_save(out/"private_frozen_scores.pt",data)
    write_json(out/"audit_seal.json",{"plans_file_sha256":sha256_file(out/"private_plans.json"),
        "scores_sha256":sha256_file(out/"private_frozen_scores.pt"),
        "sampling_hash":plan_hash(sampling),"train_replacement_hash":plan_hash(train),
        "validation_challenge_hash":plan_hash(validation)})
    summary={"historical_823_prediction_replay_exact":True,"test_read":False,
        "train_historical_signal":check_signal(data["train"],data["train"]["t"]),
        "val_historical_signal":check_signal(data["val"],data["val"]["t"]),
        "in_sample_A_not_cross_fitted":True,"new_native_classifier_not_historical_TCN_reproduction":True,
        "train_anchor_views":len(train["rows"]),"train_view_exclusions":train["excluded"]}
    write_json(out/"diagnostic_summary.json",summary)
    print(json.dumps(summary),flush=True)


def cache(out):
    seal=json.loads((out/"audit_seal.json").read_text())
    if sha256_file(out/"private_plans.json")!=seal["plans_file_sha256"] or \
            sha256_file(out/"private_frozen_scores.pt")!=seal["scores_sha256"]:
        raise RuntimeError("shared audited plans or scores changed")
    fixed=json.loads((out/"private_plans.json").read_text())
    if any(plan_hash(fixed[k])!=seal[h] for k,h in (
        ("sampling","sampling_hash"),("train_replacements","train_replacement_hash"),("validation","validation_challenge_hash"))):
        raise RuntimeError("shared view schedule changed")
    return fixed,torch.load(out/"private_frozen_scores.pt",map_location="cpu",weights_only=True),seal


def loss_on_batch(m,batch,device,positions=None):
    f,a,y,valid,_=batch
    f,a,y,valid=[t.to(device) for t in (f,a,y,valid)]
    z=m(f,a,valid)
    if positions is None:
        if bool((y[valid]<0).any()):raise ValueError("normal loss cannot supervise donor/ignored positions")
        return F.cross_entropy(z[valid],y[valid])
    positions=torch.tensor(positions,device=device);rows=torch.arange(len(positions),device=device)
    if int((y>=0).sum())!=len(positions) or bool((y[rows,positions]<0).any()):
        raise ValueError("anchor view must supervise only its preserved target")
    return F.cross_entropy(z[rows,positions],y[rows,positions])


def native_oracle(m,batch,device):
    f,_,_,valid,_=batch;f=f.to(device);valid=valid.to(device)
    h=m.base.stem(f).transpose(1,2)*valid[:,None]
    for block in m.base.blocks:h=block(h,valid)
    return m.base.classifier(h).transpose(1,2)*valid[...,None]


def smoke(p,out,segments,fixed,variant,device):
    directory=out/"smoke";directory.mkdir(mode=0o700,exist_ok=True)
    batch=collate_segments([s for s in segments["train"] if len(s.clip_ids)>1][:4])
    if len(batch[4])!=4:raise RuntimeError("four real multiclips are required for CUDA smoke")
    m=model(p,variant,device).eval()
    with torch.no_grad():
        initial=m(batch[0].to(device),batch[1].to(device),batch[3].to(device))
        torch.testing.assert_close(initial,native_oracle(m,batch,device),atol=2e-5,rtol=2e-5)
    optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
    ab,positions=anchor_batch(segments["train"],fixed["train_replacements"],range(4),variant in AUGMENTED)
    records=[];m.train()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(3):
        torch.manual_seed(42+step);torch.cuda.manual_seed_all(42+step)
        before={n:w.detach().clone() for n,w in m.named_parameters()}
        optimizer.zero_grad(set_to_none=True)
        loss=.5*(loss_on_batch(m,batch,device)+loss_on_batch(m,ab,device,positions))
        if not bool(torch.isfinite(loss)):raise RuntimeError("nonfinite CUDA smoke loss")
        loss.backward();gradients=gradient_norms(m)
        if not gradients or max(gradients.values())<=0:raise RuntimeError("missing CUDA gradients")
        if variant.startswith("reference") and gradients.get("reference.score.2.weight",0)<=0:
            raise RuntimeError("reference final layer received no real-feature gradient")
        if variant.startswith("reference") and step>=1 and max(
                g for n,g in gradients.items() if n.startswith("reference.project."))<=0:
            raise RuntimeError("reference projection did not learn after the first update")
        if variant=="scalar_aug" and gradients.get("scalar",0)<=0:
            raise RuntimeError("shared scalar has no gradient")
        torch.nn.utils.clip_grad_norm_(m.parameters(),5);optimizer.step()
        updates={n:float((w.detach()-before[n]).abs().max()) for n,w in m.named_parameters()}
        if not updates or max(updates.values())<=0:raise RuntimeError("CUDA smoke did not update weights")
        if variant.startswith("reference") and updates["reference.score.2.weight"]<=0:
            raise RuntimeError("reference final layer did not actually update")
        records.append({"step":step+1,"loss":float(loss.detach()),"gradients":gradients,"updates":updates})
    m.eval()
    single=collate_segments([Segment([s.clip_ids[0]],s.features[:1],s.a_logits[:1],s.labels[:1],s.source_kinds[:1]) for s in batch[4]])
    with torch.no_grad():
        z=m(single[0].to(device),single[1].to(device),single[3].to(device))
        torch.testing.assert_close(z,native_oracle(m,single,device),atol=2e-5,rtol=2e-5)
    write_json(directory/(variant+".json"),{"all_passed":True,"steps":records,"variant":variant,
        "precision":"float32_native_TCN","A_loaded_or_trained":False,
        "singleton_retains_native_center_classifier_not_forced_A":True,
        "initial_unit_coefficients_reproduce_native_plain_same_weights":True,
        "torch":str(torch.__version__),"CUDA":torch.version.cuda,"gpu":torch.cuda.get_device_name(device),
        "gpu_uuid":os.environ["CUDA_VISIBLE_DEVICES"],"peak_allocated_bytes":torch.cuda.max_memory_allocated(device)})


@torch.no_grad()
def infer(m,segments,device,mode="learned",overrides=None,details=False):
    m.eval();logits=[];coefficients={};masks={};offset=0
    for start in range(0,len(segments),16):
        batch=collate_segments(segments[start:start+16]);nclip=sum(len(s.clip_ids) for s in batch[4])
        override=None
        if overrides is not None:
            override={}
            for k,flat in overrides.items():
                mat=torch.ones(batch[3].shape)
                j=offset
                for i,s in enumerate(batch[4]):
                    mat[i,:len(s.clip_ids)]=flat[j:j+len(s.clip_ids)];j+=len(s.clip_ids)
                override[k]=mat.to(device)
        result=m(batch[0].to(device),batch[1].to(device),batch[3].to(device),
            mode=mode,overrides=override,return_details=details)
        z,d=result if details else (result,None)
        for i,s in enumerate(batch[4]):
            n=len(s.clip_ids);logits.append(z[i,:n].cpu())
            if details:
                for k,v in d["coefficients"].items():
                    coefficients.setdefault(k,[]).append(v[i,:n].cpu())
                    masks.setdefault(k,[]).append(d["edge_masks"][k][i,:n].cpu())
        offset+=nclip
    return torch.cat(logits),{k:torch.cat(v) for k,v in coefficients.items()},{k:torch.cat(v) for k,v in masks.items()}


def train(p,out,segments,fixed,data,seal,variant,device):
    folder=out/variant;folder.mkdir(mode=0o700,exist_ok=False)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    m=model(p,variant,device)
    opt=torch.optim.AdamW(m.parameters(),lr=CONFIG["learning_rate"],weight_decay=CONFIG["weight_decay"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=CONFIG["epochs"],eta_min=CONFIG["minimum_learning_rate"])
    history=[];best=-math.inf;steps=0;started=time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    initial,_,_=infer(m,segments["train"],device)
    write_json(folder/"initial_train_signal.json",check_signal(data["train"],initial))
    for epoch in range(CONFIG["epochs"]):
        unchanged(p);m.train();epoch_start=time.monotonic()
        order=fixed["sampling"]["normal_orders"][epoch]
        batches=[order[k:k+16] for k in range(0,len(order),16)]
        anchors=np.array_split(fixed["sampling"]["anchor_orders"][epoch],len(batches))
        if any(len(a)==0 for a in anchors):raise RuntimeError("not enough paired anchor views for fixed batch schedule")
        normal_sum=anchor_sum=0.;normal_count=anchor_count=0;grad_max={}
        for b,indices in enumerate(batches):
            batch=collate_segments([segments["train"][i] for i in indices])
            ab,positions=anchor_batch(segments["train"],fixed["train_replacements"],anchors[b],variant in AUGMENTED)
            seed=42+epoch*len(batches)+b
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            opt.zero_grad(set_to_none=True)
            normal=loss_on_batch(m,batch,device);anchor=loss_on_batch(m,ab,device,positions)
            loss=.5*(normal+anchor)
            if not bool(torch.isfinite(loss)):raise RuntimeError("nonfinite formal classification loss")
            loss.backward()
            for n,g in gradient_norms(m).items():grad_max[n]=max(grad_max.get(n,0),g)
            torch.nn.utils.clip_grad_norm_(m.parameters(),CONFIG["clip_gradient_norm"]);opt.step();steps+=1
            nv,av=int(batch[3].sum()),len(positions)
            normal_sum+=float(normal.detach())*nv;anchor_sum+=float(anchor.detach())*av
            normal_count+=nv;anchor_count+=av
        if normal_count!=7372 or anchor_count!=len(fixed["train_replacements"]["rows"]) or not grad_max:
            raise RuntimeError("common training coverage or gradients incomplete")
        z,_,_=infer(m,segments["val"],device);metrics=simple_metrics(data["val"],z)
        row={"epoch":epoch+1,"phase":"native_anchor_reference","cumulative_steps":steps,
            "train_loss":.5*(normal_sum/normal_count+anchor_sum/anchor_count),
            "normal_CE":normal_sum/normal_count,"anchor_CE":anchor_sum/anchor_count,
            "normal_clip_visits":normal_count,"anchor_visits":anchor_count,
            "val_macro_f1":metrics["macro_f1"],"val_accuracy":metrics["accuracy"],
            "learning_rate":opt.param_groups[0]["lr"],"gradients_max":grad_max,
            "seconds":time.monotonic()-epoch_start}
        if epoch in (0,9,29,59):
            t,_,_=infer(m,segments["train"],device);row["native_train_signal"]=check_signal(data["train"],t)
        history.append(row)
        if metrics["macro_f1"]>best+1e-12:
            best=metrics["macro_f1"]
            _torch_save(folder/"best.pt",{"schema":CONFIG["schema"],"source_commit":p["source_commit"],
                "variant":variant,"backbone":p["backbone"],"model":m.state_dict(),"epoch":epoch+1,
                "macro_f1":best,"accuracy":metrics["accuracy"],"input_sha256":p["evidence"]["input_sha256"],
                "sampling_hash":seal["sampling_hash"],"train_replacement_hash":seal["train_replacement_hash"],
                "training":p["training"]})
        atomic_json(folder/"history.json",history);scheduler.step()
        print(json.dumps({k:row[k] for k in ("epoch","train_loss","val_macro_f1","val_accuracy","learning_rate")}|{"variant":variant,"best":best}),flush=True)
    ck=torch.load(folder/"best.pt",map_location="cpu",weights_only=True)
    write_json(folder/"result.json",{"training_completed":True,"test_read":False,"variant":variant,
        "epochs_completed":len(history),"optimizer_steps":steps,"best_epoch":ck["epoch"],
        "best_macro_f1":ck["macro_f1"],"best_accuracy_same_F1_checkpoint":ck["accuracy"],
        "best_sha256":sha256_file(folder/"best.pt"),"fresh_native_classifier":True,
        "historical_checkpoint_used_as_initialization":False,"seed":42,**m.parameter_counts(),
        "sampling_hash":seal["sampling_hash"],"train_replacement_hash":seal["train_replacement_hash"],
        "resources":{"elapsed_seconds":time.monotonic()-started,"peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved(device)},"receptive_field_clips":15})


def distribution(values):
    if not len(values):return {"count":0,"mean":None,"std":None,"quantiles":[]}
    return {"count":len(values),"mean":float(values.mean()),"std":float(values.std(unbiased=False)),
            "quantiles":torch.quantile(values,torch.tensor([0.,.1,.5,.9,1.])).tolist()}


def public_safe(obj):
    raw=json.dumps(obj,ensure_ascii=False,allow_nan=False)
    if any(k in raw for k in ('"clip_id"','"recording_key"','"feature_path"','/root/','/Users/')):
        raise RuntimeError("private paths or identifiers leaked into public aggregate")
    return obj


def evaluate(p,out,segments,metadata,fixed,data,seal,device):
    folder=out/"evaluation";folder.mkdir(mode=0o700,exist_ok=False)
    native={};report={"schema":CONFIG["schema"],"source_commit":p["source_commit"],"backbone":p["backbone"],
        "training_completed":True,"counts":p["evidence"]["counts"],"training":p["training"],
        "validation_role":p["validation_role"],"test_read":False,"A_frozen":True,
        "historical_results_reference_only_not_matched_budget":True,"exploratory_not_confirmatory":True,
        "A":simple_metrics(data["val"],data["val"]["a"]),"historical_TCN":simple_metrics(data["val"],data["val"]["t"]),
        "models":{},"synthetic_interventions_not_verified_edit_boundaries":True,
        "in_sample_A_logits_not_honest_out_of_fold_predictions":True,
        "train_replacement_views":len(fixed["train_replacements"]["rows"]),
        "sampling_hash":seal["sampling_hash"],"train_replacement_hash":seal["train_replacement_hash"]}
    source_kinds=np.array([k for s in segments["val"] for k in s.source_kinds])
    y=data["val"]["labels"].numpy();a=data["val"]["a"].argmax(-1).numpy()
    for v in p["variants"]:
        ck=torch.load(out/v/"best.pt",map_location="cpu",weights_only=True)
        r=json.loads((out/v/"result.json").read_text())
        expected_steps=CONFIG["epochs"]*math.ceil(len(segments["train"])/16)
        if not r["training_completed"] or r["epochs_completed"]!=CONFIG["epochs"] or r["optimizer_steps"]!=expected_steps \
                or ck["source_commit"]!=p["source_commit"] or ck["input_sha256"]!=p["evidence"]["input_sha256"] \
                or ck["training"]!=p["training"] or ck["variant"]!=v or ck["backbone"]!=p["backbone"] \
                or ck["sampling_hash"]!=seal["sampling_hash"] or ck["train_replacement_hash"]!=seal["train_replacement_hash"] \
                or sha256_file(out/v/"best.pt")!=r["best_sha256"]:
            raise RuntimeError("trained checkpoint seal mismatch")
        m=model(p,v,device);m.load_state_dict(ck["model"],strict=True)
        z,coeff,masks=infer(m,segments["val"],device,details=True)
        metric=simple_metrics(data["val"],z)
        if not math.isclose(metric["macro_f1"],ck["macro_f1"],abs_tol=1e-12) or \
                not math.isclose(metric["accuracy"],ck["accuracy"],abs_tol=1e-12):
            raise RuntimeError("selected best checkpoint metrics do not replay")
        native[v]=z
        rows=[{"clip_id":c,"label_id":int(y[i]),"a_prediction":int(a[i]),"tcn_prediction":int(z[i].argmax())}
              for i,c in enumerate(data["val"]["clip_ids"])]
        with (folder/(v+"_private_predictions.jsonl")).open("x") as stream:
            for row in rows:stream.write(json.dumps(row)+"\n")
        agg=summarize(rows,metadata,segments["val"])
        # The generic historical metrics helper has VideoMAE display names; do
        # not mislabel the old AdaFocus backbone in an exported report.
        agg=json.loads(json.dumps(agg).replace('"VideoMAE_only"','"A_only"').replace('"VideoMAE_TCN"','"temporal_model"'))
        centered,_,_=infer(m,segments["val"],device,mode="zero")
        flat=[Segment([c],s.features[j:j+1],s.a_logits[j:j+1],s.labels[j:j+1],[s.source_kinds[j]])
              for s in segments["val"] for j,c in enumerate(s.clip_ids)]
        singleton=perturb.infer(m,flat,device)
        torch.testing.assert_close(centered,singleton,atol=2e-5,rtol=2e-5)
        interventions={"no_neighbor":{"metrics":simple_metrics(data["val"],centered),"eligible_count":823}}
        for name in ("shuffle","cross_clinical"):
            changed,eligible=perturb.intervene(m,segments["val"],z,fixed["validation"][name],device)
            subset={"labels":data["val"]["labels"][eligible],"a":data["val"]["a"][eligible]}
            interventions[name]={"overall_metrics":simple_metrics(data["val"],changed),"eligible_count":len(eligible),
                "eligible_metrics":simple_metrics(subset,changed[eligible]),
                "changed_prediction_count":int((changed.argmax(-1)!=z.argmax(-1)).sum())}
        entry={"training_result":r,"metrics":metric,"aggregate":agg,"interventions":interventions,
            "paired_vs_historical_TCN":_paired(y,z.argmax(-1).numpy(),data["val"]["t"].argmax(-1).numpy()),
            "retention_of_historical_corrections":retention(data["val"]["labels"],data["val"]["a"],data["val"]["t"],z),
            "coefficients_by_layer_signed_span":{k:distribution(coeff[k][masks[k]]) for k in coeff}}
        if v.startswith("reference") or v=="scalar_aug":
            unit,_,_=infer(m,segments["val"],device,mode="unit")
            entry["same_checkpoint_g1"]={"metrics":simple_metrics(data["val"],unit),
                "changed_predictions":int((unit.argmax(-1)!=z.argmax(-1)).sum()),
                "maximum_logit_difference":float((unit-z).abs().max())}
        if v.startswith("reference"):
            rng=np.random.default_rng(42);permuted={};moved=changed_coeff=0
            for k,values in coeff.items():
                values=values.clone()
                for kind in ("clinical","network"):
                    positions=np.flatnonzero(masks[k].numpy() & (source_kinds==kind))
                    if len(positions)>1:
                        donor=np.roll(positions,int(rng.integers(1,len(positions)))).copy()
                        values[positions]=coeff[k][donor];moved+=len(positions)
                changed_coeff+=int(((values-coeff[k]).abs()>1e-8)[masks[k]].sum())
                permuted[k]=values
            perm,_,_=infer(m,segments["val"],device,overrides=permuted)
            entry["coefficient_permutation"]={"seed":42,"grouping":"signed_span_and_audit_source_kind_only",
                "moved_positions":moved,"actually_changed_coefficients":changed_coeff,
                "effective_control":changed_coeff>0,"metrics":simple_metrics(data["val"],perm),
                "changed_predictions":int((perm.argmax(-1)!=z.argmax(-1)).sum())}
        report["models"][v]=entry
    for v,entry in report["models"].items():
        for control in ("plain","plain_aug","scalar_aug"):
            entry["paired_vs_"+control]=_paired(y,native[v].argmax(-1).numpy(),native[control].argmax(-1).numpy())
            entry["F1_change_vs_"+control]=entry["metrics"]["macro_f1"]-report["models"][control]["metrics"]["macro_f1"]
        entry["retention_of_same_aug_plain_corrections"]=retention(
            data["val"]["labels"],data["val"]["a"],native["plain_aug"],native[v])
    target=report["models"]["reference_aug"];ret=target["retention_of_same_aug_plain_corrections"]
    fixes=ret["old_TCN_fixes"]
    report["prespecified_exploratory_criteria"]={
        "beats_both_same_aug_plain_and_scalar":all(target["F1_change_vs_"+c]>1e-12 for c in ("plain_aug","scalar_aug")),
        "no_more_A_correct_to_wrong_than_same_aug_plain":target["metrics"]["paired_vs_A"]["worsened"]<=report["models"]["plain_aug"]["metrics"]["paired_vs_A"]["worsened"],
        "retains_at_least_90_percent_of_plain_aug_A_corrections":None if fixes==0 else ret["retained_fixes"]/fixes>=.9,
        "not_a_paper_acceptance_or_generalization_claim":True,
        "capacity_matching_and_independent_test_still_needed":True}
    write_json(folder/"public_safe_summary.json",public_safe(report))
    print(json.dumps({v:r["metrics"]["macro_f1"] for v,r in report["models"].items()}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("audit","smoke","train","evaluate"),required=True)
    parser.add_argument("--run-protocol",required=True);parser.add_argument("--variant")
    args=parser.parse_args();p=json.loads(Path(args.run_protocol).read_text());unchanged(p)
    torch.set_num_threads(2)
    if args.stage=="audit":device=torch.device("cpu")
    else:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not os.environ.get("CUDA_VISIBLE_DEVICES","").startswith("GPU-"):
            raise RuntimeError("worker requires one actual CUDA GPU selected by UUID, not DDP")
        device=torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(p["output"])
    segments,metadata,info=evidence(p["backbone"],p["index"],p["source_protocol_dir"],p["historical_run"])
    if info!=p["evidence"]:raise RuntimeError("sealed frozen evidence changed")
    if args.stage=="audit":audit(p,out,segments,metadata);return
    fixed,data,seal=cache(out)
    if args.stage in ("train","smoke") and args.variant not in p["variants"]:
        raise ValueError("explicit registered variant required")
    if args.stage=="smoke":smoke(p,out,segments,fixed,args.variant,device)
    elif args.stage=="train":train(p,out,segments,fixed,data,seal,args.variant,device)
    else:evaluate(p,out,segments,metadata,fixed,data,seal,device)


if __name__=="__main__":main()
