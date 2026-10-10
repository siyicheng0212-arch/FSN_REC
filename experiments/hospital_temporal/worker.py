"""Common fixed-budget T/T′ smoke, training and paired context diagnostics."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.hospital_holdout.prepare import encoded, sha
from experiments.metrics import compute_classification_metrics
from experiments.modern_baselines.train_videomae2 import _atomic_json, _atomic_torch_save
from experiments.relation.evaluate_suite import _paired, _directions, classification_metrics
from experiments.relation.tcn_baseline import Segment, collate_segments
from .features import load
from .model import CONFIG, VARIANTS, TemporalVariant
from .prepare import build_chains


def make_segments(matrix, labels, records):
    lookup = {r["clip_id"]: i for i,r in enumerate(records)}
    _, _, partition, _ = build_chains(records)
    segments = {}
    for split, chains in partition.items():
        segments[split] = []
        for ids in chains:
            indices = [lookup[clip] for clip in ids]
            x = matrix[indices].numpy()
            kinds = ["clinical" if records[i]["source_collection"].casefold() in {"lishui","menzhen"}
                     else "network" for i in indices]
            segments[split].append(Segment(ids, x, x[:,-7:], labels[indices].numpy(), kinds))
    return segments


def sampling_plan(count):
    rng = random.Random(CONFIG["seed"])
    return [rng.sample(range(count), count) for _ in range(CONFIG["training"]["epochs"])]


def batches(segments, order=None):
    order = list(range(len(segments))) if order is None else order
    size = CONFIG["training"]["batch_segments"]
    for start in range(0,len(order),size):
        yield collate_segments([segments[i] for i in order[start:start+size]])


def gradients(model):
    result = {}
    for family in ("stem","blocks","classifier"):
        total = 0.0
        for name, parameter in model.base.named_parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("nonfinite temporal gradient")
                if name.startswith(family+"."):
                    total += float(parameter.grad.detach().square().sum())
        result[family] = math.sqrt(total)
    return result


def loss_forward(model, batch, device):
    x, a, y, valid, _ = batch
    x,a,y,valid = (value.to(device) for value in (x,a,y,valid))
    logits = model(x,a,valid)
    loss = F.cross_entropy(logits[valid],y[valid])
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite temporal loss")
    return loss, int(valid.sum())


@torch.no_grad()
def predictions(model, segments, records, device, mode="native"):
    model.eval()
    lookup = {row["clip_id"]: i for i,row in enumerate(records)}
    out = torch.empty(len(records),7)
    seen = set()
    for batch in batches(segments):
        x,a,_,valid,items = batch
        values = model(x.to(device),a.to(device),valid.to(device),mode=mode).cpu()
        for i,segment in enumerate(items):
            indices = [lookup[clip] for clip in segment.clip_ids]
            if seen.intersection(indices):
                raise ValueError("prediction coverage contains duplicates")
            out[indices] = values[i,:len(indices)]
            seen.update(indices)
    if seen != set(range(len(records))):
        raise ValueError("prediction coverage is incomplete")
    return out


def summary(scores, records, a_scores):
    labels = np.asarray([r["label_id"] for r in records])
    predicted, a_pred = scores.argmax(-1).numpy(), a_scores.argmax(-1).numpy()
    metrics = compute_classification_metrics(scores,labels,records)
    return {"metrics": metrics, "paired_vs_A": _paired(labels,predicted,a_pred),
            "sweep_reperfusion": _directions(labels,predicted)}


def smoke(segments, variant, device):
    torch.manual_seed(42)
    model = TemporalVariant(variant).to(device)
    # Include real multi-clip evidence: singleton-only smoke cannot test context.
    chosen = sorted(segments,key=lambda s: -len(s.clip_ids))[:16]
    batch = collate_segments(chosen)
    model.eval()
    with torch.no_grad():
        initial = model(batch[0].to(device),batch[1].to(device),batch[3].to(device))
    if not torch.equal(initial,batch[1].to(device)):
        raise ValueError("initial T/T′ logits do not exactly equal frozen A")
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)
    model.train()
    steps = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss,_ = loss_forward(model,batch,device)
        loss.backward()
        norms = gradients(model)
        before = model.base.classifier.weight.detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
        optimizer.step()
        update = float((model.base.classifier.weight-before).norm())
        if norms["classifier"]<=0 or update<=0 or step and min(norms.values())<=0:
            raise ValueError("finite gradients and real head/stem/block updates are required")
        steps.append({"step":step+1,"loss":float(loss.detach()),"gradients":norms,"classifier_update":update})
    return {"all_passed":True,"variant":variant,"steps":steps,"A_in_optimizer":False,
            "initial_exact_A":True,"precision":"float32","disposable_smoke_not_training_result":True}


def train(segments, records, variant, output, device, identity):
    torch.manual_seed(42)
    if device.type == "cuda": torch.cuda.manual_seed_all(42); torch.cuda.reset_peak_memory_stats(device)
    model = TemporalVariant(variant).to(device)
    cfg = CONFIG["training"]
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    plan = sampling_plan(len(segments["train"]))
    (output/"private_sampler_plan.json").write_bytes(encoded(plan))
    best, best_epoch, history, total_steps = -1.0,None,[],0
    started = time.perf_counter()
    for epoch, order in enumerate(plan,1):
        model.train()
        cumulative, clips, epoch_steps = 0.0,0,0
        peak_grad = {k:0.0 for k in ("stem","blocks","classifier")}
        for step,batch in enumerate(batches(segments["train"],order)):
            # Common-shape T′ models receive the same dropout random schedule.
            torch.manual_seed(1000042 + (epoch-1)*len(order) + step)
            if device.type == "cuda": torch.cuda.manual_seed_all(1000042+(epoch-1)*len(order)+step)
            optimizer.zero_grad(set_to_none=True)
            loss,count = loss_forward(model,batch,device)
            loss.backward()
            norms = gradients(model)
            for k,v in norms.items(): peak_grad[k] = max(peak_grad[k],v)
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["clip_grad"],error_if_nonfinite=True)
            optimizer.step()
            epoch_steps+=1; total_steps+=1; clips+=count; cumulative+=float(loss.detach())*count
        if clips != len(records["train"]) or epoch_steps != math.ceil(len(order)/cfg["batch_segments"]) \
                or min(peak_grad.values())<=0:
            raise ValueError("actual visit/update count or gradient flow differs from protocol")
        val_scores = predictions(model,segments["val"],records["val"],device)
        metric = compute_classification_metrics(val_scores,[r["label_id"] for r in records["val"]],records["val"])
        if metric["all"]["macro_f1"]>best:
            best,best_epoch = metric["all"]["macro_f1"],epoch
            _atomic_torch_save(output/"best.pt",{"model":model.state_dict(),"variant":variant,
                "epoch":epoch,"metrics":metric,"identity":identity,"config":CONFIG})
        row = {"epoch":epoch,"train_loss":cumulative/clips,"optimizer_steps":epoch_steps,
            "total_steps":total_steps,"visited_clips":clips,"gradient_max":peak_grad,
            "val_macro_f1":metric["all"]["macro_f1"],"val_accuracy":metric["all"]["accuracy"],
            "best_epoch":best_epoch,"best_macro_f1":best,"seconds":time.perf_counter()-started}
        history.append(row)
        _atomic_json(output/"history.json",history)
        _atomic_torch_save(output/"last.pt",{"model":model.state_dict(),"optimizer":optimizer.state_dict(),
            "epochs_completed":epoch,"steps":total_steps,"identity":identity,"config":CONFIG})
        _atomic_json(output/"status.json",{"status":"running","phase":"temporal_train","epoch":epoch,
            "training_completed":False,"variant":variant})
        print(json.dumps({"variant":variant,**row}),flush=True)
    result = {"status":"complete","training_completed":True,"variant":variant,
        "epochs_completed":len(history),"optimizer_steps":total_steps,"best_epoch":best_epoch,
        "best_macro_f1":best,"best_checkpoint_sha256":sha(output/"best.pt"),
        "sampler_sha256":sha(output/"private_sampler_plan.json"),"parameters":model.parameter_counts(),
        "seconds":time.perf_counter()-started,"peak_cuda_bytes":torch.cuda.max_memory_allocated(device)
        if device.type=="cuda" else None,"identity":identity,"config":CONFIG}
    _atomic_json(output/"result.json",result)
    return result


@torch.no_grad()
def challenge(model, segments, records, device, condition):
    if condition not in {"shuffle_neighbors","cross_record_neighbors"}:
        raise ValueError("unregistered challenge")
    rng = random.Random(CONFIG["challenge_seed"])
    lookup = {r["clip_id"]:i for i,r in enumerate(records)}
    all_x = {clip:segment.features[i] for segment in segments for i,clip in enumerate(segment.clip_ids)}
    scores,changed = torch.empty(len(records),7),0
    model.eval()
    for segment in segments:
        for anchor,clip in enumerate(segment.clip_ids):
            neighbors = [i for i in range(len(segment.clip_ids)) if i!=anchor]
            vector = segment.features.copy()
            if condition=="shuffle_neighbors":
                donors = neighbors.copy(); rng.shuffle(donors)
                for target,donor in zip(neighbors,donors): vector[target] = segment.features[donor]
            else:
                anchor_record = records[lookup[clip]]["record_id"]
                choices = [r["clip_id"] for r in records if r["record_id"]!=anchor_record]
                if neighbors and not choices: raise ValueError("cross-record challenge has no independent recording donor")
                for target in neighbors: vector[target] = all_x[rng.choice(choices)]
            # Replace visual feature and raw A scores together, never the anchor.
            if not np.array_equal(vector[anchor],segment.features[anchor]):
                raise AssertionError("challenge changed the anchor")
            changed += sum(not np.array_equal(vector[i],segment.features[i]) for i in neighbors)
            x = torch.from_numpy(vector)[None].to(device)
            valid = torch.ones(1,len(vector),dtype=torch.bool,device=device)
            scores[lookup[clip]] = model(x,x[...,-7:],valid)[0,anchor].cpu()
    return scores,changed


def evaluate_all(a_run, feature_dir, root, device, expected_commit=None):
    matrix,labels,records,seal = load(a_run,feature_dir)
    segments = make_segments(matrix,labels,records)
    val = [r for r in records if r["split"]=="val"]
    a_scores = matrix[[i for i,r in enumerate(records) if r["split"]=="val"],-7:]
    singleton = {s.clip_ids[0] for s in segments["val"] if len(s.clip_ids)==1}
    output = {"A":summary(a_scores,val,a_scores),"A_training_metrics_not_held_out":seal.get("A_metrics",{}).get("train"),
        "variants":{},"Menzhen_role":"development_validation",
        "independent_test_evaluated":False,"patient_independence_verified":False,"R_D_enabled":False,
        "A_checkpoint_sha256":seal["A_checkpoint_sha256"],"synthetic_challenges_are_not_real_cut_detection":True}
    ordinary, ordinary_metrics, prediction_bank = None, None, {}
    for variant in VARIANTS:
        d = root/variant
        result = json.loads((d/"result.json").read_text())
        if result.get("training_completed") is not True or result.get("epochs_completed")!=CONFIG["training"]["epochs"] \
                or result.get("best_checkpoint_sha256")!=sha(d/"best.pt"):
            raise ValueError("all six variants must complete before unified evaluation")
        payload = torch.load(d/"best.pt",map_location="cpu",weights_only=True)
        if payload.get("variant")!=variant or payload.get("config")!=CONFIG \
                or payload.get("identity")!=result["identity"] \
                or any(payload["identity"].get(key)!=seal[key] for key in
                       ("A_checkpoint_sha256","features_sha256","protocol_sha256")) \
                or result.get("optimizer_steps")!=CONFIG["training"]["epochs"] * math.ceil(
                    len(segments["train"])/CONFIG["training"]["batch_segments"]) \
                or payload.get("epoch")!=result.get("best_epoch"):
            raise ValueError("selected temporal checkpoint identity changed")
        if expected_commit is not None and payload["identity"].get("source_commit")!=expected_commit:
            raise ValueError("temporal checkpoints came from different code commits")
        model = TemporalVariant(variant).to(device)
        model.load_state_dict(payload["model"],strict=True)
        score = predictions(model,segments["val"],val,device)
        stats = summary(score,val,a_scores)
        if stats["metrics"]!=payload["metrics"]:
            raise ValueError("same-best-checkpoint development replay differs")
        predicted,apred,true = score.argmax(-1).numpy(),a_scores.argmax(-1).numpy(),np.array([r["label_id"] for r in val])
        if ordinary is None:
            ordinary, ordinary_metrics = predicted.copy(),stats["metrics"]
        stats["paired_vs_plain_TCN"] = _paired(true,predicted,ordinary)
        prediction_bank[variant] = predicted.copy()
        original_good = (apred!=true)&(ordinary==true)
        original_bad = (apred==true)&(ordinary!=true)
        stats["ordinary_correction_retention"] = {
            "ordinary_improved":int(original_good.sum()),"improvements_retained":int((original_good&(predicted==true)).sum()),
            "ordinary_worsened":int(original_bad.sum()),"harms_recovered":int((original_bad&(predicted==true)).sum())}
        stats["context_slices"] = {}
        for name,flag in (("isolated",True),("has_neighbors",False)):
            idx = [i for i,r in enumerate(val) if (r["clip_id"] in singleton)==flag]
            stats["context_slices"][name] = summary(score[idx],[val[i] for i in idx],a_scores[idx])
        if variant=="plain_tcn":
            protected = score.clone()
            ids = [i for i,r in enumerate(val) if r["clip_id"] in singleton]
            protected[ids] = a_scores[ids]
            output["same_plain_checkpoint_isolated_keep_A"] = summary(protected,val,a_scores)
        challenges = {"center_only":summary(predictions(model,segments["val"],val,device,"center_only"),val,a_scores)}
        for condition in ("shuffle_neighbors","cross_record_neighbors"):
            values,changed = challenge(model,segments["val"],val,device,condition)
            challenges[condition] = {**summary(values,val,a_scores),"actual_changed_neighbor_positions":changed}
        # Export only resource numbers, never private identity/path metadata.
        stats.update(challenges=challenges,resources={k:result[k] for k in (
            "best_epoch","epochs_completed","optimizer_steps","parameters","seconds","peak_cuda_bytes")})
        stats["per_class_change_vs_A"] = [{"class_id":i,"support":r["support"],
            "f1_change":r["f1"]-output["A"]["metrics"]["all"]["per_class"][i]["f1"],
            "macro_f1_contribution":(r["f1"]-output["A"]["metrics"]["all"]["per_class"][i]["f1"])/7}
            for i,r in enumerate(stats["metrics"]["all"]["per_class"])]
        stats["per_class_change_vs_plain_TCN"] = [{"class_id":i,
            "f1_change":r["f1"]-ordinary_metrics["all"]["per_class"][i]["f1"],
            "macro_f1_contribution":(r["f1"]-ordinary_metrics["all"]["per_class"][i]["f1"])/7}
            for i,r in enumerate(stats["metrics"]["all"]["per_class"])]
        grouping = defaultdict(list)
        for i,r in enumerate(val): grouping[r["record_id"]].append(i)
        stats["recording_aggregate"] = {"recordings":len(grouping),"not_patients":True,
            "mean_record_accuracy":float(np.mean([classification_metrics(true[ids],predicted[ids])["accuracy"]
                                                   for ids in grouping.values()]))}
        output["variants"][variant] = stats
    for variant, stats in output["variants"].items():
        stats["paired_vs_trained_protected_TCN"] = _paired(
            true,prediction_bank[variant],prediction_bank["protected_tcn"])
    if len({json.loads((root/v/"result.json").read_text())["sampler_sha256"] for v in VARIANTS})!=1:
        raise ValueError("variants did not share one fixed training sampling plan")
    _atomic_json(root/"public_safe_summary.json",output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("smoke","train","evaluate"),required=True)
    parser.add_argument("--a-run",type=Path,required=True)
    parser.add_argument("--features",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--variant",choices=VARIANTS)
    parser.add_argument("--expected-commit",required=True)
    args = parser.parse_args()
    from .run import verify_code
    verify_code(args.expected_commit)
    suite = args.output.resolve() if args.stage=="evaluate" else args.output.resolve().parent
    if not (suite/".launch_once").is_file() or args.features.resolve()!=suite/"features" \
            or suite==args.a_run.resolve() or args.a_run.resolve() in suite.parents:
        raise ValueError("worker writes must be within the fresh temporal suite, not A or arbitrary directories")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError("one independently mapped CUDA GPU required; not DDP")
    if args.stage=="evaluate":
        if (args.output/"public_safe_summary.json").exists() or not (args.output/".launch_once").is_file():
            raise ValueError("fresh suite evaluation required, no overwrite or standalone output")
        evaluate_all(args.a_run,args.features,args.output,device,args.expected_commit); return
    args.output.mkdir(parents=True,exist_ok=False)
    matrix,labels,records,seal = load(args.a_run,args.features)
    if {s:sum(r["split"]==s for r in records) for s in ("train","val")}!={"train":7307,"val":888}:
        raise ValueError("formal worker requires all new7307/888 clips")
    segments = make_segments(matrix,labels,records)
    if args.stage=="smoke":
        result = {v:smoke(segments["train"],v,device) for v in VARIANTS}
        _atomic_json(args.output/"cuda_smoke.json",{"all_passed":True,"variants":result})
        return
    if args.variant is None: raise ValueError("training requires one registered variant")
    identity = {"source_commit":args.expected_commit,"A_checkpoint_sha256":seal["A_checkpoint_sha256"],
        "features_sha256":seal["features_sha256"],"protocol_sha256":seal["protocol_sha256"]}
    split_records = {s:[r for r in records if r["split"]==s] for s in ("train","val")}
    try:
        train(segments,split_records,args.variant,args.output,device,identity)
    except BaseException as error:
        _atomic_json(args.output/"status.json",{"status":"failed","error_type":type(error).__name__,
                     "training_completed":False,"automatic_retry":False})
        raise


if __name__ == "__main__":
    main()


if __name__=="__main__":
    main()
