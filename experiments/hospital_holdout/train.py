"""VideoMAEv2: train7307, Menzhen-dev888 best-F1 selection (not independent test).

Only pretrained-model loading and the already-tested optimization loop are
reused. Old 7372/823 datasets, validators, best selection and resume are not.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.hospital_holdout.prepare import sha
from experiments.hospital_holdout.video_data import HospitalDataset, PREPROCESS, read_protocol
from experiments.metrics import compute_classification_metrics
from experiments.modern_baselines import videomae2 as model_base
from experiments.modern_baselines.train_videomae2 import (
    FORMAL_CONFIG, _atomic_json, _atomic_torch_save, _seed_all, _set_phase,
    make_optimizer_scheduler, train_epoch,
)

SELECTION = "best_finetune_Menzhen888_macro_f1"
SCHEMA = "fsn-videomae2-menzhen-development-v1"
UPDATES_PER_EPOCH = 115
TOTAL_UPDATES = 12075


def public_config():
    config = asdict(FORMAL_CONFIG)
    config.pop("patience")  # Fixed budget, no early stopping; best dev-F1 selection.
    return {**config, "early_stopping": False, "selection": SELECTION,
            "epochs_total": 105, "optimizer_steps_total": TOTAL_UPDATES,
            "validation": "Menzhen888_each_epoch_select_best_F1",
            "independent_test_evaluated": False}


def cuda_device():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 \
            or not torch.cuda.is_bf16_supported():
        raise ValueError("need exactly one mapped CUDA device supporting bf16")
    device = torch.device("cuda:0")
    if "RTX 3090" not in torch.cuda.get_device_name(device):
        raise ValueError("expected the user-authorized RTX3090")
    return device


def weights_for(dataset, device):
    counts = Counter(r["label_id"] for r in dataset.raw_rows)
    if set(counts) != set(range(7)) or sum(counts.values()) != len(dataset):
        raise ValueError("training classes or count differ")
    return torch.tensor([len(dataset)/(7*counts[i]) for i in range(7)],
                        device=device, dtype=torch.float32).sqrt()


def new_directory(path):
    path = Path(path).resolve()
    repo = Path(__file__).resolve().parents[2]
    if path == repo or repo in path.parents or path in repo.parents:
        raise ValueError("private experiment output cannot be inside the checkout")
    path.mkdir(parents=True, exist_ok=False)
    return path


def infer(model, dataset, device):
    model.eval()
    logits, targets = [], []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4,
                                pin_memory=device.type=="cuda"):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type=="cuda"):
                value = model(batch["video"].to(device))
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite development-validation prediction")
            logits.append(value.float().cpu())
            targets.append(batch["label"])
    return compute_classification_metrics(torch.cat(logits), torch.cat(targets), dataset.raw_rows)


def smoke(protocol_dir, cache_dir, official_source, pretrained, output):
    output = new_directory(output)
    device = cuda_device()
    dataset = HospitalDataset(protocol_dir, cache_dir, "train")
    _seed_all(42)
    model, load = model_base.build_pretrained_model(official_source, pretrained)
    model.to(device).train()
    optimizer, _, _ = make_optimizer_scheduler(model, FORMAL_CONFIG)
    _set_phase(model, "finetune", optimizer)
    for group in optimizer.param_groups:
        group["lr"] = FORMAL_CONFIG.base_lr * group["lr_scale"]
    weights = weights_for(dataset, device)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)))
    if batch["video"].shape != (4, 3, 16, 224, 224):
        raise ValueError("real cached CUDA batch is not [4,3,16,224,224]")
    videos, labels = batch["video"].to(device), batch["label"].to(device)
    records = []
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(3):
        before = {k: v.detach().clone() for k,v in
                  {"head": model.head.weight, "patch_embed": model.patch_embed.proj.weight}.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(videos)
            loss = F.cross_entropy(logits.float(), labels, weight=weights)
        if logits.shape != (4,7) or not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise ValueError("nonfinite CUDA forward/loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
        if not torch.isfinite(norm):
            raise ValueError("nonfinite CUDA gradient")
        grads = {"head": model.head.weight.grad, "patch_embed": model.patch_embed.proj.weight.grad}
        if any(g is None or not torch.isfinite(g).all() or not g.abs().sum() > 0
               for g in grads.values()):
            raise ValueError("head/backbone gradient absent, zero or nonfinite")
        optimizer.step()
        updates = {"head": float((model.head.weight-before["head"]).float().norm()),
                   "patch_embed": float((model.patch_embed.proj.weight-before["patch_embed"]).float().norm())}
        if any(not math.isfinite(v) or v <= 0 for v in updates.values()):
            raise ValueError("CUDA optimizer did not update head/backbone")
        records.append({"step": step+1, "loss": float(loss.detach()),
                        "grad_norm": float(norm), "parameter_updates": updates})
    result = {"all_passed": True, "steps": records, "dtype": "CUDA bf16",
              "batch": 4, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "load_report": load, "training_completed": False, "test_read": False}
    _atomic_json(output/"cuda_smoke.json", result)
    print(json.dumps(result), flush=True)


def train(protocol_dir, cache_dir, official_source, pretrained, output, expected_commit, val_cache):
    output = new_directory(output)
    device = cuda_device()
    _seed_all(42)  # Smoke is another process; formal model/optimizer start fresh.
    dataset = HospitalDataset(protocol_dir, cache_dir, "train")
    validation = HospitalDataset(protocol_dir, val_cache, "val")
    model, load = model_base.build_pretrained_model(official_source, pretrained)
    model.to(device)
    optimizer, scheduler, groups = make_optimizer_scheduler(model, FORMAL_CONFIG)
    weights = weights_for(dataset, device)
    shuffle, augmentation = torch.Generator().manual_seed(42), torch.Generator().manual_seed(1000042)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=shuffle,
                        num_workers=4, pin_memory=True, drop_last=False)
    identity = {"schema": SCHEMA, "code_commit": expected_commit,
                "config": public_config(), "preprocessing": PREPROCESS,
                "protocol_sha256": sha(Path(protocol_dir)/"protocol.json"),
                "cache_seal_sha256": sha(Path(cache_dir)/"cache_seal.json"),
                "val_cache_seal_sha256": sha(Path(val_cache)/"cache_seal.json"),
                "load_report": load, "optimizer_groups": groups,
                "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "Menzhen_role": "development_validation_for_checkpoint_and_model_selection",
                "independent_test_evaluated": False, "R_D_enabled": False}
    _atomic_json(output/"run_protocol.json", identity)
    _atomic_json(output/"load_report.json", load)
    history, steps, started = [], 0, time.perf_counter()
    best, best_epoch, best_metrics = -1.0, None, None
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for index in range(FORMAL_CONFIG.total_epochs):
            phase = "head_warmup" if index < 5 else "finetune"
            _set_phase(model, phase, optimizer)
            _atomic_json(output/"status.json", {"status": "running", "phase": phase,
                         "epoch": index+1 if index<5 else index-4, "completed_epochs": len(history),
                         "training_completed": False, "test_read": False})
            row = train_epoch(model, loader, optimizer, weights, device, FORMAL_CONFIG, augmentation)
            if row["optimizer_steps"] != UPDATES_PER_EPOCH:
                raise ValueError("7307 training step count differs from pre-registration")
            gradient = row["first_step_gradients"]
            if not gradient["head"] or not math.isfinite(gradient["head"]) \
                    or (phase == "finetune" and
                        (not gradient["patch_embed"] or not math.isfinite(gradient["patch_embed"]))):
                raise ValueError("formal training head/backbone gradient is invalid")
            steps += row["optimizer_steps"]
            row.update(phase=phase, epoch=index+1 if index<5 else index-4,
                       total_epoch=index+1, total_optimizer_steps=steps)
            metrics = infer(model, validation, device)
            row.update(val_macro_f1=metrics["all"]["macro_f1"],
                       val_accuracy=metrics["all"]["accuracy"],val_per_class=metrics["all"]["per_class"])
            if phase=="finetune" and row["val_macro_f1"]>best:
                best, best_epoch, best_metrics = row["val_macro_f1"], row["epoch"], metrics
                _atomic_torch_save(output/"best.pt", {"schema": SCHEMA,"model":model.state_dict(),
                       "identity":identity,"checkpoint_epoch":best_epoch,"selection":SELECTION,
                       "metrics":metrics,"training_completed_at_save":False})
            row.update(best_epoch=best_epoch,best_macro_f1=None if best<0 else best)
            scheduler.step()
            history.append(row)
            _atomic_torch_save(output/"last.pt", {"schema": SCHEMA, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "identity": identity, "epochs_completed": len(history), "optimizer_steps": steps,
                 "training_completed": False, "automatic_resume": False})
            _atomic_json(output/"history.json", history)
            print(json.dumps(row), flush=True)
        if steps != TOTAL_UPDATES:
            raise ValueError("final optimizer budget mismatch")
        _atomic_torch_save(output/"final.pt", {"schema": SCHEMA, "model": model.state_dict(),
                "identity": identity, "epochs_completed": 105, "optimizer_steps": steps,
                "selection": "unselected_final_epoch_snapshot", "training_completed": True})
        result = {"status": "complete", "training_completed": True, "epochs_completed": 105,
                  "finetune_epochs": 100, "optimizer_steps": steps, "selection": SELECTION,
                  "final_checkpoint_sha256": sha(output/"final.pt"), "test_read": False,
                  "best_checkpoint_sha256":sha(output/"best.pt"),"best_epoch":best_epoch,
                  "best_metrics":best_metrics,"Menzhen_role":"development_validation",
                  "independent_test_evaluated":False,
                  "seconds": time.perf_counter()-started,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
        _atomic_json(output/"result.json", result)
        _atomic_json(output/"status.json", result)
    except BaseException as e:
        _atomic_json(output/"status.json", {"status": "failed", "error_type": type(e).__name__,
                     "epochs_completed": len(history), "training_completed": False,
                     "test_read": False, "automatic_retry": False})
        raise


def evaluate(protocol_dir, cache_dir, official_source, pretrained, train_dir, output):
    output = new_directory(output)
    protocol = read_protocol(protocol_dir)
    result = json.loads((Path(train_dir)/"result.json").read_text())
    checkpoint = Path(train_dir)/"best.pt"
    if result.get("training_completed") is not True or result.get("optimizer_steps") != TOTAL_UPDATES \
            or result.get("selection") != SELECTION or sha(checkpoint) != result["best_checkpoint_sha256"]:
        raise ValueError("training is incomplete or selected dev-best checkpoint changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA or payload.get("checkpoint_epoch") != result["best_epoch"] \
            or payload.get("selection") != SELECTION \
            or payload["identity"]["protocol_sha256"] != sha(Path(protocol_dir)/"protocol.json"):
        raise ValueError("selected checkpoint does not belong to this hospital protocol")
    device = cuda_device()
    _seed_all(42)
    model, _ = model_base.build_pretrained_model(official_source, pretrained)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    dataset = HospitalDataset(protocol_dir, cache_dir, "val")
    metrics = infer(model,dataset,device)
    if metrics["all"] != result["best_metrics"]["all"]:
        raise ValueError("best checkpoint development metrics did not replay exactly")
    cm = metrics["all"]["confusion_matrix"]
    public = {"status": "complete", "metrics": metrics,
              "sweep_to_reperfusion": cm[3][4], "reperfusion_to_sweep": cm[4][3],
              "checkpoint_selection": SELECTION, "training_count": 7307, "val_count": 888,
              "best_checkpoint_sha256": result["best_checkpoint_sha256"],"best_epoch":result["best_epoch"],
              "Menzhen_role": protocol["Menzhen_role"], "patient_independence_verified": False,
              "independent_test_evaluated":False,
              "note": "Menzhen is used for model selection; results are development validation, not independent test."}
    _atomic_json(output/"public_safe_summary.json", public)
    _atomic_json(output/"result.json", {"status": "complete", "dev_best_replayed": True,
                 "best_checkpoint_sha256": result["best_checkpoint_sha256"]})
    print(json.dumps(public, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("smoke", "train", "evaluate"), required=True)
    p.add_argument("--protocol-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--official-source", type=Path, required=True)
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--train-dir", type=Path)
    p.add_argument("--val-cache",type=Path)
    a = p.parse_args()
    from experiments.hospital_holdout.run import verify_code
    verify_code(a.expected_commit)
    if a.stage == "evaluate":
        if a.train_dir is None:
            p.error("evaluate requires --train-dir")
        evaluate(a.protocol_dir,a.cache_dir,a.official_source,a.pretrained,a.train_dir,a.output)
    elif a.stage == "smoke":
        smoke(a.protocol_dir,a.cache_dir,a.official_source,a.pretrained,a.output)
    else:
        if a.val_cache is None:p.error("train requires --val-cache (Menzhen development validation)")
        train(a.protocol_dir,a.cache_dir,a.official_source,a.pretrained,a.output,a.expected_commit,a.val_cache)


if __name__ == "__main__":
    main()
