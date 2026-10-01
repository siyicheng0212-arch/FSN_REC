"""Matched Original/plain/aligned/capacity development training; no test access."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.aligned_data import ResolvedFullClipDataset, load_aligned_manifest, mapping_sha256
from experiments.aligned_protocol import DATA_PROTOCOL, EVALUATION_STATUS, validate_manifest_protocol
from experiments.audit_local_motion import _rng_state as _torch_rng_state, _restore_rng as _restore_torch_rng
from experiments.audit_local_motion import json_diagnostics, prediction_effect
from experiments.metrics import compute_classification_metrics
from experiments.model_wrappers import AdaFocusFSN, load_official_adafocus_checkpoint, load_shared_adafocus_weights
from experiments.train_adafocus import seed_all, sha256_file, make_class_weights, make_optimizer

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"original": "none", "context_plain": "plain", "context_aligned": "aligned",
            "local_capacity": "capacity"}


def _rng_state():
    return _torch_rng_state(), random.getstate(), np.random.get_state()


def _restore_rng(state):
    _restore_torch_rng(state[0])
    random.setstate(state[1])
    np.random.set_state(state[2])


def source_hashes():
    files = list((ROOT / "models/Uni-AdaFocus-TSM-FSN").rglob("*.py"))
    files += [ROOT / "experiments" / name for name in (
        "train_aligned_context.py", "run_aligned_context.py", "aligned_data.py", "aligned_protocol.py",
        "model_wrappers.py", "full_data.py", "pilot_data.py", "metrics.py",
        "train_adafocus.py", "audit_local_motion.py")]
    files += [ROOT / "scripts/run_aligned_context_4gpu.sh"]
    return {str(path.relative_to(ROOT)): sha256_file(path)
            for path in sorted(set(files)) if path.is_file()}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def make_model(args, device):
    common = dict(num_classes=7, modified=False, device=device, num_glance_segments=8,
                  num_input_focus_segments=36, num_focus_segments=12, patch_size=128,
                  mc_sample_times=128)
    seed_all(args.seed)
    baseline = AdaFocusFSN(**common)
    report = load_official_adafocus_checkpoint(baseline, args.checkpoint)
    state = baseline.state_dict()
    heads = {key for key in state if any(token in key for token in ("new_fc", "new_new_fc", "aux_fc"))}
    expected = set(state) - heads
    if (set(report["missing_keys"]) != heads or report["unexpected_keys"]
            or report["skipped_shape_or_unknown_tensors"] != 0
            or report["loaded_tensors"] != len(expected)
            or report["skipped_head_tensors"] != len(heads)):
        raise RuntimeError(f"official shared checkpoint not completely loaded: {report}")
    report.update(checkpoint_sha256=sha256_file(args.checkpoint),
                  expected_shared_tensors=len(expected), reset_head_tensors=len(heads))
    if VARIANTS[args.variant] == "none":
        return baseline.to(device), report
    saved_rng = _rng_state()
    shared = {key: value.detach().cpu().clone() for key, value in state.items()}
    seed_all(args.seed)
    candidate = AdaFocusFSN(
        **common, context_mode=VARIANTS[args.variant], context_dim=args.context_dim,
        context_grid=args.context_grid, context_time_scale=args.context_time_scale,
        context_spatial_scale=args.context_spatial_scale)
    loaded = load_shared_adafocus_weights(candidate, shared)
    missing = loaded["missing_keys"]
    if (loaded["unexpected_keys"] or len(loaded["loaded_keys"]) != len(shared)
            or not missing or any(".aligned_context." not in key for key in missing)):
        raise RuntimeError(f"new module shared-state mismatch: {loaded}")
    _restore_rng(saved_rng)
    del baseline, shared
    gc.collect()
    report.update(shared_tensors_loaded=len(loaded["loaded_keys"]), new_module_keys=missing)
    return candidate.to(device), report


def video_to_device(video, device):
    result = video.to(device=device, dtype=torch.float32, non_blocking=True)
    return result.div_(255.) if video.dtype == torch.uint8 else result


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def loss_parts(model, output, target):
    """Separate CE's class-weight mass from crop penalty's sample denominator."""
    native = model.compute_loss(output, target)
    penalty = ((output["random_branch"][6][:, 2:4] - 1.) ** 2).mean() * .5
    return native - penalty, penalty


def accumulation_loss(classification, penalty, batch_samples, window_samples,
                      batch_weight_mass, window_weight_mass):
    return (classification * (batch_weight_mass / window_weight_mass)
            + penalty * (batch_samples / window_samples))


def metadata(batch):
    return [{"clip_id": batch["clip_id"][i], "group_id": batch["group_id"][i],
             "source": batch["source"][i], "duration": float(batch["duration"][i])}
            for i in range(len(batch["label"]))]


@torch.no_grad()
def evaluate(model, loader, device, seed):
    saved = _rng_state()
    model.eval()
    seed_all(seed)
    logits, targets, meta, rows = [], [], [], []
    started = time.perf_counter()
    try:
        for batch in loader:
            video = video_to_device(batch["video"], device)
            with autocast(device):
                output = model(video)
            scores = output["logits"].float().cpu()
            if not torch.isfinite(scores).all():
                raise RuntimeError("non-finite validation logits")
            target = batch["label"].cpu()
            info = metadata(batch)
            logits.append(scores); targets.append(target); meta.extend(info)
            for i, row in enumerate(info):
                rows.append({**row, "target": int(target[i]), "prediction": int(scores[i].argmax()),
                             "logits": scores[i].tolist()})
        if len(rows) != len(loader.dataset):
            raise RuntimeError("validation did not visit every manifest clip exactly once")
        return compute_classification_metrics(torch.cat(logits), torch.cat(targets), meta), rows, time.perf_counter()-started
    finally:
        _restore_rng(saved)


def train_epoch(model, loader, optimizer, device, args, augmentation_rng):
    model.train()
    iterator = iter(loader)
    total_samples, classification_sum, penalty_sum, steps = 0, 0., 0., []
    diagnostics = {}
    started = time.perf_counter()
    while True:
        window = []
        for _ in range(args.accumulation_steps):
            batch = next(iterator, None)
            if batch is None:
                break
            window.append(batch)
        if not window:
            break
        n = sum(len(batch["label"]) for batch in window)
        masses = [float(model.class_weights.detach().cpu()[batch["label"]].sum()) for batch in window]
        mass = sum(masses)
        optimizer.zero_grad(set_to_none=True)
        for batch, own_mass in zip(window, masses):
            video = batch["video"]
            if torch.rand((), generator=augmentation_rng).item() < .5:
                video = video.flip(-1)  # shared by every frame and both pathways
            video = video_to_device(video, device)
            target = batch["label"].to(device, non_blocking=True)
            with autocast(device):
                output = model(video)
                ce, penalty = loss_parts(model, output, target)
                loss = accumulation_loss(ce, penalty, len(target), n, own_mass, mass)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            classification_sum += float(ce.detach()) * len(target)
            penalty_sum += float(penalty.detach()) * len(target)
            total_samples += len(target)
        if not steps:
            module = model.context_module
            if module is not None:
                diagnostics["first_step_gradients"] = {
                    name: None if p.grad is None else float(p.grad.detach().float().norm())
                    for name, p in module.named_parameters()}
                diagnostics["forward"] = json_diagnostics(model.get_context_diagnostics())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        if not torch.isfinite(norm):
            raise RuntimeError("non-finite gradient norm")
        optimizer.step()
        steps.append(n)
    if not total_samples:
        raise RuntimeError("empty training epoch")
    if total_samples != len(loader.dataset):
        raise RuntimeError("training epoch did not visit every manifest clip exactly once")
    return {"train_loss": (classification_sum + penalty_sum) / total_samples,
            "loss_aggregation": "sample-weighted microbatch logging; gradients use exact window weight/sample denominators",
            "train_seconds": time.perf_counter()-started, "samples_seen": total_samples,
            "optimizer_steps": len(steps), "effective_batch_min": min(steps),
            "effective_batch_max": max(steps), "context_diagnostics": diagnostics}


@torch.no_grad()
def paired_audit(model, loader, device, seed, intervention):
    module = model.context_module
    names = {"module": "enabled", "time_prior": "time_prior_enabled", "spatial_prior": "spatial_prior_enabled"}
    attribute = names[intervention]
    previous = getattr(module, attribute)
    outer_rng = _rng_state()
    model.eval(); seed_all(seed)
    on_all, off_all, targets, meta, rows = [], [], [], [], []
    try:
        for batch in loader:
            video = video_to_device(batch["video"], device)
            before = _rng_state()
            setattr(module, attribute, True)
            with autocast(device):
                on = model(video)["logits"].float().cpu()
            after = _rng_state()
            _restore_rng(before)
            setattr(module, attribute, False)
            with autocast(device):
                off = model(video)["logits"].float().cpu()
            if not torch.isfinite(on).all() or not torch.isfinite(off).all():
                raise RuntimeError("non-finite paired audit logits")
            _restore_rng(after)
            target = batch["label"].cpu(); info = metadata(batch)
            on_all.append(on); off_all.append(off); targets.append(target); meta.extend(info)
            for i, row in enumerate(info):
                rows.append({**row, "target": int(target[i]), "prediction_on": int(on[i].argmax()),
                             "prediction_off": int(off[i].argmax()), "logits_on": on[i].tolist(),
                             "logits_off": off[i].tolist()})
        on, off, target = torch.cat(on_all), torch.cat(off_all), torch.cat(targets)
        return {"intervention": intervention, "comparison": "same_checkpoint; disabled module is not independently trained Original",
                "rng_replayed": True, **prediction_effect(on, off, target),
                "metrics_on": compute_classification_metrics(on, target, meta),
                "metrics_off": compute_classification_metrics(off, target, meta)}, rows
    finally:
        setattr(module, attribute, previous); _restore_rng(outer_rng)


def make_datasets(args):
    records, raw_rows = {}, {}
    for split in ("train", "val"):
        records[split], raw_rows[split] = load_aligned_manifest(args.manifest_dir/f"{split}.jsonl", split)
    for field in ("clip_id", "group_id"):
        if {getattr(row, field) for row in records["train"]} & {getattr(row, field) for row in records["val"]}:
            raise RuntimeError(f"split leakage between train and val: {field}")
    audit = {"counts": {split: len(rows) for split, rows in records.items()},
             "manifest_sha256": {split: sha256_file(args.manifest_dir/f"{split}.jsonl") for split in records},
             "class_counts": {split: dict(sorted(Counter(row.label_id for row in rows).items())) for split, rows in records.items()},
             "role_policy": "inner_split_role when present; otherwise split; source/cache split unchanged",
             "source_split_counts": {split: dict(Counter(row["split"] for row in rows)) for split, rows in raw_rows.items()}}
    validate_manifest_protocol(audit["counts"], audit["manifest_sha256"])
    if any(set(map(int, counts)) != set(range(7)) for counts in audit["class_counts"].values()):
        raise RuntimeError("all seven labels required in both splits")
    datasets = {split: ResolvedFullClipDataset(args.manifest_dir/f"{split}.jsonl", args.cache_dir, expected_split=split)
                for split in ("train", "val")}
    mapping = datasets["train"].mapping + datasets["val"].mapping
    audit["cache_mapping_sha256"] = mapping_sha256(mapping)
    audit["data_protocol"] = DATA_PROTOCOL
    audit["evaluation_status"] = EVALUATION_STATUS
    return datasets, audit


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("formal training requires CUDA")
    device = torch.device("cuda:0")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 CUDA support required")
    out = args.output_dir / args.variant / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"status.json", {"status": "preparing", "training_completed": False})
    try:
        seed_all(args.seed)
        datasets, split_audit = make_datasets(args)
        loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split=="train",
                    num_workers=args.workers, pin_memory=True, persistent_workers=args.workers>0,
                    drop_last=False, generator=torch.Generator().manual_seed(args.seed) if split=="train" else None)
                   for split, dataset in datasets.items()}
        model, load_report = make_model(args, device)
        weights = make_class_weights(Counter(r.label_id for r in datasets["train"].records), args.class_weight_mode, device)
        model.set_class_weights(weights)
        model.train()  # keep official partial-BN freeze
        original_grad = {id(p): p.requires_grad for p in model.parameters()}
        module = model.context_module
        parameter_report = {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                            "total_parameters": sum(p.numel() for p in model.parameters()),
                            "added_parameters": 0 if module is None else sum(p.numel() for p in module.parameters())}
        write_json(out/"load_report.json", load_report)
        write_json(out/"run_config.json", {**vars(args), "data_protocol": DATA_PROTOCOL, "source_sha256": source_hashes(),
                  "split_audit": split_audit, "parameters": parameter_report,
                  "gradient_policy": "Original final-head detach unchanged; local auxiliary CE supervises new module"})
        history = []
        augmentation = torch.Generator().manual_seed(args.seed+1)
        heads = [p for name, p in model.named_parameters() if any(token in name for token in ("new_fc", "new_new_fc", "aux_fc"))]
        head_ids = {id(p) for p in heads}
        if args.head_warmup_epochs:
            for p in model.parameters():
                p.requires_grad = id(p) in head_ids
            optimizer = torch.optim.AdamW(heads, lr=args.head_warmup_lr, weight_decay=0.)
            for epoch in range(1, args.head_warmup_epochs+1):
                row = train_epoch(model, loaders["train"], optimizer, device, args, augmentation)
                metrics, _, val_seconds = evaluate(model, loaders["val"], device, args.seed+1000)
                history.append({**row, "phase": "head_warmup", "epoch": epoch,
                                "val_metrics": metrics, "val_seconds": val_seconds})
                write_json(out/"history.json", history)
                torch.save({"model": model.state_dict(), "phase": "head_warmup", "epoch": epoch,
                            "args": vars(args)}, out/"last.pt")
                print(json.dumps({"variant": args.variant, "phase": "head_warmup", "epoch": epoch,
                                  "loss": row["train_loss"], "val_macro_f1": metrics["all"]["macro_f1"]}), flush=True)
            del optimizer
            for p in model.parameters():
                p.requires_grad = original_grad[id(p)]
        optimizer, optimizer_groups = make_optimizer(model, args)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        best_score, best_epoch, stale = -math.inf, None, 0
        write_json(out/"status.json", {"status": "finetune", "training_completed": False})
        for epoch in range(1, args.epochs+1):
            row = train_epoch(model, loaders["train"], optimizer, device, args, augmentation)
            metrics, _, val_seconds = evaluate(model, loaders["val"], device, args.seed+1000)
            score = metrics["all"]["macro_f1"]
            if not math.isfinite(score):
                raise RuntimeError("non-finite selection metric")
            row.update(phase="finetune", epoch=epoch, val_metrics=metrics, val_seconds=val_seconds,
                       lr_by_group={group["name"]: group["lr"] for group in optimizer.param_groups})
            history.append(row); write_json(out/"history.json", history)
            checkpoint = {"model": model.state_dict(), "epoch": epoch, "phase": "finetune",
                          "val_macro_f1": score, "args": vars(args), "split_audit": split_audit,
                          "load_report": load_report, "source_sha256": source_hashes()}
            torch.save(checkpoint, out/"last.pt")
            print(json.dumps({"variant": args.variant, "phase": "finetune", "epoch": epoch,
                              "loss": row["train_loss"], "val_macro_f1": score}), flush=True)
            if score > best_score:
                best_score, best_epoch, stale = score, epoch, 0
                torch.save(checkpoint, out/"best.pt")
            else:
                stale += 1
            scheduler.step()
            if stale >= args.patience:
                break
        checkpoint = torch.load(out/"best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        metrics, predictions, _ = evaluate(model, loaders["val"], device, args.seed+1000)
        reevaluation_delta = metrics["all"]["macro_f1"]-best_score
        (out/"val_predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in predictions))
        audits = {}
        if module is not None:
            interventions = ["module"] + (["time_prior", "spatial_prior"] if args.variant=="context_aligned" else [])
            for kind in interventions:
                result, paired = paired_audit(model, loaders["val"], device, args.seed+1000, kind)
                audits[kind] = result; write_json(out/f"val_{kind}_audit.json", result)
                (out/f"val_{kind}_paired.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in paired))
        result = {"variant": args.variant, "seed": args.seed, "data_protocol": DATA_PROTOCOL, "training_completed": True,
                  "best_epoch": best_epoch, "best_val_macro_f1": best_score,
                  "best_val_metrics_reevaluated": metrics, "split_audit": split_audit,
                  "best_val_reevaluation_delta": reevaluation_delta,
                  "parameters": parameter_report, "load_report": load_report,
                  "optimizer_groups": optimizer_groups, "same_checkpoint_audits": audits,
                  "training_seconds": sum(r["train_seconds"] for r in history),
                  "peak_cuda_memory_allocated": torch.cuda.max_memory_allocated(device),
                  "selection_phase": "finetune_only", "source_sha256": source_hashes(),
                  "test_metrics": None,
                  "timestamp_note": "normalized cache sample positions; not verified decoded PTS"}
        write_json(out/"result.json", result)
        write_json(out/"status.json", {"status": "completed", "training_completed": True})
        return result
    except BaseException as error:
        write_json(out/"status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                      "training_completed": False, "error_type": type(error).__name__})
        raise


def smoke(args):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=4:
        raise RuntimeError("smoke requires four visible CUDA GPUs")
    if args.smoke_report.exists():
        raise FileExistsError(args.smoke_report)
    datasets, audit = make_datasets(args)
    batch = next(iter(DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=False)))
    report = {"schema": "fsn-aligned-context-smoke-v1", "data_protocol": DATA_PROTOCOL,
              "split_audit": audit, "source_sha256": source_hashes(),
              "configuration": {key: value for key, value in vars(args).items() if key not in
                  {"variant", "manifest_dir", "cache_dir", "checkpoint", "output_dir", "smoke_only", "smoke_report"}},
              "checkpoint_sha256": sha256_file(args.checkpoint), "manifest_sha256": audit["manifest_sha256"],
              "cache_mapping_sha256": audit["cache_mapping_sha256"], "variants": {},
              "gpu_names": [torch.cuda.get_device_name(i) for i in range(4)], "all_passed": False}
    args.smoke_report.parent.mkdir(parents=True, exist_ok=True)
    # Leave inspectable evidence even if a later variant raises or OOMs.
    with args.smoke_report.open("x") as handle:
        json.dump(report, handle, indent=2)
    for variant, index in (("original", 0), ("context_plain", 1), ("context_aligned", 2), ("local_capacity", 3)):
        torch.cuda.set_device(index)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"GPU{index} lacks bf16 support")
        device = torch.device(f"cuda:{index}")
        torch.cuda.reset_peak_memory_stats(device)
        report["current_variant"] = variant
        write_json(args.smoke_report, report)
        args.variant = "original"; baseline, _ = make_model(args, device)
        args.variant = variant; model, _ = make_model(args, device)
        video = video_to_device(batch["video"], device); target = batch["label"].to(device)
        baseline.eval(); model.eval(); seed_all(991)
        before = _rng_state()
        with torch.no_grad(), autocast(device):
            expected = baseline(video)["logits"].float()
        _restore_rng(before)
        with torch.no_grad(), autocast(device):
            actual = model(video)["logits"].float()
        difference = float((expected-actual).abs().max())
        if not math.isfinite(difference) or difference>1e-5:
            raise RuntimeError(f"initial Original mismatch {variant}: {difference}")
        del baseline, expected, actual
        weights = make_class_weights(Counter(r.label_id for r in datasets["train"].records), args.class_weight_mode, device)
        model.set_class_weights(weights); model.train()
        parameter_report = {"total_parameters": sum(p.numel() for p in model.parameters()),
                            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                            "added_parameters": 0 if model.context_module is None else sum(p.numel() for p in model.context_module.parameters())}
        optimizer, _ = make_optimizer(model, args)
        losses, gradients = [], []
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                output = model(video); loss = model.compute_loss(output, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite smoke loss {variant}")
            loss.backward()
            module = model.context_module
            grad = {} if module is None else {name: None if p.grad is None else float(p.grad.float().norm())
                                              for name, p in module.named_parameters()}
            if any(not math.isfinite(value) for value in grad.values() if value is not None):
                raise RuntimeError(f"non-finite smoke gradient {variant}")
            if module is not None and (grad.get("up.weight", 0.) or 0.)<=0:
                raise RuntimeError(f"zero projection gradient {variant}")
            if module is not None and step==2:
                downstream = [value for name, value in grad.items() if name!="up.weight"]
                if not downstream or any(value is None for value in downstream) or not any(value>0 for value in downstream):
                    raise RuntimeError(f"no downstream module learning {variant}")
            gradients.append(grad); losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            if not torch.isfinite(norm) or norm<=0:
                raise RuntimeError(f"non-finite or zero full-model gradient {variant}")
            optimizer.step()
        report["variants"][variant] = {"passed": True, "steps": 3, "device_index": index,
            "initial_max_abs_logit_diff": difference, "losses": losses, "gradient_checks": gradients,
            "parameters": parameter_report, "class_weights": weights.detach().cpu().tolist(),
            "peak_cuda_memory_allocated": torch.cuda.max_memory_allocated(device),
            "module_diagnostics": json_diagnostics(model.get_context_diagnostics())}
        write_json(args.smoke_report, report)
        del optimizer, model, video, target, output, loss
        gc.collect(); torch.cuda.empty_cache()
    report["all_passed"] = True
    report.pop("current_variant", None)
    write_json(args.smoke_report, report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="original")
    for option in ("manifest-dir", "cache-dir", "checkpoint"):
        parser.add_argument("--"+option, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("aligned_context_results"))
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-report", type=Path)
    for option, default in (("seed",42),("epochs",100),("patience",10),("batch-size",4),
                            ("accumulation-steps",16),("workers",4),("head-warmup-epochs",5),
                            ("context-dim",64),("context-grid",3)):
        parser.add_argument("--"+option, type=int, default=default)
    for option, default in (("lr",.002),("weight-decay",.0005),("head-warmup-lr",.001),
                            ("clip-grad",20.),("context-time-scale",.25),("context-spatial-scale",1.),
                            ("context-lr-ratio",1.),("global-lr-ratio",.5),("stn-lr-ratio",.2),
                            ("temporal-lr-ratio",.2)):
        parser.add_argument("--"+option, type=float, default=default)
    parser.add_argument("--class-weight-mode", choices=("none","sqrt_inverse","inverse"), default="sqrt_inverse")
    args = parser.parse_args(argv)
    for field in ("epochs","patience","batch_size","accumulation_steps","context_dim","context_grid"):
        if getattr(args,field)<1:
            parser.error(f"{field} must be positive")
    if args.workers<0 or args.head_warmup_epochs<0:
        parser.error("workers and warmup must be nonnegative")
    for field in ("lr","weight_decay","head_warmup_lr","clip_grad","context_time_scale",
                  "context_spatial_scale","context_lr_ratio","global_lr_ratio","stn_lr_ratio","temporal_lr_ratio"):
        if not math.isfinite(getattr(args,field)) or getattr(args,field)<0:
            parser.error(f"{field} must be finite and nonnegative")
    if min(args.lr,args.head_warmup_lr,args.clip_grad,args.context_time_scale,args.context_spatial_scale)<=0:
        parser.error("learning rates, clip norm and prior scales must be positive")
    for field in ("manifest_dir","cache_dir","checkpoint","output_dir","smoke_report"):
        if getattr(args,field) is not None:
            setattr(args,field,getattr(args,field).resolve())
    if args.smoke_only and args.smoke_report is None:
        parser.error("--smoke-only requires --smoke-report")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(smoke(arguments) if arguments.smoke_only else run(arguments), ensure_ascii=False, indent=2, default=str))
