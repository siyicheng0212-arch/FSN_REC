"""Train/evaluate matched Kinetics-pretrained video backbone comparisons.

Only the sealed full train7372/val823 development protocol is opened. The test
split is never read. Backbones share exact manifests, cached 36-frame clips,
the same 16 temporal sample positions, class weighting, horizontal-flip stream,
and checkpoint-selection metric. Each backbone uses the normalization and
spatial scale appropriate to its published torchvision Kinetics weights.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from experiments.metrics import compute_classification_metrics
from experiments.modern_baselines import videomae2 as video_data
from experiments.comparison.vision_models import VISION_BACKBONES, build_backbone


class RawCacheClipDataset(Dataset):
    """Return the unnormalized, sealed uint8 36-frame cache and clip metadata."""

    def __init__(self, single_view_dataset):
        self.base = single_view_dataset.base
        self.split = single_view_dataset.split

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        row = self.base[index]
        if row["video"].dtype != torch.uint8 or tuple(row["video"].shape) != (36, 3, 224, 224):
            raise ValueError("raw cache data must be uint8 [36,3,224,224]")
        return {
            "video": row["video"],
            "label": int(row["label"]),
            "clip_id": row["clip_id"],
            "source": row["source"],
            "duration": float(row["duration"]),
        }


def _seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_output(path: Path, protected: tuple[Path, ...]) -> Path:
    path = path.expanduser().resolve()
    if path.exists() or not path.parent.is_dir():
        raise FileExistsError("output must be a fresh directory and its parent must exist")
    for root in protected:
        root = root.resolve()
        if path == root or path in root.parents or root in path.parents:
            raise ValueError("output must be outside the repository, feature source, manifests and video cache")
    return path


def _class_weights(dataset: RawCacheClipDataset, device, mode: str) -> torch.Tensor:
    counts = Counter(record.label_id for record in dataset.base.records)
    if set(counts) != set(range(7)) or sum(counts.values()) != len(dataset):
        raise ValueError("train split must cover all seven classes with exact counts")
    inverse = torch.tensor(
        [len(dataset) / (7 * counts[index]) for index in range(7)],
        dtype=torch.float32,
        device=device,
    )
    if mode == "sqrt_inverse":
        return inverse.sqrt()
    if mode == "none":
        return torch.ones_like(inverse)
    raise ValueError("class weight mode must be sqrt_inverse or none")


def _autocast(device: torch.device):
    enabled = device.type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def _metadata(batch):
    return [
        {"clip_id": batch["clip_id"][i], "source": batch["source"][i],
         "duration": float(batch["duration"][i])}
        for i in range(len(batch["clip_id"]))
    ]


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    logits_rows, label_rows, meta_rows, private_rows = [], [], [], []
    seen = set()
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        ids = batch["clip_id"]
        if any(clip_id in seen for clip_id in ids):
            raise ValueError("duplicate clip encountered in validation")
        seen.update(ids)
        raw = batch["video"].to(device, non_blocking=device.type == "cuda")
        labels = batch["label"].to(device, non_blocking=device.type == "cuda")
        with _autocast(device):
            logits = model(raw)
        logits = logits.float()
        if logits.shape != (len(ids), 7) or not bool(torch.isfinite(logits).all()):
            raise ValueError("invalid validation logits")
        logits_cpu = logits.cpu()
        labels_cpu = labels.cpu()
        rows = _metadata(batch)
        logits_rows.append(logits_cpu)
        label_rows.append(labels_cpu)
        meta_rows.extend(rows)
        predictions = logits_cpu.argmax(dim=-1).tolist()
        for row, label, prediction, scores in zip(rows, labels_cpu.tolist(), predictions, logits_cpu.tolist()):
            private_rows.append({
                "clip_id": row["clip_id"],
                "target": int(label),
                "prediction": int(prediction),
                "source": row["source"],
                "duration": float(row["duration"]),
                "logits": [float(value) for value in scores],
            })
    if len(seen) != 823:
        raise ValueError(f"validation must cover exact val823; got {len(seen)}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    else:
        elapsed, peak_allocated, peak_reserved = time.perf_counter() - start, 0, 0
    scores = torch.cat(logits_rows)
    targets = torch.cat(label_rows)
    metrics = compute_classification_metrics(scores, targets, meta_rows)
    return {
        "metrics": metrics,
        "predictions": private_rows,
        "elapsed_seconds": elapsed,
        "clips_per_second": len(seen) / max(elapsed, 1e-9),
        "peak_cuda_allocated_bytes": int(peak_allocated),
        "peak_cuda_reserved_bytes": int(peak_reserved),
    }


def _make_optimizer(model, mode: str, args):
    classifier_ids = {id(parameter) for parameter in model.classifier_layer().parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(mode == "finetune")
    for parameter in model.classifier_layer().parameters():
        parameter.requires_grad_(True)

    grouped = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_head = id(parameter) in classifier_ids
        no_decay = parameter.ndim == 1 or name.endswith(".bias")
        key = ("head" if is_head else "backbone", "nodecay" if no_decay else "decay")
        if key not in grouped:
            grouped[key] = {
                "params": [],
                "lr": args.head_lr if is_head else args.backbone_lr,
                "weight_decay": 0.0 if no_decay else args.weight_decay,
                "name": f"{key[0]}_{key[1]}",
            }
        grouped[key]["params"].append(parameter)
    if not grouped:
        raise ValueError("no trainable parameters selected for visual benchmark")
    groups = [grouped[key] for key in sorted(grouped)]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    summary = [
        {
            "name": group["name"],
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "parameters": sum(parameter.numel() for parameter in group["params"]),
        }
        for group in groups
    ]
    return optimizer, summary


def _set_training_mode(model, mode):
    model.train()
    if mode == "linear_probe":
        # Freeze BatchNorm running statistics and dropout in a frozen encoder.
        model.core.eval()
        model.classifier_layer().train()


def _train_epoch(model, loader, optimizer, weights, device, args, flip_generator):
    _set_training_mode(model, args.mode)
    iterator = iter(loader)
    steps = samples = 0
    loss_values = []
    started = time.perf_counter()

    while True:
        window = []
        for _ in range(args.accumulation_steps):
            item = next(iterator, None)
            if item is None:
                break
            window.append(item)
        if not window:
            break

        window_weight_mass = 0.0
        for item in window:
            target = item["label"].to(device, non_blocking=device.type == "cuda")
            window_weight_mass += float(weights[target].sum().item())
        if window_weight_mass <= 0 or not math.isfinite(window_weight_mass):
            raise ValueError("invalid loss weight mass")
        optimizer.zero_grad(set_to_none=True)

        for item in window:
            videos = item["video"].clone()
            flip = torch.rand(len(videos), generator=flip_generator) < args.horizontal_flip_probability
            if bool(flip.any()):
                videos[flip] = videos[flip].flip(-1)
            videos = videos.to(device, non_blocking=device.type == "cuda")
            target = item["label"].to(device, non_blocking=device.type == "cuda")
            with _autocast(device):
                logits = model(videos)
                numerator = F.cross_entropy(
                    logits.float(), target, weight=weights, reduction="sum"
                )
                loss = numerator / window_weight_mass
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("nonfinite training loss/logits")
            loss.backward()
            loss_values.append(float(numerator.detach().cpu()))
            samples += len(target)

        grad_norm = nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            args.grad_clip_norm,
        )
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError("nonfinite gradient norm")
        optimizer.step()
        steps += 1

    if samples != len(loader.dataset):
        raise ValueError(f"train epoch visited {samples} clips, expected {len(loader.dataset)}")
    return {
        "samples_seen": samples,
        "optimizer_steps": steps,
        "weighted_ce_numerator_sum": float(sum(loss_values)),
        "train_seconds": time.perf_counter() - started,
    }


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _write_jsonl(path: Path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _run_one(name, datasets, args, device, output):
    _seed(args.seed)
    model, model_audit = build_backbone(name, pretrained=not args.random_init)
    model.set_mode(args.mode)
    model.to(device)
    optimizer, optimizer_groups = _make_optimizer(model, args.mode, args)
    train_weights = _class_weights(datasets["train"], device, args.class_weight_mode)
    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    flip_generator = torch.Generator().manual_seed(args.seed + 1777)
    history, best_state = [], None
    best_f1, best_epoch, stale = -1.0, 0, 0
    total_start = time.perf_counter()
    peak_train_allocated = 0
    peak_train_reserved = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        train_result = _train_epoch(
            model, train_loader, optimizer, train_weights, device, args, flip_generator
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_train_allocated = max(peak_train_allocated, torch.cuda.max_memory_allocated(device))
            peak_train_reserved = max(peak_train_reserved, torch.cuda.max_memory_reserved(device))
        val_result = evaluate(model, val_loader, device)
        peak_train_allocated = max(peak_train_allocated, int(val_result["peak_cuda_allocated_bytes"]))
        peak_train_reserved = max(peak_train_reserved, int(val_result["peak_cuda_reserved_bytes"]))
        score = float(val_result["metrics"]["all"]["macro_f1"])
        if score > best_f1:
            best_f1, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        history_row = {
            "epoch": epoch,
            **train_result,
            "val_macro_f1": score,
            "val_accuracy": float(val_result["metrics"]["all"]["accuracy"]),
            "best_macro_f1": best_f1,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
        }
        history.append(history_row)
        print(
            f"[{name}/{args.mode}] epoch={epoch}/{args.epochs} "
            f"val_macro_f1={score:.6f} acc={val_result['metrics']['all']['accuracy']:.6f} "
            f"best={best_f1:.6f}",
            flush=True,
        )
        _write_json(output / "history.json", {"epochs": history})
        if stale >= args.patience:
            break
        # A fixed per-epoch cosine schedule is shared across all model types.
        # It is intentionally simple and identical for linear-probe/fine-tune runs.
        progress = epoch / max(args.epochs, 1)
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
        for group, initial_group in zip(optimizer.param_groups, optimizer_groups):
            group["lr"] = initial_group["lr"] * multiplier

    if best_state is None:
        raise RuntimeError(f"{name}: no validation checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    final = evaluate(model, val_loader, device)
    total_seconds = time.perf_counter() - total_start
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    checkpoint = {
        "schema": "fsn-vision-backbone-benchmark-v1",
        "backbone": name,
        "mode": args.mode,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "model_audit": model_audit,
        "optimizer_groups": optimizer_groups,
        "state_dict": best_state,
    }
    torch.save(checkpoint, output / "best.pt")
    _write_jsonl(output / "private_val_predictions.jsonl", final["predictions"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = max(peak_train_allocated, int(final["peak_cuda_allocated_bytes"]))
        peak_reserved = max(peak_train_reserved, int(final["peak_cuda_reserved_bytes"]))
    else:
        peak_allocated = peak_reserved = 0
    return {
        "backbone": name,
        "mode": args.mode,
        "seed": args.seed,
        "pretrained": not args.random_init,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_val_macro_f1": best_f1,
        "parameters_total": total_params,
        "parameters_trainable": trainable_params,
        "optimizer_groups": optimizer_groups,
        "training_seconds_total": total_seconds,
        "validation": final["metrics"],
        "validation_inference_seconds": final["elapsed_seconds"],
        "validation_clips_per_second": final["clips_per_second"],
        "peak_cuda_allocated_bytes": int(peak_allocated),
        "peak_cuda_reserved_bytes": int(peak_reserved),
        "model_audit": model_audit,
    }


def run(args):
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("epochs, patience, and batch sizes must be positive")
    if args.accumulation_steps < 1 or args.workers < 0:
        raise ValueError("accumulation steps must be positive and workers nonnegative")
    if args.backbone_lr <= 0 or args.head_lr <= 0 or not math.isfinite(args.backbone_lr + args.head_lr):
        raise ValueError("learning rates must be positive and finite")
    if args.weight_decay < 0 or not math.isfinite(args.weight_decay):
        raise ValueError("weight_decay must be nonnegative and finite")
    if not 0 <= args.horizontal_flip_probability <= 1:
        raise ValueError("horizontal flip probability must be in [0,1]")
    names = tuple(part.strip().lower() for part in args.models.split(",") if part.strip())
    if not names or len(set(names)) != len(names) or not set(names) <= set(VISION_BACKBONES):
        raise ValueError(f"--models must be unique values from {VISION_BACKBONES}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    manifest_dir, cache_dir = Path(args.manifest_dir).expanduser().resolve(), Path(args.cache_dir).expanduser().resolve()
    # This preflight verifies the immutable manifests, every cache entry, split
    # and class coverage before any output directory or model is created.
    datasets, data_audit = video_data.audit_frozen_data(manifest_dir, cache_dir)
    repo = Path(__file__).resolve().parents[2]
    output = _safe_output(Path(args.output), (repo, manifest_dir, cache_dir))
    raw_datasets = {split: RawCacheClipDataset(dataset) for split, dataset in datasets.items()}
    output.mkdir(parents=False)

    protocol = {
        "schema": "fsn-vision-backbone-benchmark-v1",
        "status": "running",
        "data_protocol": video_data.DATA_PROTOCOL,
        "counts": data_audit["counts"],
        "manifest_sha256": data_audit["manifest_sha256"],
        "cache_mapping_sha256": data_audit["cache_mapping_sha256"],
        "validation_role": "val823_reused_development_validation_not_independent_test",
        "test_read": False,
        "model_order": list(names),
        "initialization": "Kinetics-400 supervised torchvision weights unless --random-init",
        "mode": args.mode,
        "training": {
            "seed": args.seed,
            "max_epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.accumulation_steps,
            "optimizer": "AdamW",
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.grad_clip_norm,
            "class_weight_mode": args.class_weight_mode,
            "horizontal_flip_probability": args.horizontal_flip_probability,
            "selection_metric": "full seven-class val823 Macro-F1; earliest checkpoint on ties",
        },
        "data_audit": data_audit,
    }
    _write_json(output / "private_protocol.json", protocol)
    _write_json(output / "status.json", {"status": "running", "completed_models": []})
    results = {}
    try:
        for name in names:
            model_dir = output / name
            model_dir.mkdir()
            print(f"starting visual backbone {name} mode={args.mode}", flush=True)
            results[name] = _run_one(name, raw_datasets, args, device, model_dir)
            _write_json(output / "status.json", {
                "status": "running",
                "completed_models": list(results),
            })
        public = {
            "schema": "fsn-vision-backbone-benchmark-v1",
            "status": "completed",
            "data_protocol": video_data.DATA_PROTOCOL,
            "counts": data_audit["counts"],
            "validation_role": "val823_reused_development_validation_not_independent_test",
            "test_read": False,
            "mode": args.mode,
            "models": results,
            "comparison": {
                name: {
                    "macro_f1": row["validation"]["all"]["macro_f1"],
                    "accuracy": row["validation"]["all"]["accuracy"],
                    "parameters_total": row["parameters_total"],
                    "parameters_trainable": row["parameters_trainable"],
                    "validation_clips_per_second": row["validation_clips_per_second"],
                    "peak_cuda_allocated_bytes": row["peak_cuda_allocated_bytes"],
                }
                for name, row in results.items()
            },
            "interpretation_warning": (
                "single-seed development results; do not call val823 an independent test; "
                "Kinetics-pretrained backbones differ in pretraining recipe; preprocessing was "
                "kept architecture-compatible while sampling the same 16 temporal positions"
            ),
        }
        _write_json(output / "public_safe_summary.json", public)
        protocol["status"] = "completed"
        _write_json(output / "private_protocol.json", protocol)
        _write_json(output / "status.json", {"status": "completed", "completed_models": list(results)})
        return public
    except BaseException as error:
        _write_json(output / "status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "completed_models": list(results),
            "error_type": type(error).__name__,
        })
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", default="mvit_v2_s,swin3d_t,r2plus1d_18")
    parser.add_argument("--mode", choices=("linear_probe", "finetune"), default="linear_probe")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--class-weight-mode", choices=("sqrt_inverse", "none"), default="sqrt_inverse")
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args)["comparison"], ensure_ascii=False, indent=2), flush=True)
