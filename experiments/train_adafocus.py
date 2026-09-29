#!/usr/bin/env python3
"""Formal single-GPU FSN trainer for original vs. modified Uni-AdaFocus.

Launch the two variants concurrently on separate GPUs with identical arguments.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.full_data import FullClipDataset
from experiments.metrics import compute_classification_metrics
from experiments.model_wrappers import (
    AdaFocusFSN,
    load_official_adafocus_checkpoint,
    load_shared_adafocus_weights,
    trainable_parameter_count,
)
from experiments.pilot_data import load_pilot_manifest


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_splits(manifest_dir: Path, include_test: bool) -> dict[str, Any]:
    split_names = ("train", "val", "test") if include_test else ("train", "val")
    records = {
        split: load_pilot_manifest(manifest_dir / f"{split}.jsonl")
        for split in split_names
    }
    pairs = [("train", "val")]
    if include_test:
        pairs.extend((("train", "test"), ("val", "test")))
    for left, right in pairs:
        clip_overlap = {r.clip_id for r in records[left]} & {r.clip_id for r in records[right]}
        group_overlap = {r.group_id for r in records[left]} & {r.group_id for r in records[right]}
        if clip_overlap or group_overlap:
            raise RuntimeError(f"split leakage between {left} and {right}")
    return {
        "counts": {split: len(rows) for split, rows in records.items()},
        "manifest_sha256": {
            split: sha256_file(manifest_dir / f"{split}.jsonl") for split in records
        },
        "class_counts": {
            split: dict(sorted(Counter(row.label_id for row in rows).items()))
            for split, rows in records.items()
        },
    }


def metadata_rows(batch: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "clip_id": batch["clip_id"][index],
            "source": batch["source"][index],
            "duration": float(batch["duration"][index]),
        }
        for index in range(len(batch["clip_id"]))
    ]


@torch.no_grad()
def evaluate(model: AdaFocusFSN, loader: DataLoader, device: torch.device) -> tuple[dict, list[dict], float]:
    model.eval()
    logits_all, targets_all, metadata_all, prediction_rows = [], [], [], []
    started = time.perf_counter()
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(video)["logits"]
        logits_cpu, target_cpu = logits.float().cpu(), target.cpu()
        rows = metadata_rows(batch)
        metadata_all.extend(rows)
        logits_all.append(logits_cpu)
        targets_all.append(target_cpu)
        predictions = logits_cpu.argmax(dim=1)
        for index, row in enumerate(rows):
            prediction_rows.append({
                **row,
                "target": int(target_cpu[index]),
                "prediction": int(predictions[index]),
                "logits": [float(value) for value in logits_cpu[index]],
            })
    metrics = compute_classification_metrics(
        torch.cat(logits_all), torch.cat(targets_all), metadata_all
    )
    return metrics, prediction_rows, time.perf_counter() - started


def make_model(args: argparse.Namespace, device: torch.device) -> tuple[AdaFocusFSN, dict]:
    common = dict(
        num_classes=7,
        device=device,
        num_glance_segments=8,
        num_input_focus_segments=36,
        num_focus_segments=12,
        patch_size=128,
        mc_sample_times=128,
    )
    seed_all(args.seed)
    baseline = AdaFocusFSN(modified=False, **common)
    if args.checkpoint:
        checkpoint_report = load_official_adafocus_checkpoint(baseline, args.checkpoint)
    elif args.allow_random_init:
        checkpoint_report = {"checkpoint": None, "warning": "random initialization"}
    else:
        raise RuntimeError("--checkpoint is required for formal training")
    baseline_state = {key: value.detach().cpu().clone() for key, value in baseline.state_dict().items()}
    if args.variant == "original":
        return baseline.to(device), checkpoint_report
    seed_all(args.seed)
    modified = AdaFocusFSN(modified=True, **common)
    shared_report = load_shared_adafocus_weights(modified, baseline_state)
    del baseline, baseline_state
    gc.collect()
    torch.cuda.empty_cache()
    return modified.to(device), {
        "official_checkpoint": checkpoint_report,
        "shared_tensors_loaded": len(shared_report["loaded_keys"]),
        "new_module_tensors": len(shared_report["missing_keys"]),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_optimizer(
    model: AdaFocusFSN,
    args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    """Build the official Uni-AdaFocus learning-rate parameter groups."""
    policies = model.core.get_optim_policies(args)
    parameter_groups: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    seen: set[int] = set()
    for policy in policies:
        parameters = [parameter for parameter in policy["params"] if parameter.requires_grad]
        duplicates = [parameter for parameter in parameters if id(parameter) in seen]
        if duplicates:
            raise RuntimeError(f"duplicate optimizer parameters in {policy['name']}")
        seen.update(id(parameter) for parameter in parameters)
        learning_rate = args.lr * policy["lr_mult"]
        weight_decay = args.weight_decay * policy["decay_mult"]
        parameter_groups.append({
            "params": parameters,
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "name": policy["name"],
        })
        summary.append({
            "name": policy["name"],
            "parameter_tensors": len(parameters),
            "parameters": sum(parameter.numel() for parameter in parameters),
            "lr_mult": policy["lr_mult"],
            "initial_lr": learning_rate,
            "decay_mult": policy["decay_mult"],
            "weight_decay": weight_decay,
        })
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if seen != expected:
        raise RuntimeError(
            f"optimizer parameter coverage mismatch: missing={len(expected - seen)}, "
            f"extra={len(seen - expected)}"
        )
    return torch.optim.SGD(
        parameter_groups,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    ), summary


def make_class_weights(
    counts: Counter[int],
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    total = sum(counts.values())
    inverse = torch.tensor(
        [total / (7 * counts[class_id]) for class_id in range(7)],
        dtype=torch.float32,
        device=device,
    )
    if mode == "none":
        return torch.ones_like(inverse)
    if mode == "sqrt_inverse":
        return inverse.sqrt()
    if mode == "inverse":
        return inverse
    raise ValueError(f"unsupported class weight mode: {mode}")


def train_epoch(
    model: AdaFocusFSN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int,
    clip_grad: float,
    augmentation_generator: torch.Generator,
    phase: str,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses, started = [], time.perf_counter()
    total_batches = len(loader)
    for batch_index, batch in enumerate(loader, 1):
        video = batch["video"]
        if torch.rand((), generator=augmentation_generator).item() < 0.5:
            video = video.flip(-1)
        video = video.to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        group_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps + 1
        group_size = min(accumulation_steps, total_batches - group_start + 1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(video)
            native_loss = model.compute_loss(output, target)
            loss = native_loss / group_size
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss during {phase}, batch {batch_index}"
            )
        loss.backward()
        if batch_index % accumulation_steps == 0 or batch_index == total_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(native_loss.detach()))
    return float(np.mean(losses)), time.perf_counter() - started


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("formal training requires CUDA")
    device = torch.device("cuda:0")
    seed_all(args.seed)
    include_test = args.test_after_training
    split_audit = validate_splits(args.manifest_dir, include_test=include_test)
    split_names = ("train", "val", "test") if include_test else ("train", "val")
    datasets = {
        split: FullClipDataset(
            args.manifest_dir / f"{split}.jsonl",
            args.cache_dir,
            num_frames=36,
            crop_size=224,
            sampling=args.sampling,
        )
        for split in split_names
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
            drop_last=split == "train",
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
    }
    model, load_report = make_model(args, device)
    train_counts = Counter(record.label_id for record in datasets["train"].records)
    weights = make_class_weights(train_counts, args.class_weight_mode, device)
    model.set_class_weights(weights)
    augmentation_generator = torch.Generator().manual_seed(args.seed + 1)
    output_dir = args.output_dir / args.variant / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best_macro_f1, best_epoch, stale = [], -1.0, None, 0
    best_path = output_dir / "best.pt"

    original_requires_grad = {
        id(parameter): parameter.requires_grad for parameter in model.parameters()
    }
    head_tokens = ("new_fc", "new_new_fc", "aux_fc")
    head_parameters = [
        parameter for name, parameter in model.named_parameters()
        if any(token in name for token in head_tokens)
    ]
    head_parameter_ids = {id(parameter) for parameter in head_parameters}
    if not head_parameters:
        raise RuntimeError("no seven-class head parameters found for warm-up")
    warmup_optimizer_groups = [{
        "name": "seven_class_heads",
        "parameter_tensors": len(head_parameters),
        "parameters": sum(parameter.numel() for parameter in head_parameters),
        "initial_lr": args.head_warmup_lr,
        "weight_decay": 0.0,
    }]
    if args.head_warmup_epochs:
        for parameter in model.parameters():
            parameter.requires_grad = id(parameter) in head_parameter_ids
        warmup_optimizer = torch.optim.AdamW(
            head_parameters, lr=args.head_warmup_lr, weight_decay=0.0
        )
        for epoch in range(1, args.head_warmup_epochs + 1):
            train_loss, train_seconds = train_epoch(
                model, loaders["train"], warmup_optimizer, device,
                args.accumulation_steps, args.clip_grad,
                augmentation_generator, "head_warmup",
            )
            val_metrics, _, val_seconds = evaluate(model, loaders["val"], device)
            macro_f1 = val_metrics["all"]["macro_f1"]
            row = {
                "phase": "head_warmup", "epoch": epoch,
                "train_loss": train_loss, "train_seconds": train_seconds,
                "val_seconds": val_seconds, "val_metrics": val_metrics,
                "lr": args.head_warmup_lr,
                "lr_by_group": {"seven_class_heads": args.head_warmup_lr},
            }
            history.append(row)
            print(json.dumps({
                "variant": args.variant, "phase": "head_warmup",
                "epoch": epoch, "loss": train_loss,
                "val_macro_f1": macro_f1,
            }), flush=True)
            if macro_f1 > best_macro_f1:
                best_macro_f1, best_epoch = macro_f1, f"head_warmup_{epoch}"
                torch.save({
                    "model": model.state_dict(), "epoch": best_epoch,
                    "val_macro_f1": macro_f1, "split_audit": split_audit,
                    "load_report": load_report,
                    "optimizer_groups": warmup_optimizer_groups,
                    "args": vars(args),
                }, best_path)
        del warmup_optimizer
        for parameter in model.parameters():
            parameter.requires_grad = original_requires_grad[id(parameter)]

    optimizer, optimizer_groups = make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_seconds = train_epoch(
            model, loaders["train"], optimizer, device,
            args.accumulation_steps, args.clip_grad,
            augmentation_generator, "finetune",
        )
        scheduler.step()
        val_metrics, _, val_seconds = evaluate(model, loaders["val"], device)
        macro_f1 = val_metrics["all"]["macro_f1"]
        row = {
            "phase": "finetune", "epoch": epoch,
            "train_loss": train_loss,
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            "val_metrics": val_metrics,
            "lr": max(group["lr"] for group in optimizer.param_groups),
            "lr_by_group": {
                group["name"]: group["lr"] for group in optimizer.param_groups
            },
        }
        history.append(row)
        print(json.dumps({
            "variant": args.variant, "phase": "finetune", "epoch": epoch,
            "loss": row["train_loss"], "val_macro_f1": macro_f1,
        }), flush=True)
        if macro_f1 > best_macro_f1:
            best_macro_f1, best_epoch, stale = macro_f1, f"finetune_{epoch}", 0
            torch.save({
                "model": model.state_dict(), "epoch": epoch,
                "val_macro_f1": macro_f1, "split_audit": split_audit,
                "load_report": load_report, "optimizer_groups": optimizer_groups,
                "args": vars(args),
            }, best_path)
        else:
            stale += 1
            if stale >= args.patience:
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = None
    if args.test_after_training:
        test_metrics, predictions, test_seconds = evaluate(model, loaders["test"], device)
        write_jsonl(output_dir / "test_predictions.jsonl", predictions)
    else:
        test_seconds = None
    result = {
        "variant": args.variant,
        "sampling": args.sampling,
        "seed": args.seed,
        "trainable_parameters": trainable_parameter_count(model),
        "split_audit": split_audit,
        "class_weights": [float(value) for value in weights],
        "class_weight_mode": args.class_weight_mode,
        "load_report": load_report,
        "warmup_optimizer_groups": warmup_optimizer_groups,
        "optimizer_groups": optimizer_groups,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "history": history,
        "test_seconds": test_seconds,
        "test_metrics": test_metrics,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("original", "fsn"), required=True)
    parser.add_argument("--manifest-dir", type=Path, default=Path("processed_server/manifests"))
    parser.add_argument("--cache-dir", type=Path, default=Path("full_cache_36f224"))
    parser.add_argument("--sampling", choices=("uniform", "three_windows"), default="uniform")
    parser.add_argument("--output-dir", type=Path, default=Path("formal_results"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--global-lr-ratio", type=float, default=0.5)
    parser.add_argument("--stn-lr-ratio", type=float, default=0.2)
    parser.add_argument("--temporal-lr-ratio", type=float, default=0.2)
    parser.add_argument("--fsn-module-lr-ratio", type=float, default=1.0)
    parser.add_argument("--head-warmup-epochs", type=int, default=5)
    parser.add_argument("--head-warmup-lr", type=float, default=1e-3)
    parser.add_argument(
        "--class-weight-mode",
        choices=("none", "sqrt_inverse", "inverse"),
        default="sqrt_inverse",
    )
    parser.add_argument("--clip-grad", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-after-training", action="store_true")
    args = parser.parse_args()
    if args.head_warmup_epochs < 0:
        parser.error("--head-warmup-epochs must be non-negative")
    args.manifest_dir = args.manifest_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.checkpoint:
        args.checkpoint = args.checkpoint.resolve()
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, default=str))
