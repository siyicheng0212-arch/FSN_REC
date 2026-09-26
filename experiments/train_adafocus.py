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


def validate_splits(manifest_dir: Path) -> dict[str, Any]:
    records = {
        split: load_pilot_manifest(manifest_dir / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("formal training requires CUDA")
    device = torch.device("cuda:0")
    seed_all(args.seed)
    split_audit = validate_splits(args.manifest_dir)
    datasets = {
        split: FullClipDataset(
            args.manifest_dir / f"{split}.jsonl",
            args.cache_dir,
            num_frames=36,
            crop_size=224,
        )
        for split in ("train", "val", "test")
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
    total = sum(train_counts.values())
    weights = torch.tensor(
        [total / (7 * train_counts[class_id]) for class_id in range(7)],
        dtype=torch.float32,
        device=device,
    )
    model.set_class_weights(weights)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    augmentation_generator = torch.Generator().manual_seed(args.seed + 1)
    output_dir = args.output_dir / args.variant / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best_macro_f1, best_epoch, stale = [], -1.0, 0, 0
    best_path = output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses, started = [], time.perf_counter()
        for batch_index, batch in enumerate(loaders["train"], 1):
            video = batch["video"]
            if torch.rand((), generator=augmentation_generator).item() < 0.5:
                video = video.flip(-1)
            video = video.to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(video)
                loss = model.compute_loss(output, target) / args.accumulation_steps
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}, batch {batch_index}")
            loss.backward()
            if batch_index % args.accumulation_steps == 0 or batch_index == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()) * args.accumulation_steps)
        scheduler.step()
        val_metrics, _, val_seconds = evaluate(model, loaders["val"], device)
        macro_f1 = val_metrics["all"]["macro_f1"]
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_seconds": time.perf_counter() - started,
            "val_seconds": val_seconds,
            "val_metrics": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps({
            "variant": args.variant, "epoch": epoch,
            "loss": row["train_loss"], "val_macro_f1": macro_f1,
        }), flush=True)
        if macro_f1 > best_macro_f1:
            best_macro_f1, best_epoch, stale = macro_f1, epoch, 0
            torch.save({
                "model": model.state_dict(), "epoch": epoch,
                "val_macro_f1": macro_f1, "split_audit": split_audit,
                "load_report": load_report, "args": vars(args),
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
        "seed": args.seed,
        "trainable_parameters": trainable_parameter_count(model),
        "split_audit": split_audit,
        "class_weights": [float(value) for value in weights],
        "load_report": load_report,
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
    parser.add_argument("--output-dir", type=Path, default=Path("formal_results"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--clip-grad", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-after-training", action="store_true")
    args = parser.parse_args()
    args.manifest_dir = args.manifest_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.checkpoint:
        args.checkpoint = args.checkpoint.resolve()
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, default=str))
