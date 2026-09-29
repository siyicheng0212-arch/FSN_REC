#!/usr/bin/env python3
"""Train-only paired sweep/reperfusion probe: RGB vs equal-capacity RGB+motion.

This is a *diagnostic* on a video-group-disjoint split of the original training
data. It neither produces clinician/patient labels nor uses the 823-clip
development validation set for architecture or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from experiments.full_data import FullClipDataset
from experiments.pair_evidence_probe import PairEvidenceProbe
from experiments.pilot_data import load_pilot_manifest
from experiments.training_resume import (
    RESUME_SCHEMA, atomic_json_save, atomic_torch_save, capture_rng_state,
    load_epoch_checkpoint, protocol_fingerprint, restore_rng_state,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def binary_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    predictions = logits.argmax(dim=1)
    confusion = torch.zeros((2, 2), dtype=torch.long)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        confusion[target, prediction] += 1
    per_class = []
    for label, name in enumerate(("扫散", "再灌注")):
        tp = int(confusion[label, label])
        support = int(confusion[label].sum())
        predicted = int(confusion[:, label].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({
            "label": name, "support": support,
            "precision": precision, "recall": recall, "f1": f1,
        })
    return {
        "count": len(targets),
        "accuracy": float((predictions == targets).float().mean()),
        "macro_f1": sum(row["f1"] for row in per_class) / 2,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "predicted_counts": [int((predictions == index).sum()) for index in range(2)],
    }


@torch.no_grad()
def evaluate(model: PairEvidenceProbe, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    logits, targets, sources, durations = [], [], [], []
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch["video"].to(device, non_blocking=True))
        logits.append(output.float().cpu())
        targets.append((batch["label"] - 3).long())
        sources.extend(batch["source"])
        durations.extend(float(value) for value in batch["duration"])
    all_logits = torch.cat(logits)
    all_targets = torch.cat(targets)
    result = {"all": binary_metrics(all_logits, all_targets), "source_accuracy": {}}
    for source in sorted(set(sources)):
        indices = [index for index, value in enumerate(sources) if value == source]
        block = binary_metrics(all_logits[indices], all_targets[indices])
        result["source_accuracy"][source] = {
            "count": block["count"], "accuracy": block["accuracy"],
        }
    result["duration_accuracy"] = {}
    for name, selected in (
        ("le_10s", [i for i, duration in enumerate(durations) if duration <= 10]),
        ("gt_10s", [i for i, duration in enumerate(durations) if duration > 10]),
    ):
        result["duration_accuracy"][name] = (
            {"count": len(selected), "accuracy": binary_metrics(
                all_logits[selected], all_targets[selected]
            )["accuracy"]}
            if selected else {"count": 0, "accuracy": None}
        )
    return result


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("pair probe requires CUDA")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience and batch size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    manifests = {split: args.manifest_dir / f"{split}.jsonl" for split in ("train", "val")}
    records = {split: load_pilot_manifest(path) for split, path in manifests.items()}
    if {row.group_id for row in records["train"]} & {row.group_id for row in records["val"]}:
        raise RuntimeError("train-internal pair probe has overlapping video groups")
    hashes = {split: file_sha256(path) for split, path in manifests.items()}
    split_audit = {"manifest_sha256": hashes}
    datasets = {
        split: FullClipDataset(manifests[split], args.cache_dir, 36, 224, "uniform")
        for split in manifests
    }
    pair_indices = {
        split: [index for index, record in enumerate(dataset.records) if record.label_id in (3, 4)]
        for split, dataset in datasets.items()
    }
    if not all(pair_indices.values()):
        raise RuntimeError("both pair-probe splits require sweep/reperfusion clips")
    pair_datasets = {split: Subset(datasets[split], indices) for split, indices in pair_indices.items()}
    sampler_generator = torch.Generator().manual_seed(args.seed)
    torch.empty((), dtype=torch.int64).random_(generator=sampler_generator)
    worker_generators = {
        split: torch.Generator().manual_seed(args.seed + 1000 + index)
        for index, split in enumerate(("train", "val"))
    }
    augmentation_generator = torch.Generator().manual_seed(args.seed + 1)
    loaders = {
        split: DataLoader(
            dataset, batch_size=args.batch_size,
            sampler=RandomSampler(dataset, generator=sampler_generator) if split == "train" else None,
            num_workers=args.workers, persistent_workers=args.workers > 0,
            pin_memory=True, generator=worker_generators[split],
        )
        for split, dataset in pair_datasets.items()
    }
    counts = Counter(records["train"][index].label_id for index in pair_indices["train"])
    if any(counts[label] == 0 for label in (3, 4)):
        raise RuntimeError("both pair labels must be present in training")
    total = counts[3] + counts[4]
    class_weights = torch.tensor(
        [math.sqrt(total / (2 * counts[label])) for label in (3, 4)],
        device=device, dtype=torch.float32,
    )
    model = PairEvidenceProbe(args.mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.variant = f"pair_probe_{args.mode}"
    commit = code_commit()
    fingerprint = protocol_fingerprint(args, split_audit, None, commit)
    directory = args.output_dir / args.variant / f"seed_{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    best_path, last_path, result_path = (directory / name for name in ("best.pt", "last.pt", "result.json"))
    if result_path.exists():
        raise RuntimeError("result already exists; refusing overwrite")
    if args.resume_from is None and (best_path.exists() or last_path.exists()):
        raise RuntimeError("checkpoint exists; pass --resume-from last.pt or a new output")
    history, best_f1, best_epoch, stale, start = [], -1.0, None, 0, 1
    if args.resume_from:
        state = load_epoch_checkpoint(args.resume_from, fingerprint)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng_state(state["rng"], sampler_generator, worker_generators, augmentation_generator)
        history, best_f1, best_epoch, stale = (
            list(state["history"]), float(state["best_val_macro_f1"]),
            state["best_epoch"], int(state["stale"]),
        )
        start = state["epoch"] + 1
        if start > args.epochs + 1:
            raise ValueError("resume epoch exceeds configured epochs")
        if stale >= args.patience:
            start = args.epochs + 1
        saved_best = None
        if best_path.exists():
            try:
                saved_best = float(torch.load(
                    best_path, map_location="cpu", weights_only=False
                )["val_macro_f1"])
            except Exception:
                saved_best = None
        if saved_best is None or saved_best < best_f1:
            if not state["best_is_current"]:
                raise RuntimeError("best.pt is missing/stale and cannot be reconstructed")
            atomic_torch_save({"model": model.state_dict(), "epoch": state["epoch"],
                               "val_macro_f1": best_f1, "args": vars(args)}, best_path)
    for epoch in range(start, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        losses = []
        for batch in loaders["train"]:
            frames = batch["video"]
            if torch.rand((), generator=augmentation_generator).item() < 0.5:
                frames = frames.flip(-1)
            frames = frames.to(device, non_blocking=True)
            target = (batch["label"].to(device, non_blocking=True) - 3).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(frames)
                loss = nn_cross_entropy(logits.float(), target, class_weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite pair-probe loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        train_seconds = time.perf_counter() - started
        validation = evaluate(model, loaders["val"], device)
        score = validation["all"]["macro_f1"]
        is_best = score > best_f1
        if is_best:
            best_f1, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        row = {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "train_seconds": train_seconds, "validation": validation,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        state = {
            "schema_version": RESUME_SCHEMA, "protocol_fingerprint": fingerprint,
            "code_commit": commit, "phase": "finetune", "epoch": epoch,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "best_epoch": best_epoch,
            "best_val_macro_f1": best_f1, "best_is_current": is_best,
            "stale": stale, "history": history,
            "rng": capture_rng_state(sampler_generator, worker_generators, augmentation_generator),
            "load_report": {}, "optimizer_groups": [{"name": "pair_probe", "lr": args.lr}],
        }
        atomic_torch_save(state, last_path)
        if is_best:
            atomic_torch_save({"model": model.state_dict(), "epoch": epoch,
                               "val_macro_f1": best_f1, "args": vars(args)}, best_path)
        print(json.dumps({"mode": args.mode, "seed": args.seed, "epoch": epoch,
                          "loss": row["train_loss"], "inner_val_macro_f1": score,
                          "best": best_f1}, ensure_ascii=False), flush=True)
        if stale >= args.patience:
            break
    result = {
        "schema_version": "fsn-pair-probe-internal-1.0", "mode": args.mode,
        "seed": args.seed, "code_commit": commit, "manifest_sha256": hashes,
        "counts": {split: len(indices) for split, indices in pair_indices.items()},
        "class_counts": {str(key): counts[key] for key in (3, 4)},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch, "best_inner_val_macro_f1": best_f1,
        "best_validation": history[best_epoch - 1]["validation"],
        "history": history, "test_metrics": None,
    }
    atomic_json_save(result, result_path)
    return result


def nn_cross_entropy(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, target, weight=weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=PairEvidenceProbe.MODES, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    args = parser.parse_args()
    for key in ("manifest_dir", "cache_dir", "output_dir", "resume_from"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, default=str))
