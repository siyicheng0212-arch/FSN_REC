#!/usr/bin/env python3
"""Train a small semantic corroboration head on a frozen, trained Original.

The input is still one clip. This script never reads an independent test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from experiments.full_data import FullClipDataset
from experiments.model_wrappers import AdaFocusFSN
from experiments.semantic_window_evidence import FSNSemanticWindowEvidence
from experiments.train_adafocus import (
    evaluate,
    make_class_weights,
    seed_all,
    sha256_file,
    training_code_commit,
    validate_splits,
)
from experiments.training_resume import atomic_json_save, atomic_torch_save


SCHEMA = "fsn-semantic-window-epoch-v1"


def _rng_state(
    sampler: torch.Generator,
    workers: dict[str, torch.Generator],
    augmentation: torch.Generator,
) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "sampler": sampler.get_state(),
        "workers": {key: value.get_state() for key, value in workers.items()},
        "augmentation": augmentation.get_state(),
    }


def _restore_rng(
    state: dict[str, Any],
    sampler: torch.Generator,
    workers: dict[str, torch.Generator],
    augmentation: torch.Generator,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    sampler.set_state(state["sampler"])
    for key, value in workers.items():
        value.set_state(state["workers"][key])
    augmentation.set_state(state["augmentation"])


def _fingerprint(
    args: argparse.Namespace, split: dict[str, Any],
    baseline_sha256: str, code_commit: str | None,
) -> str:
    fields = {
        "schema": SCHEMA,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in ("resume_from", "allow_cpu")
        },
        "split": split,
        "baseline_sha256": baseline_sha256,
        "code_commit": code_commit,
    }
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _trained_original(
    path: Path, device: torch.device, split: dict[str, Any]
) -> AdaFocusFSN:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = split["manifest_sha256"]
    observed = checkpoint.get("split_audit", {}).get("manifest_sha256", {})
    if observed.get("train") != expected["train"] or observed.get("val") != expected["val"]:
        raise ValueError("trained Original checkpoint has a different data protocol")
    if checkpoint.get("args", {}).get("variant") != "original":
        raise ValueError("baseline checkpoint must come from Original")
    state = checkpoint["model"]
    baseline = AdaFocusFSN(
        num_classes=7,
        modified=False,
        device=device,
        num_glance_segments=8,
        num_input_focus_segments=36,
        num_focus_segments=12,
        patch_size=128,
        mc_sample_times=128,
    )
    # A None-registered buffer cannot load a tensor key until materialized.
    if "class_weights" in state:
        baseline.set_class_weights(state["class_weights"])
    baseline.load_state_dict(state, strict=True)
    baseline.eval()
    return baseline.to(device)


def _train_epoch(
    model: FSNSemanticWindowEvidence,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int,
    clip_grad: float,
    augmentation: torch.Generator,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    started = time.perf_counter()
    total = len(loader)
    for index, batch in enumerate(loader, 1):
        video = batch["video"]
        if torch.rand((), generator=augmentation).item() < 0.5:
            video = video.flip(-1)
        video = video.to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        group_start = ((index - 1) // accumulation_steps) * accumulation_steps + 1
        group_size = min(accumulation_steps, total - group_start + 1)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(video)
            native_loss = model.compute_loss(output, target)
            loss = native_loss / group_size
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite semantic evidence loss at batch {index}")
        loss.backward()
        if index % accumulation_steps == 0 or index == total:
            torch.nn.utils.clip_grad_norm_(
                model.evidence_module.parameters(), clip_grad
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(native_loss.detach()))
    return float(np.mean(losses)), time.perf_counter() - started


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs < 1 or args.patience < 1 or args.accumulation_steps < 1:
        raise ValueError("epochs, patience and accumulation_steps must be positive")
    if args.mode not in ("corroborated", "plain_mean"):
        raise ValueError("unsupported semantic evidence mode")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("formal semantic evidence training requires CUDA")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    split = validate_splits(args.manifest_dir, include_test=False)
    datasets = {
        name: FullClipDataset(
            args.manifest_dir / f"{name}.jsonl",
            args.cache_dir,
            num_frames=36,
            crop_size=224,
            sampling="uniform",
        )
        for name in ("train", "val")
    }
    sampler_generator = torch.Generator().manual_seed(args.seed)
    torch.empty((), dtype=torch.int64).random_(generator=sampler_generator)
    worker_generators = {
        name: torch.Generator().manual_seed(args.seed + 1000 + index)
        for index, name in enumerate(("train", "val"))
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=(
                RandomSampler(dataset, generator=sampler_generator)
                if name == "train" else None
            ),
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
            drop_last=name == "train",
            generator=worker_generators[name],
        )
        for name, dataset in datasets.items()
    }
    baseline_sha = sha256_file(args.baseline_checkpoint)
    baseline = _trained_original(args.baseline_checkpoint, device, split)
    seed_all(args.seed)
    model = FSNSemanticWindowEvidence(
        baseline,
        corroboration=args.mode == "corroborated",
        pair_aux_weight=args.pair_aux_weight,
    ).to(device)
    counts = Counter(row.label_id for row in datasets["train"].records)
    class_weights = make_class_weights(counts, "sqrt_inverse", device)
    pair_counts = counts[3] + counts[4]
    pair_weights = torch.tensor(
        [
            math.sqrt(pair_counts / (2 * counts[3])),
            math.sqrt(pair_counts / (2 * counts[4])),
        ],
        device=device,
    )
    pair_weights /= pair_weights.mean()
    model.set_class_weights(class_weights, pair_weights)
    optimizer = torch.optim.AdamW(
        model.evidence_module.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    augmentation_generator = torch.Generator().manual_seed(args.seed + 1)
    output = args.output_dir / args.mode / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    best_path, last_path = output / "best.pt", output / "last.pt"
    result_path = output / "result.json"
    if result_path.exists():
        raise RuntimeError("result.json already exists; refusing overwrite")
    if args.resume_from is None and (best_path.exists() or last_path.exists()):
        raise RuntimeError("checkpoint exists; use --resume-from last.pt or new output")
    code_commit = training_code_commit()
    fingerprint = _fingerprint(args, split, baseline_sha, code_commit)
    history: list[dict[str, Any]] = []
    best_f1, best_epoch, stale, start_epoch = -1.0, None, 0, 1
    best_head_state: dict[str, torch.Tensor] | None = None
    if args.resume_from is not None:
        state = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if state.get("schema_version") != SCHEMA or state.get("fingerprint") != fingerprint:
            raise ValueError("resume checkpoint protocol fingerprint mismatch")
        model.evidence_module.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = list(state["history"])
        best_f1, best_epoch, stale = (
            float(state["best_val_macro_f1"]), state["best_epoch"], int(state["stale"])
        )
        best_head_state = state["best_head"]
        _restore_rng(
            state["rng"], sampler_generator, worker_generators,
            augmentation_generator,
        )
        start_epoch = int(state["epoch"]) + 1
        saved_best_f1 = None
        if best_path.exists():
            try:
                saved_best = torch.load(
                    best_path, map_location="cpu", weights_only=False
                )
                if (
                    saved_best.get("fingerprint") == fingerprint
                    and saved_best.get("baseline_sha256") == baseline_sha
                ):
                    saved_best_f1 = float(saved_best["val_macro_f1"])
            except Exception:
                saved_best_f1 = None
        if best_head_state is not None and (
            saved_best_f1 is None or saved_best_f1 < best_f1
        ):
            atomic_torch_save(
                {
                    "schema_version": SCHEMA,
                    "head": best_head_state,
                    "baseline_sha256": baseline_sha,
                    "epoch": best_epoch,
                    "val_macro_f1": best_f1,
                    "fingerprint": fingerprint,
                },
                best_path,
            )

    for epoch in range(start_epoch, args.epochs + 1):
        if stale >= args.patience:
            break
        train_loss, train_seconds = _train_epoch(
            model, loaders["train"], optimizer, device,
            args.accumulation_steps, args.clip_grad, augmentation_generator,
        )
        scheduler.step()
        val_metrics, _, val_seconds = evaluate(model, loaders["val"], device)
        macro_f1 = float(val_metrics["all"]["macro_f1"])
        if not math.isfinite(macro_f1):
            raise RuntimeError("non-finite development-validation macro-F1")
        row = {
            "phase": "semantic_head",
            "epoch": epoch,
            "train_loss": train_loss,
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            "val_metrics": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "alpha": float(model.evidence_module.alpha.detach()),
        }
        history.append(row)
        print(json.dumps({
            "variant": "semantic_window",
            "mode": args.mode,
            "seed": args.seed,
            "epoch": epoch,
            "loss": train_loss,
            "val_macro_f1": macro_f1,
            "alpha": row["alpha"],
        }), flush=True)
        improved = macro_f1 > best_f1
        if improved:
            best_f1, best_epoch, stale = macro_f1, epoch, 0
            best_head_state = {
                key: value.detach().cpu().clone()
                for key, value in model.evidence_module.state_dict().items()
            }
        else:
            stale += 1
        state = {
            "schema_version": SCHEMA,
            "fingerprint": fingerprint,
            "baseline_sha256": baseline_sha,
            "code_commit": code_commit,
            "epoch": epoch,
            "head": model.evidence_module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "best_val_macro_f1": best_f1,
            "best_epoch": best_epoch,
            "best_head": best_head_state,
            "stale": stale,
            "rng": _rng_state(
                sampler_generator, worker_generators, augmentation_generator
            ),
        }
        atomic_torch_save(state, last_path)
        if improved:
            atomic_torch_save(
                {
                    "schema_version": SCHEMA,
                    "head": best_head_state,
                    "baseline_sha256": baseline_sha,
                    "epoch": best_epoch,
                    "val_macro_f1": best_f1,
                    "fingerprint": fingerprint,
                },
                best_path,
            )
        if stale >= args.patience:
            break

    if not best_path.exists():
        raise RuntimeError("best checkpoint missing after training")
    result = {
        "schema_version": "fsn-semantic-window-result-v1",
        "variant": "semantic_window",
        "mode": args.mode,
        "seed": args.seed,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.evidence_module.parameters()
        ),
        "split_audit": split,
        "baseline_checkpoint_sha256": baseline_sha,
        "code_commit": code_commit,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "history": history,
        "test_metrics": None,
        "selection_warning": (
            "823 clips select best.pt each epoch; this is development validation, "
            "not an independent test"
        ),
    }
    atomic_json_save(result, result_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("corroborated", "plain_mean"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pair-aux-weight", type=float, default=0.5)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    for field in ("manifest_dir", "cache_dir", "baseline_checkpoint", "output_dir"):
        setattr(args, field, getattr(args, field).resolve())
    if args.resume_from is not None:
        args.resume_from = args.resume_from.resolve()
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, default=str))
