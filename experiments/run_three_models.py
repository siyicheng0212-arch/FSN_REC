#!/usr/bin/env python3
"""Run a reproducible real-data pilot for three FSN video classifiers.

This is a pipeline/overfit pilot, not a paper result.  The local modern
reference is MViT-V2-S because the formal 2024 VideoMamba-Ti reference needs a
Linux CUDA selective-scan extension unavailable on this Intel Mac.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.metrics import compute_classification_metrics
from experiments.model_wrappers import (
    AdaFocusFSN,
    build_model,
    load_shared_adafocus_weights,
    trainable_parameter_count,
)
from experiments.pilot_data import PilotClipDataset


MODEL_NAMES = ("adafocus_original", "adafocus_fsn", "mvit_v2_s_reference")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    manifest: Path,
    cache_dir: Path,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = PilotClipDataset(
        manifest,
        cache_dir,
        num_frames=8,
        crop_size=224,
        verify_all=True,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def metadata_rows(batch: dict[str, Any]) -> list[dict[str, Any]]:
    count = len(batch["clip_id"])
    return [
        {
            "clip_id": batch["clip_id"][index],
            "source": batch["source"][index],
            "duration": float(batch["duration"][index]),
        }
        for index in range(count)
    ]


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    model.eval()
    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    metadata_all: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch in loader:
        video = batch["video"].to(device)
        target = batch["label"].to(device)
        logits = model(video)["logits"]
        rows = metadata_rows(batch)
        logits_cpu = logits.detach().cpu()
        target_cpu = target.detach().cpu()
        logits_all.append(logits_cpu)
        targets_all.append(target_cpu)
        metadata_all.extend(rows)
        predicted = logits_cpu.argmax(dim=1)
        for index, row in enumerate(rows):
            predictions.append(
                {
                    **row,
                    "target": int(target_cpu[index]),
                    "prediction": int(predicted[index]),
                    "logits": [float(value) for value in logits_cpu[index]],
                }
            )
    elapsed = time.perf_counter() - started
    metrics = compute_classification_metrics(
        torch.cat(logits_all), torch.cat(targets_all), metadata_all
    )
    return metrics, predictions, elapsed


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    max_batches: int | None,
) -> tuple[float, float, int]:
    model.train()
    losses: list[float] = []
    started = time.perf_counter()
    completed = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        video = batch["video"].to(device)
        target = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(video)
        loss = model.compute_loss(output, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at batch {batch_index}")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"non-finite gradient at batch {batch_index}")
        optimizer.step()
        losses.append(float(loss.detach()))
        completed += len(target)
    return float(np.mean(losses)), time.perf_counter() - started, completed


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


@torch.no_grad()
def baseline_equivalence(
    device: torch.device,
    baseline_state: dict[str, torch.Tensor],
    sample_video: torch.Tensor,
) -> dict[str, Any]:
    set_seed(1234)
    original = build_model("adafocus_original", device).eval()
    original.load_state_dict(baseline_state)
    set_seed(1234)
    modified = build_model("adafocus_fsn", device).eval()
    load_report = load_shared_adafocus_weights(modified, baseline_state)
    original_logits = original(sample_video.to(device))["logits"]
    modified_logits = modified(sample_video.to(device))["logits"]
    maximum_difference = float((original_logits - modified_logits).abs().max().cpu())
    del original, modified
    gc.collect()
    return {
        "max_abs_logit_difference": maximum_difference,
        "passed": maximum_difference <= 1e-5,
        "shared_tensors_loaded": len(load_report["loaded_keys"]),
        "new_module_tensors": len(load_report["missing_keys"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = args.manifest_dir.resolve()
    cache_dir = args.cache_dir.resolve()

    eval_loaders = {
        split: make_loader(
            manifests / f"{split}.jsonl",
            cache_dir,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
        )
        for split in ("val", "test")
    }

    set_seed(args.seed)
    initial_baseline = build_model("adafocus_original", device)
    baseline_state = clone_state(initial_baseline)
    sample_video = next(iter(eval_loaders["val"]))["video"][:1]
    del initial_baseline
    gc.collect()
    equivalence = baseline_equivalence(device, baseline_state, sample_video)
    if not equivalence["passed"]:
        raise RuntimeError(f"baseline equivalence failed: {equivalence}")

    result: dict[str, Any] = {
        "protocol": "real-data balanced CPU pilot; not a paper result",
        "seed": args.seed,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "baseline_equivalence": equivalence,
        "models": {},
    }

    for model_name in MODEL_NAMES:
        print(f"\n=== {model_name} ===", flush=True)
        set_seed(args.seed)
        model = build_model(model_name, device)
        load_report = None
        if model_name == "adafocus_original":
            model.load_state_dict(baseline_state)
        elif model_name == "adafocus_fsn":
            load_report = load_shared_adafocus_weights(model, baseline_state)
        # Recreate the shuffled train loader and reset RNG for every model so
        # all three see the same sample order and stochastic starting state.
        train_loader = make_loader(
            manifests / "train.jsonl",
            cache_dir,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
        )
        set_seed(args.seed)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        history: list[dict[str, Any]] = []
        best_state = clone_state(model)
        best_macro_f1 = -1.0
        for epoch in range(args.epochs):
            train_loss, train_seconds, train_samples = train_one_epoch(
                model, train_loader, device, optimizer, args.max_train_batches
            )
            val_metrics, _, val_seconds = evaluate(model, eval_loaders["val"], device)
            macro_f1 = val_metrics["all"]["macro_f1"]
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_seconds": train_seconds,
                    "train_samples": train_samples,
                    "val_seconds": val_seconds,
                    "val_metrics": val_metrics,
                }
            )
            print(
                f"epoch={epoch + 1} loss={train_loss:.4f} "
                f"val_macro_f1={macro_f1:.4f}",
                flush=True,
            )
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_state = clone_state(model)
        model.load_state_dict(best_state)
        test_metrics, predictions, test_seconds = evaluate(model, eval_loaders["test"], device)
        prediction_path = output_dir / f"{model_name}_test_predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        result["models"][model_name] = {
            "trainable_parameters": trainable_parameter_count(model),
            "history": history,
            "best_val_macro_f1": best_macro_f1,
            "test_seconds": test_seconds,
            "test_metrics": test_metrics,
            "shared_weight_load": None if load_report is None else {
                "loaded_keys": len(load_report["loaded_keys"]),
                "new_module_tensors": len(load_report["missing_keys"]),
                "unexpected_keys": load_report["unexpected_keys"],
            },
            "prediction_file": prediction_path.name,
        }
        del model, optimizer, best_state
        gc.collect()
    result_path = output_dir / "comparison_results.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=Path("experiments/pilot_manifests"))
    parser.add_argument("--cache-dir", type=Path, default=Path("experiments/pilot_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/pilot_three_models"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")
    return args


if __name__ == "__main__":
    summary = run(parse_args())
    print(json.dumps({
        "result": "experiments/results/pilot_three_models/comparison_results.json",
        "baseline_equivalence": summary["baseline_equivalence"],
        "models": {
            name: {
                "best_val_macro_f1": values["best_val_macro_f1"],
                "test_macro_f1": values["test_metrics"]["all"]["macro_f1"],
                "test_accuracy": values["test_metrics"]["all"]["accuracy"],
            }
            for name, values in summary["models"].items()
        },
    }, ensure_ascii=False, indent=2))
