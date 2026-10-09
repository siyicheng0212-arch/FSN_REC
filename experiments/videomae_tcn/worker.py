"""Fresh TCN smoke, train and same-checkpoint evaluation stages.

A is represented by read-only numeric features. No visual backbone is imported
into the optimizer; formal TCN computation remains float32 like the old TCN.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.metrics import DEFAULT_CLASS_NAMES
from experiments.relation.data import sha256_file
from experiments.relation.tcn_baseline import (
    FrozenEvidenceTCN, TCNConfig, SegmentDataset, collate_segments, _forward_loss, evaluate)
from .evidence import SCHEMA, TRAINING, prepare
from .io import atomic_json, unchanged, write_json
from .metrics import summarize


def seed_all():
    random.seed(TRAINING["seed"])
    np.random.seed(TRAINING["seed"])
    torch.manual_seed(TRAINING["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAINING["seed"])


def loaders(segments):
    return {split: DataLoader(SegmentDataset(items), batch_size=TRAINING["batch_size_segments"],
                              shuffle=split == "train", num_workers=0,
                              generator=torch.Generator().manual_seed(TRAINING["seed"]),
                              collate_fn=collate_segments)
            for split, items in segments.items()}


def model_from(info, device):
    return FrozenEvidenceTCN(TCNConfig(**info["model_config"])).to(device)


def gradient_norms(model):
    result = {}
    for family in ("stem", "blocks", "classifier"):
        squares = []
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError("nonfinite TCN gradient")
                if name.startswith(family + "."):
                    squares.append(float(parameter.grad.detach().float().square().sum()))
        result[family] = math.sqrt(sum(squares))
    return result


def checked_evaluate(model, loader, device):
    # Reuse the legacy evaluator, but fail before argmax can hide NaN logits.
    def check_output(_module, _inputs, output):
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("nonfinite TCN evaluation logits")
    handle = model.register_forward_hook(check_output)
    try:
        return evaluate(model, loader, device)
    finally:
        handle.remove()


def smoke_batch(batch, info, device):
    """Three disposable updates on a real train batch; not a training result."""
    seed_all()
    model = model_from(info, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING["lr"],
                                  weight_decay=TRAINING["weight_decay"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    with torch.no_grad():
        initial = model(batch[0].to(device), batch[1].to(device), batch[3].to(device))
        if not torch.equal(initial, batch[1].to(device)):
            raise RuntimeError("zero-initialized ordinary TCN does not exactly reproduce frozen A")
    started, steps = time.monotonic(), []
    model.train()
    for step in range(1, 4):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = _forward_loss(model, batch, device)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite real feature-batch smoke loss")
        loss.backward()
        norms = gradient_norms(model)
        before = model.classifier.weight.detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["clip_grad_norm"])
        optimizer.step()
        change = float((model.classifier.weight.detach() - before).abs().max())
        if norms["classifier"] <= 0 or change <= 0 or \
                (step > 1 and (norms["stem"] <= 0 or norms["blocks"] <= 0)):
            raise RuntimeError("TCN failed real gradient/update smoke")
        steps.append({"step": step, "loss": float(loss.detach()), "gradient_norms": norms,
                      "classifier_max_weight_update": change})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"all_passed": True, "A_instantiated": False, "A_in_optimizer": False,
            "initial_logits_equal_A": True, "disposable_updates": 3,
            "batch_segments": len(batch[-1]), "valid_clips": int(batch[3].sum()),
            "input_dim": int(batch[0].shape[-1]), "precision": "float32",
            "steps": steps, "elapsed_seconds": time.monotonic() - started,
            "runtime": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                        "numpy": str(np.__version__), "cuda": torch.version.cuda,
                        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else None}


def _torch_save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def train_head(segments, info, output, device, source_commit):
    output.mkdir(mode=0o700, exist_ok=False)
    seed_all()  # smoke randomness/optimizer are not inherited
    model = model_from(info, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING["lr"],
                                  weight_decay=TRAINING["weight_decay"])
    batches = loaders(segments)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history, best, stale, optimizer_steps = [], -math.inf, 0, 0
    train_seconds, val_seconds = 0.0, 0.0
    start = time.monotonic()
    for epoch in range(1, TRAINING["epochs"] + 1):
        model.train()
        weighted_loss, visits, epoch_steps = 0.0, 0, 0
        grad_max = {family: 0.0 for family in ("stem", "blocks", "classifier")}
        began = time.monotonic()
        for batch in batches["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss, _, valid = _forward_loss(model, batch, device)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite TCN train loss")
            loss.backward()
            norms = gradient_norms(model)
            grad_max = {key: max(grad_max[key], norms[key]) for key in grad_max}
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["clip_grad_norm"])
            optimizer.step()
            count = int(valid.sum())
            weighted_loss += float(loss.detach()) * count
            visits += count
            optimizer_steps += 1
            epoch_steps += 1
        # The zero-initialized classifier blocks gradients into the stem/TCN
        # on the very first update. A one-batch synthetic epoch must preserve
        # that legacy behavior; later updates must reach every family.
        required = ("classifier",) if optimizer_steps == 1 else ("stem", "blocks", "classifier")
        if visits != info["counts"]["train"] or any(grad_max[key] <= 0 for key in required):
            raise RuntimeError("incomplete train coverage or missing epoch TCN gradient")
        seconds_train = time.monotonic() - began
        train_seconds += seconds_train
        began = time.monotonic()
        metrics, _ = checked_evaluate(model, batches["val"], device)
        seconds_val = time.monotonic() - began
        val_seconds += seconds_val
        value = metrics["macro_f1"]
        history.append({"phase": "tcn_train", "epoch": epoch,
                        "train_cross_entropy": weighted_loss / visits,
                        "train_clip_visits": visits, "optimizer_steps": epoch_steps,
                        "cumulative_optimizer_steps": optimizer_steps,
                        "gradient_norm_max": grad_max,
                        "val_macro_f1": value, "val_accuracy": metrics["accuracy"],
                        "train_seconds": seconds_train, "val_seconds": seconds_val})
        checkpoint = {"schema": SCHEMA, "source_commit": source_commit,
                      "model": model.state_dict(), "config": asdict(model.config),
                      "training": TRAINING, "input_sha256": info["input_sha256"],
                      "A_checkpoint_sha256": info["A_checkpoint_sha256"],
                      "epoch": epoch, "val_macro_f1": value, "optimizer_steps": optimizer_steps}
        if value > best + 1e-12:
            best, stale = value, 0
            _torch_save(output / "best.pt", checkpoint)
        else:
            stale += 1
        _torch_save(output / "last.pt", {**checkpoint, "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(), "numpy_rng": repr(np.random.get_state()),
                    "python_rng": repr(random.getstate()), "history": history,
                    "note": "diagnostic last state; this launcher does not implement resume"})
        atomic_json(output / "history.json", history)
        print(json.dumps({**history[-1], "best_macro_f1": best}, allow_nan=False), flush=True)
        if stale >= TRAINING["patience"]:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    best_checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    result = {"schema": SCHEMA, "training_completed": True, "test_read": False,
              "validation_role": "val823_development_not_independent_test", "seed": 42,
              "best_epoch": best_checkpoint["epoch"], "epochs_completed": len(history),
              "best_val_macro_f1": best, "best_checkpoint_sha256": sha256_file(output / "best.pt"),
              "optimizer_steps": optimizer_steps, "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "A_parameters_trained": 0, "precision": "float32", "receptive_field_clips": model.config.receptive_field_clips,
              "resources": {"elapsed_seconds": time.monotonic() - start,
                            "train_seconds": train_seconds, "val_seconds": val_seconds,
                            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
                            if device.type == "cuda" else None,
                            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)
                            if device.type == "cuda" else None}}
    write_json(output / "result.json", result)
    return result


def evaluate_best(segments, metadata, info, output, device, source_commit):
    training = output / "tcn"
    result = json.loads((training / "result.json").read_text())
    best_path = training / "best.pt"
    if result.get("training_completed") is not True or \
            sha256_file(best_path) != result.get("best_checkpoint_sha256"):
        raise ValueError("TCN result is incomplete or its best checkpoint changed")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    expected = {"schema": SCHEMA, "config": info["model_config"], "training": TRAINING,
                "source_commit": source_commit, "input_sha256": info["input_sha256"],
                "A_checkpoint_sha256": info["A_checkpoint_sha256"], "epoch": result["best_epoch"],
                "val_macro_f1": result["best_val_macro_f1"]}
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("best checkpoint does not match fixed model, input seals or selection")
    model = model_from(info, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    _, private = checked_evaluate(model, loaders(segments)["val"], device)
    aggregate = summarize(private, metadata, segments["val"])
    for key in ("accuracy", "macro_f1", "confusion"):
        if aggregate["variants"]["VideoMAE_only"][key] != info["A_only"][key]:
            raise ValueError("A changed during TCN evaluation")
    selected = aggregate["variants"]["VideoMAE_TCN"]["macro_f1"]
    if not math.isclose(selected, result["best_val_macro_f1"], abs_tol=1e-12, rel_tol=0):
        raise ValueError("TCN best-F1 checkpoint replay differs from training selection")
    summary = {"schema": SCHEMA, "class_names": list(DEFAULT_CLASS_NAMES),
               "training_completed": True, "test_read": False, "seed": 42,
               "source_commit": source_commit, "data_protocol": "fsn-full-7372-823-development-v1",
               "validation_role": "val823_development_not_independent_test", "A_frozen": True,
               "A_checkpoint_sha256": info["A_checkpoint_sha256"],
               "best_checkpoint_sha256": result["best_checkpoint_sha256"],
               "candidate_adjacency_is_not_verified_continuity": True,
               "offline_bidirectional": True, "singletons_use_original_TCN_forward": True,
               "spatial_mean_and_max_are_same_ViT_tokens": True,
               "counts": info["counts"], "best_epoch": result["best_epoch"],
               "epochs_completed": result["epochs_completed"],
               "optimizer_steps": result["optimizer_steps"],
               "trainable_parameters": result["trainable_parameters"],
               "runtime": json.loads((output / "cuda_smoke.json").read_text()).get("runtime")
                          if (output / "cuda_smoke.json").is_file() else None,
               "receptive_field_clips": result["receptive_field_clips"],
               "resources": result["resources"], **aggregate}
    evaluation = output / "evaluation"
    evaluation.mkdir(mode=0o700, exist_ok=False)
    write_json(evaluation / "public_safe_summary.json", summary)
    # Prediction rows have IDs: keep them private, including their seven logits
    # if richer audits are added later. This file is never copied into a report.
    fd = os.open(evaluation / "private_val_predictions.jsonl", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        for row in private:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(evaluation / "result.json", {"status": "complete", "test_read": False,
               "best_checkpoint_sha256": result["best_checkpoint_sha256"],
               "val_clip_count": len(private), "macro_f1": selected})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "train", "evaluate"))
    parser.add_argument("--run-protocol", required=True)
    args = parser.parse_args(argv)
    plan = json.loads(Path(args.run_protocol).read_text())
    output = Path(plan["output"])
    if Path(args.run_protocol).resolve() != output / "run_protocol.json" or \
            plan.get("schema") != SCHEMA or plan.get("training") != TRAINING or \
            not (output / ".launch_once").is_file():
        raise ValueError("worker requires the launched fixed run protocol")
    unchanged(plan)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != plan["gpu_uuid"] or \
            not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker must see exactly the requested single GPU UUID")
    device = torch.device("cuda:0")
    segments, metadata, info = prepare(plan["index"], plan["source_protocol_dir"])
    if info != plan["evidence"]:
        raise RuntimeError("feature replay or input seals changed after launch")
    if args.stage == "smoke":
        batch = collate_segments(segments["train"][:TRAINING["batch_size_segments"]])
        write_json(output / "cuda_smoke.json", smoke_batch(batch, info, device))
    elif args.stage == "train":
        if json.loads((output / "cuda_smoke.json").read_text()).get("all_passed") is not True:
            raise RuntimeError("successful real CUDA smoke is required before training")
        train_head(segments, info, output / "tcn", device, plan["source_commit"])
    else:
        evaluate_best(segments, metadata, info, output, device, plan["source_commit"])
    unchanged(plan)


if __name__ == "__main__":
    main()
