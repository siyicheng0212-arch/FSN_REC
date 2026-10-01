"""Train R on frozen offline evidence; no visual/decoder parameters are updated."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import RelationDataset, audit_edges, load_edges, load_feature_index, sha256_file
from .network import RelationConfig, RelationNet


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _batch(batch, device):
    return ({key: value.to(device) for key, value in batch["left"].items()},
            {key: value.to(device) for key, value in batch["right"].items()},
            batch["target"].to(device))


@torch.no_grad()
def evaluate(model, loader, device, threshold=.9, unknown=False):
    model.eval()
    count, total_loss, tp, fp, fn, tn, active = 0, 0., 0, 0, 0, 0, 0
    for batch in loader:
        left, right, target = _batch(batch, device)
        logits = model(left, right)
        if not torch.isfinite(logits).all():
            raise RuntimeError("nonfinite validation relation logits")
        predicted = logits.sigmoid() >= threshold
        count += len(target)
        active += int(predicted.sum())
        if not unknown:
            if bool(((target != 0) & (target != 1)).any()):
                raise ValueError("unknown targets must never enter supervised BCE")
            total_loss += float(F.binary_cross_entropy_with_logits(logits, target, reduction="sum"))
            positive = target == 1
            tp += int((predicted & positive).sum())
            fp += int((predicted & ~positive).sum())
            fn += int((~predicted & positive).sum())
            tn += int((~predicted & ~positive).sum())
    metrics = {"count": count, "threshold": threshold, "active_count": active,
               "activation_fraction": active / count if count else None,
               "threshold_is_calibrated": False}
    if not unknown:
        metrics.update(bce=total_loss / count if count else None, tp=tp, fp=fp, fn=fn, tn=tn,
                       precision_as_C=tp / (tp + fp) if tp + fp else None,
                       recall_as_C=tp / (tp + fn) if tp + fn else None)
    return metrics


def train(args):
    if any(value < 1 for value in (args.epochs, args.batch_size, args.patience)):
        raise ValueError("epochs, batch_size and patience must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.threshold) or not 0 < args.threshold < 1:
        raise ValueError("lr must be finite/positive and threshold must be in (0,1)")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("output already exists; use a fresh directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    index = load_feature_index(args.index)
    edges = load_edges(args.edges, index)
    audit = audit_edges(edges)
    if not audit["train"]["C"] or not audit["train"]["D"]:
        raise ValueError("R training requires both human-defined C and D edges")
    if not audit["val"]["C"] or not audit["val"]["D"]:
        raise ValueError("R validation requires both C and D edges")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = RelationConfig(global_dim=index.global_dim, local_dim=index.local_dim, dim=args.dim,
                            heads=args.heads, endpoint_tokens=args.endpoint_tokens,
                            dropout=args.dropout, mode=args.mode)
    model = RelationNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    train_loader = DataLoader(RelationDataset(index, edges, "train"), batch_size=args.batch_size,
                              shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(RelationDataset(index, edges, "val"), batch_size=args.batch_size)
    unknown_loader = DataLoader(RelationDataset(index, edges, "val", statuses=("U",)), batch_size=args.batch_size)
    protocol = {"index_sha256": sha256_file(args.index), "edges_sha256": sha256_file(args.edges),
                "checkpoint_sha256": index.checkpoint_sha256, "config": asdict(config),
                "edge_counts": audit, "seed": args.seed, "arguments": vars(args),
                "position_basis": "cache_index_fraction_not_verified_pts",
                "R_parameters": sum(value.numel() for value in model.parameters()),
                "optimizer": "AdamW", "weight_decay": .01,
                "selection": "val C/D BCE", "U_policy": "excluded from BCE; activation audit only",
                "threshold_is_calibrated": False, "torch_version": str(torch.__version__),
                "unhashed_feature_count": sum("feature_sha256" not in row for row in index.clips.values()),
                "source_sha256": {name: sha256_file(Path(__file__).resolve().parent / name)
                                  for name in ("network.py", "data.py", "train.py")}}
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "protocol.json", protocol)
    history, best, stale, start = [], float("inf"), 0, time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0., 0
        for batch in train_loader:
            left, right, target = _batch(batch, device)
            if bool(((target != 0) & (target != 1)).any()):
                raise ValueError("unknown targets must never enter supervised BCE")
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(left, right), target)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite R training loss")
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
                raise RuntimeError("nonfinite R gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(target)
            count += len(target)
        val = evaluate(model, val_loader, device, args.threshold)
        history.append({"epoch": epoch, "train_bce": total / count, "val": val})
        if val["bce"] < best:
            best, stale = val["bce"], 0
            torch.save({"model": model.state_dict(), "config": asdict(config), "protocol": protocol,
                        "epoch": epoch, "val_bce": best}, output / "best.pt")
        else:
            stale += 1
        _write(output / "history.json", history)
        print(f"epoch {epoch}: train BCE {total / count:.6f}; val BCE {val['bce']:.6f}", flush=True)
        if stale >= args.patience:
            break
    saved = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    metrics = {"best_epoch": saved["epoch"], "epochs_completed": len(history),
               "elapsed_seconds": time.monotonic() - start,
               "val": evaluate(model, val_loader, device, args.threshold),
               "val_unknown": evaluate(model, unknown_loader, device, args.threshold, unknown=True),
               "interpretation": "development metrics; threshold is not calibrated or certified"}
    _write(output / "metrics.json", metrics)
    return metrics


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "edges", "output"):
        result.add_argument(f"--{name}", required=True)
    result.add_argument("--mode", choices=("mlp", "dual"), default="dual")
    result.add_argument("--device", default="cpu")
    for name, default in (("epochs", 30), ("batch-size", 16), ("patience", 5), ("seed", 42),
                          ("dim", 64), ("heads", 4), ("endpoint-tokens", 2)):
        result.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("lr", 1e-3), ("dropout", .1), ("threshold", .9)):
        result.add_argument(f"--{name}", type=float, default=default)
    return result


if __name__ == "__main__":
    train(parser().parse_args())
