"""Benchmark temporal heads on the same sealed frozen-VideoMAE evidence.

Models: frozen A, a train-only Markov/Viterbi decoder, clip-wise MLP, BiGRU,
residual TCN, an MS-TCN++-inspired multi-stage clip-sequence refiner, and an
ASFormer-inspired local Transformer.
The two literature-inspired models are adaptations to clip-level seven-class
sequence classification, not faithful reproductions of dense action segmentation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from experiments.dual_timescale.run import (
    CLASS_NAMES,
    RawSegmentDataset,
    _evaluate,
    _materialize_raw_segments,
    _preflight,
    _read_json,
    _seed_everything,
    _to_device,
    _write_json,
    _write_jsonl,
    collate_segments,
)
from experiments.comparison.temporal_models import (
    MarkovViterbiDecoder,
    TemporalComparisonModel,
    VALID_MODELS,
    estimate_transition_log_probs,
)


VARIANTS = ("a_only", "markov_viterbi", *VALID_MODELS)


def _safe_output(path: Path, protected: tuple[Path, ...]) -> Path:
    path = path.expanduser().resolve()
    if path.exists() or not path.parent.is_dir():
        raise FileExistsError("output must be a fresh directory and its parent must already exist")
    for root in protected:
        root = root.resolve()
        if path == root or path in root.parents or root in path.parents:
            raise ValueError("output must be separate from the repository, input cache and source protocol")
    return path


def _train_model(model_name, index, train_segments, val_segments, args, device, model_dir):
    _seed_everything(args.seed)
    model = TemporalComparisonModel(
        model_name=model_name,
        global_dim=index.global_dim,
        local_dim=index.local_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        tcn_dilations=tuple(args.tcn_dilations),
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        local_radius=args.local_radius,
        max_sequence_length=args.max_sequence_length,
    ).to(device)
    train_loader = DataLoader(
        RawSegmentDataset(train_segments),
        batch_size=args.batch_size_segments,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collate_segments,
        num_workers=0,
    )
    val_loader = DataLoader(
        RawSegmentDataset(val_segments),
        batch_size=args.batch_size_segments,
        shuffle=False,
        collate_fn=collate_segments,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score, best_epoch, stale, best_state = -1.0, 0, 0, None
    history = []
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            moved = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            stage_logits = model.forward_all(
                moved["global_tokens"], moved["local_tokens"], moved["a_logits"], moved["valid"]
            )
            target, valid = moved["labels"], moved["valid"]
            stage_losses = [
                F.cross_entropy(logits[valid].float(), target[valid])
                for logits in stage_logits
            ]
            loss = torch.stack(stage_losses).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"{model_name} epoch {epoch}: nonfinite loss")
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(f"{model_name} epoch {epoch}: nonfinite gradient norm")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_summary, _ = _evaluate(model, val_loader, device)
        score = float(val_summary["macro_f1"])
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "mean_train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_macro_f1": score,
            "val_accuracy": float(val_summary["accuracy"]),
            "best_macro_f1": best_score,
        }
        history.append(row)
        print(
            f"[{model_name}] epoch {epoch}/{args.epochs} "
            f"loss={row['mean_train_loss']:.6f} val_macro_f1={score:.6f} best={best_score:.6f}",
            flush=True,
        )
        _write_json(model_dir / "history.json", {"epochs": history})
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError(f"{model_name}: no best checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    final_summary, private_rows = _evaluate(model, val_loader, device)
    torch.save(
        {
            "schema": "fsn-temporal-head-benchmark-v1",
            "model_name": model_name,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_macro_f1": best_score,
            "model_config": {
                "global_dim": index.global_dim,
                "local_dim": index.local_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "tcn_dilations": list(args.tcn_dilations),
                "transformer_layers": args.transformer_layers,
                "transformer_heads": args.transformer_heads,
                "local_radius": args.local_radius,
                "max_sequence_length": args.max_sequence_length,
            },
            "state_dict": best_state,
        },
        model_dir / "best.pt",
    )
    _write_jsonl(model_dir / "private_val_predictions.jsonl", private_rows)
    result = {
        "model_name": model_name,
        "training_completed": True,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_macro_f1": best_score,
        "parameter_count": model.parameter_count,
        "elapsed_seconds": time.perf_counter() - start,
        "metrics": final_summary,
    }
    _write_json(model_dir / "public_safe_summary.json", result)
    return result


def run(args):
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if any(getattr(args, name) < 1 for name in (
        "epochs", "patience", "batch_size_segments", "hidden_dim",
        "transformer_layers", "transformer_heads", "max_sequence_length"
    )):
        raise ValueError("training and model dimensions must be positive")
    if args.lr <= 0 or not math.isfinite(args.lr) or args.weight_decay < 0 or not math.isfinite(args.weight_decay):
        raise ValueError("invalid optimizer settings")
    if args.grad_clip_norm <= 0 or not math.isfinite(args.grad_clip_norm):
        raise ValueError("grad_clip_norm must be positive and finite")
    if not math.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0,1)")
    if not math.isfinite(args.markov_strength) or args.markov_strength < 0:
        raise ValueError("markov_strength must be nonnegative and finite")
    if len(args.tcn_dilations) < 1 or any(value < 1 for value in args.tcn_dilations):
        raise ValueError("--tcn-dilations needs positive comma-separated integers")
    if args.hidden_dim % args.transformer_heads != 0:
        raise ValueError("hidden dim must be divisible by transformer heads")

    model_names = tuple(item.strip().lower() for item in args.models.split(",") if item.strip())
    if not model_names or len(set(model_names)) != len(model_names) or not set(model_names) <= set(VARIANTS):
        raise ValueError(f"--models must be unique choices from {VARIANTS}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    index_path, source_dir = Path(args.index).expanduser().resolve(), Path(args.source_protocol_dir).expanduser().resolve()
    index, metadata, split_segments, audit, seals = _preflight(index_path, source_dir)
    repo = Path(__file__).resolve().parents[2]
    output = _safe_output(Path(args.output), (repo, index_path.parent, source_dir, source_dir.parent))
    train_segments = _materialize_raw_segments(index, metadata, split_segments["train"])
    val_segments = _materialize_raw_segments(index, metadata, split_segments["val"])
    del split_segments
    transition_log_probs = (
        estimate_transition_log_probs(train_segments, smoothing=args.markov_smoothing)
        if "markov_viterbi" in model_names else None
    )

    output.mkdir(parents=False)
    protocol = {
        "schema": "fsn-temporal-head-benchmark-v1",
        "status": "running",
        "data_protocol": "fsn-full-7372-823-development-v1",
        "counts": {"train": 7372, "val": 823},
        "validation_role": "val823_reused_development_validation_not_independent_test",
        "test_read": False,
        "A_frozen": True,
        "source_policy_is_weak_adjacency_metadata_not_verified_continuity": True,
        "input_paths": {"index": str(index_path), "source_protocol_dir": str(source_dir)},
        "seals": seals,
        "model_order": list(model_names),
        "training": {
            "seed": args.seed,
            "max_epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "batch_size_segments": args.batch_size_segments,
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "selection": "full seven-class val823 Macro-F1; earliest checkpoint on ties",
            "tcn_dilations": list(args.tcn_dilations),
            "transformer_layers": args.transformer_layers,
            "transformer_heads": args.transformer_heads,
            "local_attention_radius_clips": args.local_radius,
            "markov_smoothing": args.markov_smoothing,
            "markov_transition_strength": args.markov_strength,
            "markov_fit_data": "train-only labels on source-protocol-eligible chains",
        },
    }
    _write_json(output / "private_protocol.json", protocol)
    _write_json(output / "status.json", {"status": "running", "completed_models": []})
    results = {}
    val_loader = DataLoader(
        RawSegmentDataset(val_segments),
        batch_size=args.batch_size_segments,
        shuffle=False,
        collate_fn=collate_segments,
        num_workers=0,
    )
    try:
        for model_name in model_names:
            print(f"starting temporal head {model_name}", flush=True)
            model_dir = output / model_name
            model_dir.mkdir()
            if model_name == "a_only":
                metrics, private_rows = _evaluate(None, val_loader, torch.device("cpu"))
                result = {
                    "model_name": model_name,
                    "training_completed": False,
                    "parameter_count": 0,
                    "metrics": metrics,
                }
                _write_jsonl(model_dir / "private_val_predictions.jsonl", private_rows)
                _write_json(model_dir / "public_safe_summary.json", result)
            elif model_name == "markov_viterbi":
                decoder = MarkovViterbiDecoder(
                    transition_log_probs, transition_strength=args.markov_strength
                ).to(device).eval()
                metrics, private_rows = _evaluate(decoder, val_loader, device)
                result = {
                    "model_name": model_name,
                    "training_completed": False,
                    "parameter_count": 0,
                    "transition_smoothing": args.markov_smoothing,
                    "transition_strength": args.markov_strength,
                    "transition_estimation": "train labels only on eligible chains",
                    "metrics": metrics,
                }
                _write_jsonl(model_dir / "private_val_predictions.jsonl", private_rows)
                _write_json(model_dir / "public_safe_summary.json", result)
            else:
                result = _train_model(
                    model_name, index, train_segments, val_segments, args, device, model_dir
                )
            results[model_name] = result
            _write_json(output / "status.json", {
                "status": "running",
                "completed_models": list(results),
            })

        baseline_f1 = results["a_only"]["metrics"]["macro_f1"] if "a_only" in results else None
        comparison = {
            name: {
                "macro_f1": row["metrics"]["macro_f1"],
                "accuracy": row["metrics"]["accuracy"],
                "macro_f1_delta_vs_A": (
                    None if baseline_f1 is None else row["metrics"]["macro_f1"] - baseline_f1
                ),
                "parameters": row.get("parameter_count", 0),
            }
            for name, row in results.items()
        }
        public = {
            "schema": "fsn-temporal-head-benchmark-v1",
            "status": "completed",
            "data_protocol": "fsn-full-7372-823-development-v1",
            "counts": {"train": 7372, "val": 823},
            "validation_role": "val823_reused_development_validation_not_independent_test",
            "test_read": False,
            "frozen_feature_checkpoint_sha256": index.checkpoint_sha256,
            "source_policy_is_weak_adjacency_metadata_not_verified_continuity": True,
            "model_definitions": {
                "a_only": "original frozen VideoMAE logits",
                "markov_viterbi": "first-order Viterbi decoder with transition probabilities estimated from train-only eligible chains",
                "mlp": "per-clip residual MLP with no neighbor access",
                "bigru": "two-layer bidirectional GRU over candidate-clip sequence",
                "tcn": "current residual dilated TCN baseline",
                "mstcnpp_adapted": "three-stage TCN-inspired per-clip prediction refinement; clip-label adaptation, not official dense MS-TCN++",
                "local_transformer": "two-layer Transformer encoder with local temporal attention mask; ASFormer-inspired clip-label adaptation",
            },
            "comparison": comparison,
            "models": results,
            "interpretation_warning": (
                "single-seed development experiment; compare only within this frozen VideoMAE feature protocol; "
                "no independent test or statistical significance claim"
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
    parser.add_argument("--index", required=True, help="sealed VideoMAE feature index.jsonl")
    parser.add_argument("--source-protocol-dir", required=True, help="exact source protocol directory")
    parser.add_argument("--output", required=True, help="fresh private output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", default="a_only,markov_viterbi,mlp,bigru,tcn,mstcnpp_adapted,local_transformer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size-segments", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tcn-dilations", type=lambda value: tuple(int(item) for item in value.split(",")),
                        default=(1, 2, 4))
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--local-radius", type=int, default=4)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--markov-smoothing", type=float, default=1.0)
    parser.add_argument("--markov-strength", type=float, default=0.25)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary["comparison"], indent=2, ensure_ascii=False, allow_nan=False), flush=True)
