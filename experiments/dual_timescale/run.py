"""Train and compare frozen-VideoMAE dual-timescale ablations M0, M1S, M1, M2, M2S and M3.

M0: original A logits; no learned parameters.
M1: within-clip multi-scale feature pooling only.
M2: inter-clip TCN only, on the existing mean-pooled frozen evidence.
M3: within-clip multi-scale branch plus inter-clip TCN.

Only the fixed train7372/val823 development protocol is used. No test split is
read or generated. Outputs include private per-clip artifacts and a separate
identifier-free aggregate summary; keep the output directory private.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from experiments.aligned_protocol import DATA_PROTOCOL, EXPECTED_COUNTS, EXPECTED_MANIFEST_SHA
from experiments.dual_timescale.model import DualTimescaleModel
from experiments.relation.data import load_edges, load_feature_index, read_jsonl, sha256_file
from experiments.relation.evaluate_suite import _directions, _paired, classification_metrics
from experiments.relation.protocol import SOURCE_POLICY, validate_source_artifacts
from experiments.relation.tcn_baseline import build_segments


CLASS_NAMES = ["消毒", "进针", "运针", "扫散", "再灌注", "拔针", "固定"]
VARIANTS = ("M0", "M1S", "M1", "M2", "M2S", "M3")


@dataclass
class RawSegment:
    clip_ids: list[str]
    global_tokens: np.ndarray
    local_tokens: np.ndarray
    a_logits: np.ndarray
    labels: np.ndarray
    source_kinds: list[str]
    durations: np.ndarray


class RawSegmentDataset(Dataset):
    def __init__(self, segments: list[RawSegment]):
        self.segments = segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        return self.segments[index]


def collate_segments(segments: list[RawSegment]) -> dict:
    if not segments:
        raise ValueError("cannot collate an empty segment batch")
    batch_size = len(segments)
    max_length = max(len(segment.clip_ids) for segment in segments)
    positions = segments[0].global_tokens.shape[1]
    global_dim = segments[0].global_tokens.shape[2]
    local_dim = segments[0].local_tokens.shape[2]
    global_tokens = torch.zeros(batch_size, max_length, positions, global_dim, dtype=torch.float32)
    local_tokens = torch.zeros(batch_size, max_length, positions, local_dim, dtype=torch.float32)
    a_logits = torch.zeros(batch_size, max_length, 7, dtype=torch.float32)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    valid = torch.zeros(batch_size, max_length, dtype=torch.bool)
    for row, segment in enumerate(segments):
        length = len(segment.clip_ids)
        global_tokens[row, :length] = torch.from_numpy(segment.global_tokens)
        local_tokens[row, :length] = torch.from_numpy(segment.local_tokens)
        a_logits[row, :length] = torch.from_numpy(segment.a_logits)
        labels[row, :length] = torch.from_numpy(segment.labels)
        valid[row, :length] = True
    return {
        "global_tokens": global_tokens,
        "local_tokens": local_tokens,
        "a_logits": a_logits,
        "labels": labels,
        "valid": valid,
        "segments": segments,
    }


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _fresh_output(path: Path, protected: tuple[Path, ...]) -> Path:
    path = path.expanduser().resolve()
    if path.exists() or not path.parent.is_dir():
        raise FileExistsError("output must be a fresh directory whose parent already exists")
    for root in protected:
        root = root.resolve()
        if path == root or path.is_relative_to(root):
            raise ValueError(f"output must be outside protected input/source roots: {root.name}")
    return path


def _preflight(index_path: Path, source_dir: Path):
    index_path, source_dir = index_path.resolve(), source_dir.resolve()
    index = load_feature_index(index_path)
    counts = {
        split: sum(row["split"] == split for row in index.clips.values())
        for split in ("train", "val")
    }
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"only the sealed full protocol is allowed: expected {EXPECTED_COUNTS}, got {counts}")
    if index.shapes.get("global_tokens") != (8, 768) or index.shapes.get("local_tokens") != (8, 768):
        raise ValueError("expected eight frozen ViT-B/16 tubelet tokens with 768 channels")
    export_protocol = _read_json(index_path.parent / "protocol.json")
    if (
        export_protocol.get("status") != "complete"
        or export_protocol.get("A_frozen") is not True
        or export_protocol.get("index_sha256") != sha256_file(index_path)
        or export_protocol.get("checkpoint_sha256") != index.checkpoint_sha256
        or export_protocol.get("source_audit_sha256") != sha256_file(source_dir / "audit.json")
        or export_protocol.get("data_protocol") != DATA_PROTOCOL
        or export_protocol.get("split_counts") != EXPECTED_COUNTS
        or export_protocol.get("train_manifest_sha256") != EXPECTED_MANIFEST_SHA["train"]
        or export_protocol.get("val_manifest_sha256") != EXPECTED_MANIFEST_SHA["val"]
        or export_protocol.get("test_read") is not False
    ):
        raise ValueError("frozen VideoMAE feature export does not match the sealed protocol")

    edges_path = source_dir / "edges.jsonl"
    audit = validate_source_artifacts(source_dir, index, edges_path)
    if audit.get("label_policy") != SOURCE_POLICY:
        raise ValueError("dual-timescale suite requires the exact existing source-protocol labels")
    if audit.get("manifest_sha256") != EXPECTED_MANIFEST_SHA:
        raise ValueError("source protocol manifests differ from the frozen train/validation manifests")

    suite_root = source_dir.parent
    prior_seal_path = suite_root / "evaluation" / "private_protocol.json"
    prior_summary_path = suite_root / "evaluation" / "public_safe_summary.json"
    if not prior_seal_path.is_file() or not prior_summary_path.is_file():
        raise FileNotFoundError("completed sealed source-protocol evaluation artifacts are required")
    prior_seal, prior_summary = _read_json(prior_seal_path), _read_json(prior_summary_path)
    expected_prior = {
        "index_sha256": sha256_file(index_path),
        "source_audit_sha256": sha256_file(source_dir / "audit.json"),
        "checkpoint_sha256": index.checkpoint_sha256,
        "chains_sha256": sha256_file(source_dir / "chains_val.jsonl"),
    }
    if any(prior_seal.get(key) != value for key, value in expected_prior.items()):
        raise ValueError("dual-timescale input differs from the completed frozen-A suite")

    metadata_rows = read_jsonl(source_dir / "manifest_metadata.jsonl")
    metadata = {row["clip_id"]: row for row in metadata_rows}
    if metadata.keys() != index.clips.keys():
        raise ValueError("source metadata must cover every indexed clip exactly once")
    for clip_id, row in metadata.items():
        if row.get("source_kind") not in {"clinical", "network"}:
            raise ValueError(f"{clip_id}: source_kind must be clinical or network")
        duration = row.get("duration")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"{clip_id}: duration must be a positive finite number")
        if row.get("label_id") != index.clips[clip_id].get("label_id"):
            raise ValueError(f"{clip_id}: action label differs from the frozen feature index")

    edges = load_edges(edges_path, index)
    split_segments = {}
    for split in ("train", "val"):
        chains = read_jsonl(source_dir / f"chains_{split}.jsonl")
        # Reuse the original, strict structural-cut validation. Every clip is
        # partitioned once and temporal heads never cross an ineligible edge.
        split_segments[split] = build_segments(index, metadata_rows, chains, edges, split)
        covered = sum(len(segment.clip_ids) for segment in split_segments[split])
        if covered != counts[split]:
            raise ValueError(f"{split} chains cover {covered}, expected {counts[split]} clips")

    # The existing sealed suite contains the original A-only validation metrics.
    # Only val replay can be compared with those metrics; train is not compared
    # against validation statistics.
    labels = np.concatenate([segment.labels for segment in split_segments["val"]])
    predictions = np.concatenate([segment.a_logits.argmax(1) for segment in split_segments["val"]])
    replay = classification_metrics(labels, predictions)
    expected_a = prior_summary.get("variants", {}).get("A_only", {})
    for key in ("accuracy", "macro_f1", "confusion"):
        if replay[key] != expected_a.get(key):
            raise ValueError(f"frozen A val replay differs from sealed source suite: {key}")

    return index, metadata, split_segments, audit, expected_prior


def _materialize_raw_segments(index, metadata, segments) -> list[RawSegment]:
    materialized = []
    for segment in segments:
        arrays = [index.read(clip_id) for clip_id in segment.clip_ids]
        global_tokens = np.stack([item["global_tokens"] for item in arrays]).astype(np.float32, copy=False)
        local_tokens = np.stack([item["local_tokens"] for item in arrays]).astype(np.float32, copy=False)
        a_logits = np.stack([item["logits"] for item in arrays]).astype(np.float32, copy=False)
        labels = np.asarray(segment.labels, dtype=np.int64)
        sources = [metadata[clip_id]["source_kind"] for clip_id in segment.clip_ids]
        durations = np.asarray([metadata[clip_id]["duration"] for clip_id in segment.clip_ids], dtype=np.float32)
        if not (
            global_tokens.shape[0] == local_tokens.shape[0] == len(segment.clip_ids)
            and a_logits.shape == (len(segment.clip_ids), 7)
            and labels.shape == (len(segment.clip_ids),)
            and np.isfinite(global_tokens).all()
            and np.isfinite(local_tokens).all()
            and np.isfinite(a_logits).all()
            and np.isfinite(durations).all()
        ):
            raise ValueError("raw frozen clip evidence has inconsistent shapes or nonfinite values")
        materialized.append(RawSegment(
            list(segment.clip_ids), global_tokens, local_tokens, a_logits, labels, sources, durations
        ))
    return materialized


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("global_tokens", "local_tokens", "a_logits", "labels", "valid")
    }


def _predict(model, batch: dict, device: torch.device) -> torch.Tensor:
    moved = _to_device(batch, device)
    if model is None:
        return moved["a_logits"]
    return model(
        moved["global_tokens"], moved["local_tokens"], moved["a_logits"], moved["valid"]
    )


@torch.no_grad()
def _evaluate(model, loader, device: torch.device):
    if model is not None:
        model.eval()
    labels, predictions, a_predictions, sources, durations = [], [], [], [], []
    private_rows = []
    for batch in loader:
        output = _predict(model, batch, device).detach().cpu().numpy()
        a_logits = batch["a_logits"].numpy()
        batch_labels = batch["labels"].numpy()
        valid = batch["valid"].numpy()
        for row, segment in enumerate(batch["segments"]):
            length = len(segment.clip_ids)
            pred = output[row, :length].argmax(axis=-1).astype(int)
            a_pred = a_logits[row, :length].argmax(axis=-1).astype(int)
            truth = batch_labels[row, :length].astype(int)
            labels.extend(truth.tolist())
            predictions.extend(pred.tolist())
            a_predictions.extend(a_pred.tolist())
            sources.extend(segment.source_kinds)
            durations.extend(segment.durations.tolist())
            for idx, clip_id in enumerate(segment.clip_ids):
                private_rows.append({
                    "clip_id": clip_id,
                    "label_id": int(truth[idx]),
                    "a_prediction": int(a_pred[idx]),
                    "prediction": int(pred[idx]),
                    "source_kind": segment.source_kinds[idx],
                    "duration_seconds": float(segment.durations[idx]),
                })
            if not valid[row, :length].all():
                raise AssertionError("valid mask unexpectedly excludes a real clip")
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    a_predictions = np.asarray(a_predictions, dtype=np.int64)
    sources = np.asarray(sources)
    durations = np.asarray(durations, dtype=np.float64)
    if labels.shape != (EXPECTED_COUNTS["val"],):
        raise ValueError(f"evaluation must cover exactly {EXPECTED_COUNTS['val']} val clips")
    summary = classification_metrics(labels, predictions)
    summary.update(
        paired_vs_A=_paired(labels, predictions, a_predictions),
        sweep_reperfusion=_directions(labels, predictions),
        class_names=CLASS_NAMES,
    )

    def slice_metrics(mask):
        mask = np.asarray(mask, dtype=bool)
        result = classification_metrics(labels[mask], predictions[mask])
        present = [item["f1"] for item in result["per_class"] if item["support"] > 0]
        result["present_class_macro_f1"] = float(np.mean(present)) if present else 0.0
        result["paired_vs_A"] = _paired(labels[mask], predictions[mask], a_predictions[mask])
        return result

    summary["slices"] = {
        "clinical": slice_metrics(sources == "clinical"),
        "network": slice_metrics(sources == "network"),
        "duration_lt_5s": slice_metrics(durations < 5.0),
        "duration_5_to_lt_15s": slice_metrics((durations >= 5.0) & (durations < 15.0)),
        "duration_ge_15s": slice_metrics(durations >= 15.0),
    }
    return summary, private_rows


def _build_model(variant: str, index, args) -> nn.Module:
    dilations = tuple(args.dilations)
    return DualTimescaleModel(
        variant=variant,
        global_dim=index.global_dim,
        local_dim=index.local_dim,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        blocks=args.blocks,
        dropout=args.dropout,
        dilations=dilations,
    )


def _train_variant(variant, index, train_segments, val_segments, args, device, variant_dir):
    _seed_everything(args.seed)
    model = _build_model(variant, index, args).to(device)
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
    best_f1, best_epoch, best_state, stale = -1.0, 0, None, 0
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            moved = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                moved["global_tokens"], moved["local_tokens"], moved["a_logits"], moved["valid"]
            )
            loss = F.cross_entropy(logits[moved["valid"]], moved["labels"][moved["valid"]])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"{variant} epoch {epoch}: nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_summary, _ = _evaluate(model, val_loader, device)
        score = float(val_summary["macro_f1"])
        history.append({
            "epoch": epoch,
            "mean_train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_macro_f1": score,
            "val_accuracy": float(val_summary["accuracy"]),
        })
        # Strict > preserves the earliest checkpoint on ties.
        if score > best_f1:
            best_f1, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        print(f"[{variant}] epoch {epoch}/{args.epochs}: train_loss={float(np.mean(losses)) if losses else 0.0:.6f}; "
              f"val_macro_f1={score:.6f}; best={best_f1:.6f}", flush=True)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError(f"{variant}: no valid checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    final_summary, private_rows = _evaluate(
        model,
        DataLoader(RawSegmentDataset(val_segments), batch_size=args.batch_size_segments,
                   shuffle=False, collate_fn=collate_segments, num_workers=0),
        device,
    )
    checkpoint_path = variant_dir / "best.pt"
    torch.save({
        "schema": "fsn-dual-timescale-v1",
        "variant": variant,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "model_config": {
            "global_dim": index.global_dim,
            "local_dim": index.local_dim,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "blocks": args.blocks,
            "dropout": args.dropout,
            "dilations": list(args.dilations),
        },
        "state_dict": best_state,
    }, checkpoint_path)
    (variant_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(variant_dir / "private_val_predictions.jsonl", private_rows)
    elapsed = time.perf_counter() - start_time
    result = {
        "variant": variant,
        "training_completed": True,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": elapsed,
        "selected_checkpoint": "best.pt",
        "metrics": final_summary,
    }
    _write_json(variant_dir / "public_safe_summary.json", result)
    return result


def run(args) -> dict:
    index_path = Path(args.index).expanduser().resolve()
    source_dir = Path(args.source_protocol_dir).expanduser().resolve()
    index, metadata, split_segments, audit, seals = _preflight(index_path, source_dir)
    variants = tuple(item.strip().upper() for item in args.variants.split(",") if item.strip())
    if not variants or len(set(variants)) != len(variants) or not set(variants) <= set(VARIANTS):
        raise ValueError(f"--variants must be a comma-separated unique subset of {VARIANTS}")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if any(getattr(args, name) < 1 for name in ("epochs", "patience", "batch_size_segments", "embedding_dim", "hidden_dim", "blocks")):
        raise ValueError("epochs, patience, batch size, embedding dim, hidden dim and blocks must be positive")
    if not math.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        raise ValueError("dropout must be finite and in [0,1)")
    if args.grad_clip_norm <= 0 or not math.isfinite(args.grad_clip_norm):
        raise ValueError("grad_clip_norm must be positive and finite")
    if args.lr <= 0 or not math.isfinite(args.lr) or args.weight_decay < 0 or not math.isfinite(args.weight_decay):
        raise ValueError("invalid AdamW hyperparameters")
    if len(args.dilations) != args.blocks or any(value < 1 for value in args.dilations):
        raise ValueError("--dilations must contain one positive integer per context block")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    suite_root = source_dir.parent
    repo = Path(__file__).resolve().parents[2]
    output = _fresh_output(Path(args.output), (
        repo, index_path.parent, source_dir, suite_root,
    ))
    # Raw token evidence is loaded only after every seal and the original A
    # logits have been checked. No raw identifiers are written into Git.
    train_segments = _materialize_raw_segments(index, metadata, split_segments["train"])
    val_segments = _materialize_raw_segments(index, metadata, split_segments["val"])
    del split_segments

    output.mkdir()
    private_protocol = {
        "schema": "fsn-dual-timescale-v1",
        "status": "running",
        "data_protocol": DATA_PROTOCOL,
        "counts": EXPECTED_COUNTS,
        "validation_role": "val823_reused_development_validation_not_independent_test",
        "test_read": False,
        "A_frozen": True,
        "input_paths": {
            "index": str(index_path),
            "source_protocol_dir": str(source_dir),
        },
        "seals": {
            "index_sha256": sha256_file(index_path),
            "source_audit_sha256": sha256_file(source_dir / "audit.json"),
            "checkpoint_sha256": index.checkpoint_sha256,
            "manifest_sha256": audit.get("manifest_sha256"),
            **seals,
        },
        "experiment": {
            "variants": list(variants),
            "seed": args.seed,
            "max_epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "batch_size_segments": args.batch_size_segments,
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "selection_metric": "full seven-class val823 Macro-F1; earliest epoch on ties",
            "dilations": list(args.dilations),
            "execution_mode": "explicit_cli_run; no automatic resume or retry",
        },
    }
    _write_json(output / "private_protocol.json", private_protocol)
    _write_json(output / "status.json", {"status": "running", "completed_variants": []})
    summaries = {}
    val_loader = DataLoader(
        RawSegmentDataset(val_segments),
        batch_size=args.batch_size_segments,
        shuffle=False,
        collate_fn=collate_segments,
        num_workers=0,
    )
    try:
        for variant in variants:
            print(f"starting variant {variant}", flush=True)
            variant_dir = output / variant
            variant_dir.mkdir()
            if variant == "M0":
                baseline_summary, private_rows = _evaluate(None, val_loader, torch.device("cpu"))
                result = {
                    "variant": "M0",
                    "training_completed": False,
                    "parameters": 0,
                    "metrics": baseline_summary,
                }
                _write_jsonl(variant_dir / "private_val_predictions.jsonl", private_rows)
                _write_json(variant_dir / "public_safe_summary.json", result)
            else:
                result = _train_variant(
                    variant, index, train_segments, val_segments, args, device, variant_dir
                )
            summaries[variant] = result
            _write_json(output / "status.json", {
                "status": "running",
                "completed_variants": list(summaries),
            })

        baseline_f1 = summaries.get("M0", {}).get("metrics", {}).get("macro_f1")
        comparison = {}
        for variant, result in summaries.items():
            score = result["metrics"]["macro_f1"]
            comparison[variant] = {
                "macro_f1": score,
                "accuracy": result["metrics"]["accuracy"],
                "macro_f1_delta_vs_M0": None if baseline_f1 is None else score - baseline_f1,
            }
        contrast_specs = {
            "multiscale_without_context_M1_minus_M1S": ("M1", "M1S"),
            "multiscale_with_context_M3_minus_M2S": ("M3", "M2S"),
            "context_on_A_M2_minus_M0": ("M2", "M0"),
            "context_with_single_scale_M2S_minus_M1S": ("M2S", "M1S"),
            "context_with_multiscale_M3_minus_M1": ("M3", "M1"),
        }
        factorial_contrasts = {}
        for name, (left, right) in contrast_specs.items():
            if left in summaries and right in summaries:
                factorial_contrasts[name] = {
                    "left": left,
                    "right": right,
                    "macro_f1_difference": (
                        summaries[left]["metrics"]["macro_f1"] - summaries[right]["metrics"]["macro_f1"]
                    ),
                    "accuracy_difference": (
                        summaries[left]["metrics"]["accuracy"] - summaries[right]["metrics"]["accuracy"]
                    ),
                }
        public_summary = {
            "schema": "fsn-dual-timescale-v1",
            "status": "completed",
            "data_protocol": DATA_PROTOCOL,
            "counts": EXPECTED_COUNTS,
            "validation_role": "val823_reused_development_validation_not_independent_test",
            "test_read": False,
            "A_frozen": True,
            "variant_definitions": {
                "M0": "frozen VideoMAE A only",
                "M1S": "single-scale mean-pooling residual clip head; parameter-matched to M1",
                "M1": "clip-internal 1/2/4 temporal-pyramid pooling and residual clip correction",
                "M2": "inter-clip TCN on current mean-pooled global/local frozen features and A logits",
                "M2S": "single-scale mean-pooling clip head plus inter-clip TCN; parameter-matched to M3",
                "M3": "joint clip-internal multi-scale correction plus inter-clip TCN",
            },
            "experiment": private_protocol["experiment"],
            "comparison": comparison,
            "factorial_contrasts": factorial_contrasts,
            "variants": summaries,
            "interpretation_warning": (
                "single-seed development experiment only; source-rule adjacency is weak metadata policy, "
                "not ground-truth continuity; no independent test or statistical significance claim"
            ),
        }
        _write_json(output / "public_safe_summary.json", public_summary)
        private_protocol["status"] = "completed"
        _write_json(output / "private_protocol.json", private_protocol)
        _write_json(output / "status.json", {
            "status": "completed",
            "completed_variants": list(summaries),
        })
        return public_summary
    except Exception as error:
        _write_json(output / "status.json", {
            "status": "failed",
            "completed_variants": list(summaries),
            "error_type": type(error).__name__,
            "error": str(error),
        })
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="sealed VideoMAE frozen-feature index.jsonl")
    parser.add_argument("--source-protocol-dir", required=True, help="completed source-protocol directory")
    parser.add_argument("--output", required=True, help="fresh private output directory outside the repository")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", default="M0,M1S,M1,M2,M2S,M3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size-segments", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dilations", type=lambda value: tuple(int(item) for item in value.split(",")),
                        default=(1, 2, 4))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    result = run(args)
    print(json.dumps(result["comparison"], indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
