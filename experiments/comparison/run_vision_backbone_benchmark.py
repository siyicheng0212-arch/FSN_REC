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
from experiments.relation.data import load_feature_index, sha256_file


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



def _load_existing_a_reference(index_path: Path, source_protocol_dir: Path, datasets, data_audit):
    """Validate and score the current frozen VideoMAE A on these exact val clips."""
    index_path = index_path.expanduser().resolve()
    source_protocol_dir = source_protocol_dir.expanduser().resolve()
    index = load_feature_index(index_path)
    seal_path = index_path.parent / "protocol.json"
    audit_path = source_protocol_dir / "audit.json"
    if not seal_path.is_file() or not audit_path.is_file():
        raise ValueError("existing A reference requires its export seal and source-protocol audit")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if any(row.get("split") not in {"train", "val"} for row in index.clips.values()):
        raise ValueError("existing A index must contain only the sealed train/val clips; test rows are rejected")
    actual_split_counts = {
        split: sum(row["split"] == split for row in index.clips.values())
        for split in ("train", "val")
    }
    if (
        seal.get("status") != "complete"
        or seal.get("A_frozen") is not True
        or seal.get("index_sha256") != sha256_file(index_path)
        or seal.get("checkpoint_sha256") != index.checkpoint_sha256
        or seal.get("data_protocol") != video_data.DATA_PROTOCOL
        or seal.get("train_manifest_sha256") != data_audit["manifest_sha256"]["train"]
        or seal.get("val_manifest_sha256") != data_audit["manifest_sha256"]["val"]
        or seal.get("source_audit_sha256") != sha256_file(audit_path)
        or seal.get("test_read", False) is not False
        or actual_split_counts != data_audit["counts"]
    ):
        raise ValueError("existing VideoMAE A export does not match the sealed visual benchmark protocol")

    suite_root = source_protocol_dir.parent
    prior_seal_path = suite_root / "evaluation" / "private_protocol.json"
    prior_summary_path = suite_root / "evaluation" / "public_safe_summary.json"
    if not prior_seal_path.is_file() or not prior_summary_path.is_file():
        raise FileNotFoundError("completed sealed source-protocol evaluation artifacts are required")
    prior_seal = json.loads(prior_seal_path.read_text(encoding="utf-8"))
    prior_summary = json.loads(prior_summary_path.read_text(encoding="utf-8"))
    expected_prior = {
        "index_sha256": sha256_file(index_path),
        "source_audit_sha256": sha256_file(audit_path),
        "checkpoint_sha256": index.checkpoint_sha256,
        "chains_sha256": sha256_file(source_protocol_dir / "chains_val.jsonl"),
    }
    if any(prior_seal.get(key) != value for key, value in expected_prior.items()):
        raise ValueError("existing A index and source-protocol evaluation do not share exact seals")

    expected_ids = {
        split: {record.clip_id for record in dataset.base.records}
        for split, dataset in datasets.items()
    }
    indexed_ids = {
        split: {clip_id for clip_id, row in index.clips.items() if row["split"] == split}
        for split in ("train", "val")
    }
    if expected_ids != indexed_ids:
        raise ValueError("existing A index and raw-video benchmark do not cover identical train/val clips")

    for split, dataset in datasets.items():
        for record in dataset.base.records:
            row = index.clips[record.clip_id]
            if row.get("label_id") != record.label_id or row.get("group_id") != record.group_id:
                raise ValueError(f"{record.clip_id}: A feature metadata differs from the frozen manifest")
            if row.get("source_collection") != record.source_collection:
                raise ValueError(f"{record.clip_id}: A feature source metadata differs from the frozen manifest")
            duration = row.get("duration")
            if not isinstance(duration, (int, float)) or not math.isclose(
                float(duration), record.clip_duration_sec, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(f"{record.clip_id}: A feature duration differs from the frozen manifest")

    val_records = datasets["val"].base.records
    logits, targets, metadata, predictions = [], [], [], []
    for record in val_records:
        row = index.clips[record.clip_id]
        score = index.read(record.clip_id)["logits"]
        if score.shape != (7,) or not np.isfinite(score).all():
            raise ValueError(f"{record.clip_id}: invalid frozen A logits")
        prediction = int(np.asarray(score).argmax())
        logits.append(score.astype(np.float32, copy=False))
        targets.append(record.label_id)
        metadata.append({"source_collection": record.source_collection,
                         "duration": record.clip_duration_sec})
        predictions.append({
            "clip_id": record.clip_id,
            "target": int(record.label_id),
            "prediction": prediction,
            "source": record.source_collection,
            "duration": float(record.clip_duration_sec),
            "logits": [float(value) for value in score],
        })
    if len(predictions) != 823:
        raise ValueError("existing A reference must cover the complete val823")
    metrics = compute_classification_metrics(
        torch.as_tensor(np.stack(logits), dtype=torch.float32),
        torch.as_tensor(targets, dtype=torch.long),
        metadata,
    )
    sealed_metrics = prior_summary.get("variants", {}).get("A_only", {})
    if not isinstance(sealed_metrics, dict):
        raise ValueError("source-protocol evaluation has no A-only validation metrics")
    key_pairs = {
        "accuracy": "accuracy",
        "macro_f1": "macro_f1",
        "confusion": "confusion_matrix",
    }
    for sealed_key, local_key in key_pairs.items():
        actual_value = metrics["all"].get(local_key)
        expected_value = sealed_metrics.get(sealed_key)
        if local_key == "confusion_matrix":
            matches = actual_value == expected_value
        else:
            matches = (
                isinstance(expected_value, (int, float))
                and math.isclose(float(actual_value), float(expected_value), rel_tol=0.0, abs_tol=1e-12)
            )
        if not matches:
            raise ValueError(
                f"current A val823 replay differs from sealed source-protocol metrics: {sealed_key}"
            )
    return {
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
        "checkpoint_sha256": index.checkpoint_sha256,
        "metrics": metrics,
        "predictions": predictions,
        "parameter_count": None,
        "validation_inference_seconds": None,
        "validation_clips_per_second": None,
        "peak_cuda_allocated_bytes": None,
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
    enabled = False
    if device.type == "cuda":
        with torch.cuda.device(device):
            enabled = torch.cuda.is_bf16_supported()
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
    durations = np.asarray([float(row["duration"]) for row in meta_rows], dtype=np.float64)
    duration_masks = {
        "duration_lt_5s": durations < 5.0,
        "duration_5_to_lt_15s": (durations >= 5.0) & (durations < 15.0),
        "duration_ge_15s": durations >= 15.0,
    }
    metrics["duration_bins"] = {}
    for name, mask in duration_masks.items():
        if not bool(mask.any()):
            metrics["duration_bins"][name] = {
                "support": 0,
                "metrics": None,
                "present_class_macro_f1": None,
            }
            continue
        sliced = compute_classification_metrics(
            scores[torch.as_tensor(mask)],
            targets[torch.as_tensor(mask)],
            [row for row, keep in zip(meta_rows, mask.tolist()) if keep],
        )
        active = [row["f1"] for row in sliced["all"]["per_class"] if row["support"] > 0]
        metrics["duration_bins"][name] = {
            "support": int(mask.sum()),
            "metrics": sliced["all"],
            "present_class_macro_f1": float(np.mean(active)) if active else 0.0,
        }
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
    weighted_ce_sum = 0.0
    weight_mass_sum = 0.0
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
            weighted_ce_sum += float(numerator.detach().cpu())
            samples += len(target)

        grad_norm = nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            args.grad_clip_norm,
        )
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError("nonfinite gradient norm")
        optimizer.step()
        weight_mass_sum += window_weight_mass
        steps += 1

    if samples != len(loader.dataset):
        raise ValueError(f"train epoch visited {samples} clips, expected {len(loader.dataset)}")
    return {
        "samples_seen": samples,
        "optimizer_steps": steps,
        "train_weighted_ce": weighted_ce_sum / max(weight_mass_sum, 1e-12),
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
    model, model_audit = build_backbone(
        name,
        pretrained=not args.random_init,
        vjepa2_repo=args.vjepa2_repo,
        vjepa2_checkpoint=args.vjepa2_checkpoint,
        internvideo_next_model_id=args.internvideo_next_model_id,
        internvideo_next_revision=args.internvideo_next_revision,
    )
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
    foundation_names = {"vjepa2_vitl", "internvideo_next_base"}
    selected_foundations = foundation_names.intersection(names)
    if selected_foundations and args.mode != "linear_probe":
        raise ValueError(
            "V-JEPA 2 and InternVideo-Next are currently supported only in --mode linear_probe; "
            "their upstream full-finetuning recipes are not implemented in this benchmark."
        )
    if selected_foundations and args.random_init:
        raise ValueError("foundation encoders require their published pretrained weights")
    if "vjepa2_vitl" in selected_foundations:
        if args.vjepa2_repo is None or args.vjepa2_checkpoint is None:
            raise ValueError(
                "vjepa2_vitl requires --vjepa2-repo and --vjepa2-checkpoint "
                "for the official V-JEPA2 repository checkout and ViT-L checkpoint."
            )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    manifest_dir, cache_dir = Path(args.manifest_dir).expanduser().resolve(), Path(args.cache_dir).expanduser().resolve()
    # This preflight verifies the immutable manifests, every cache entry, split
    # and class coverage before any output directory or model is created.
    datasets, data_audit = video_data.audit_frozen_data(manifest_dir, cache_dir)
    a_reference = _load_existing_a_reference(
        Path(args.existing_a_index), Path(args.reference_source_protocol_dir), datasets, data_audit
    )
    repo = Path(__file__).resolve().parents[2]
    a_index_root = Path(args.existing_a_index).expanduser().resolve().parent
    reference_source_dir = Path(args.reference_source_protocol_dir).expanduser().resolve()
    output = _safe_output(
        Path(args.output), (repo, manifest_dir, cache_dir, a_index_root, reference_source_dir)
    )
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
        "model_order": ["videomae_v2_A_reference", *names],
        "initialization": (
            "existing sealed fine-tuned VideoMAE A reference; Kinetics-400 supervised torchvision "
            "weights for mvit_v2_s/swin3d_t/r2plus1d_18; frozen upstream checkpoint plus a new "
            "seven-class linear head for vjepa2_vitl/internvideo_next_base"
        ),
        "existing_A_reference": {
            "index_sha256": a_reference["index_sha256"],
            "checkpoint_sha256": a_reference["checkpoint_sha256"],
            "A_frozen": True,
        },
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
    _write_json(output / "status.json", {"status": "running", "completed_models": ["videomae_v2_A_reference"]})
    a_dir = output / "videomae_v2_A_reference"
    a_dir.mkdir()
    _write_jsonl(a_dir / "private_val_predictions.jsonl", a_reference["predictions"])
    a_row = {
        "backbone": "videomae_v2_A_reference",
        "mode": "existing_finetuned_frozen_reference",
        "pretrained": True,
        "best_epoch": None,
        "epochs_completed": 0,
        "best_val_macro_f1": float(a_reference["metrics"]["all"]["macro_f1"]),
        "parameters_total": None,
        "parameters_trainable": 0,
        "optimizer_groups": [],
        "training_seconds_total": 0.0,
        "validation": a_reference["metrics"],
        "validation_inference_seconds": None,
        "validation_clips_per_second": None,
        "peak_cuda_allocated_bytes": None,
        "peak_cuda_reserved_bytes": None,
        "model_audit": {
            "source": "existing sealed VideoMAE A feature export",
            "index_sha256": a_reference["index_sha256"],
            "checkpoint_sha256": a_reference["checkpoint_sha256"],
        },
    }
    _write_json(a_dir / "public_safe_summary.json", a_row)
    results = {"videomae_v2_A_reference": a_row}
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
                    "peak_cuda_allocated_bytes": row.get("peak_cuda_allocated_bytes"),
                }
                for name, row in results.items()
            },
            "interpretation_warning": (
                "single-seed development results; do not call val823 an independent test; "
                "Kinetics-pretrained and self-supervised foundation encoders differ in pretraining "
                "recipe; foundation models are frozen linear-probe screens, not full fine-tuning; "
                "preprocessing is model-specific while using the same 16 temporal sample positions"
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
    parser.add_argument("--existing-a-index", type=Path, required=True,
                        help="sealed current VideoMAE A index.jsonl for same-protocol reference")
    parser.add_argument("--reference-source-protocol-dir", type=Path, required=True,
                        help="the source-protocol directory hashed by the A export seal")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", default="mvit_v2_s,swin3d_t,r2plus1d_18")
    parser.add_argument("--mode", choices=("linear_probe", "finetune"), default="linear_probe")
    parser.add_argument(
        "--vjepa2-repo", type=Path, default=None,
        help="local checkout root of facebookresearch/vjepa2 (required for vjepa2_vitl)",
    )
    parser.add_argument(
        "--vjepa2-checkpoint", type=Path, default=None,
        help="local official V-JEPA2 ViT-L checkpoint with target_encoder weights",
    )
    parser.add_argument(
        "--internvideo-next-model-id",
        default="revliter/internvideo_next_base_p14_res224_f16",
        help="Hugging Face repository ID for InternVideo-Next Base",
    )
    parser.add_argument(
        "--internvideo-next-revision",
        default="f673b6016189b6dfc4e20ba92e7f01c5a18eb724",
        help="pinned Hugging Face revision for InternVideo-Next Base",
    )
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
