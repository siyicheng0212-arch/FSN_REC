"""Frozen-protocol single-GPU training for the FSN paper revision.

This runner never resumes, builds a cache, evaluates test during training, or
silently substitutes random weights. Private predictions remain in the run
directory; use ``cvm.analysis`` to produce a shareable aggregate report.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.full_data import FullClipDataset, valid_cache, cache_paths
from experiments.metrics import compute_classification_metrics

ROOT = Path(__file__).resolve().parents[1]
MODES = ("flat", "capacity_control", "aux_flat", "hierarchy")
BACKBONES = ("r2plus1d_18", "r3d_18", "mvit_v2_s", "videomamba_tiny16", "videomae_base16")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    """Hash the runnable CVM implementation, independent of git cleanliness."""
    paths = list((ROOT / "cvm").glob("*.py")) + [ROOT / "experiments" / name for name in ("full_data.py", "pilot_data.py", "metrics.py")]
    launcher = ROOT / "scripts" / "run_cvm_4gpu.sh"
    if launcher.is_file():
        paths.append(launcher)
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(paths)}


def source_digest() -> str:
    return hashlib.sha256(json.dumps(source_hashes(), sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, data: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class StopRequest:
    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None

    def handle(self, signum: int, _frame: Any) -> None:
        self.requested = True
        self.signal_number = signum


def selected_frame_count(backbone: str, frames: int | None = None) -> int:
    """Only convolutional baselines support the predeclared 32-frame sensitivity."""
    count = 16 if frames is None else frames
    if isinstance(count, bool) or count not in (16, 32):
        raise ValueError("frames must be 16 or 32")
    if count != 16 and backbone not in ("r2plus1d_18", "r3d_18"):
        raise ValueError("32-frame sensitivity is restricted to R2+1D/R3D; fixed-length pretrained transformers cannot be silently interpolated")
    return count


class CachedProtocolDataset(FullClipDataset):
    """Reuse existing uint8 cache loading, with explicit protocol cache routing."""
    def __init__(self, protocol: Any, split: str, cache_root: Path, selected_frames: int = 16):
        if isinstance(selected_frames, bool) or selected_frames not in (16, 32):
            raise ValueError("repeat audit requires selected_frames 16 or 32")
        self.torch = torch
        self.records = [protocol.cache_record(record) for record in protocol.records[split]]
        self.cache_root = Path(cache_root)
        self.num_frames, self.crop_size = 36, 224
        invalid = sum(not valid_cache(record, self.cache_root, 36, 224) for record in self.records)
        if invalid:
            raise RuntimeError(f"{invalid} invalid/missing 36f224 caches; cache rebuilding is not authorized by this runner")
        self.repeat_fractions = {}
        self.cache_repeat_fractions = {}
        for record in self.records:
            array_path, metadata_path = cache_paths(self.cache_root, record)
            value = json.loads(metadata_path.read_text(encoding="utf-8")).get("pixel_repeat_fraction")
            self.cache_repeat_fractions[record.clip_id] = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1 else None
            if split != "train":
                array = np.load(array_path, mmap_mode="r", allow_pickle=False)
                indices = torch.linspace(0, 35, selected_frames).round().long().tolist()
                unique = len({hashlib.sha256(array[index].tobytes()).digest() for index in indices})
                self.repeat_fractions[record.clip_id] = 1. - unique / selected_frames
            else:
                self.repeat_fractions[record.clip_id] = None


class Preprocessor:
    """Native torchvision/official normalization and one flip per whole clip.

    Temporal selection is from the existing 36 uniformly cached positions; it
    is not a claim to reproduce the original manuscript's decoder timestamps.
    """
    def __init__(self, backbone: str, augmentation: str = "horizontal_flip", frames: int | None = None):
        from cvm.models import preprocessing_spec, preprocessing_transform
        self.spec = preprocessing_spec(backbone)
        self.spec["frames"] = selected_frame_count(backbone, frames)
        self.transform = preprocessing_transform(backbone)
        self.augmentation = augmentation

    def __call__(self, frames: torch.Tensor, *, training: bool = False) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("cached input must be [B,T,3,H,W]")
        count = int(self.spec["frames"])
        indices = torch.linspace(0, frames.shape[1] - 1, count, device=frames.device).round().long()
        frames = frames.index_select(1, indices)
        if training and self.augmentation == "horizontal_flip":
            flips = torch.rand(frames.shape[0], device=frames.device) < 0.5
            frames = torch.where(flips[:, None, None, None, None], frames.flip(-1), frames)
        return self.transform(frames)


def make_class_weights(records: Any, scheme: str, device: torch.device) -> torch.Tensor | None:
    if scheme == "none":
        return None
    counts = torch.bincount(torch.tensor([record.label_id for record in records]), minlength=7).float()
    if (counts <= 0).any():
        raise ValueError("training must contain all seven classes")
    weights = counts.rsqrt() if scheme == "sqrt_inverse" else counts.reciprocal()
    return (weights / weights.mean()).to(device)


def autocast_context(device: torch.device, amp: str):
    if amp == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bfloat16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_scaler(device: torch.device, amp: str):
    enabled = device.type == "cuda" and amp == "float16"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def step_accumulated(optimizer: torch.optim.Optimizer, scaler: Any, samples: float,
                     clip_grad: float, components: dict[str, Any] | None = None) -> dict[str, float]:
    """Normalize by actual samples, including a partial final accumulation."""
    if samples <= 0:
        return {}
    scaler.unscale_(optimizer)
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]
                  if parameter.grad is not None]
    for parameter in parameters:
        parameter.grad.div_(samples)
    if not all(torch.isfinite(parameter.grad).all().item() for parameter in parameters):
        raise FloatingPointError("non-finite training gradient; run is stopped without automatic restart")
    norms = {name: math.sqrt(sum(float(parameter.grad.float().square().sum()) for parameter in module.parameters() if parameter.grad is not None))
             for name, module in (components or {}).items() if module is not None}
    if clip_grad > 0:
        torch.nn.utils.clip_grad_norm_(parameters, clip_grad, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return norms


def train_epoch(model: torch.nn.Module, loader: Any, optimizer: torch.optim.Optimizer,
                loss_fn: Any, preprocess: Any, device: torch.device, *, accum_steps: int = 1,
                amp: str = "none", scaler: Any = None, clip_grad: float = 20.,
                stop: StopRequest | None = None, freeze_backbone: bool = False,
                loss_normalizer: Any = None) -> dict[str, Any]:
    if accum_steps < 1:
        raise ValueError("accum_steps must be positive")
    scaler = scaler or make_scaler(device, amp)
    model.train()
    if freeze_backbone and hasattr(model, "backbone"):
        model.backbone.eval()  # freeze running statistics during head-only warmup
    optimizer.zero_grad(set_to_none=True)
    pending_samples = pending_batches = samples = optimizer_steps = 0
    total_loss = total_normalizer = 0.
    first_step_norms = None
    first_step_labels = None
    pending_labels: set[int] = set()
    observed_labels: set[int] = set()
    components = {name: getattr(model, name, None) for name in ("backbone", "flat_head", "group_head", "conditional_heads", "capacity_head")}
    started = time.perf_counter()
    for batch in loader:
        if stop and stop.requested:
            break
        frames = batch["video"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        inputs = preprocess(frames, training=True)
        with autocast_context(device, amp):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise FloatingPointError("non-finite/non-scalar training loss; no automatic restart")
        size = len(targets)
        observed_labels.update(targets.detach().cpu().tolist())
        pending_labels.update(targets.detach().cpu().tolist())
        normalizer = float(loss_normalizer(targets)) if loss_normalizer is not None else float(size)
        if not math.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("loss normalizer must be finite and positive")
        scaler.scale(loss * normalizer).backward()
        pending_samples += normalizer
        pending_batches += 1
        samples += size
        total_loss += float(loss.detach()) * normalizer
        total_normalizer += normalizer
        if pending_batches == accum_steps:
            norms = step_accumulated(optimizer, scaler, pending_samples, clip_grad, components if first_step_norms is None else None)
            if first_step_norms is None:
                first_step_norms = norms
                first_step_labels = sorted(pending_labels)
            optimizer_steps += 1
            pending_samples = pending_batches = 0
            pending_labels.clear()
    if pending_samples:
        norms = step_accumulated(optimizer, scaler, pending_samples, clip_grad, components if first_step_norms is None else None)
        if first_step_norms is None:
            first_step_norms = norms
            first_step_labels = sorted(pending_labels)
        optimizer_steps += 1
    return {"loss": total_loss / total_normalizer if total_normalizer else None, "samples": samples,
            "optimizer_steps": optimizer_steps, "seconds": time.perf_counter() - started,
            "first_step_gradient_norms": first_step_norms, "first_step_observed_labels": first_step_labels,
            "observed_labels": sorted(observed_labels),
            "interrupted": bool(stop and stop.requested)}


def trained_head_metadata(model: Any, taxonomy: Any, config: dict[str, Any], split: str,
                          probe: str = "none") -> dict[str, Any]:
    groups = [list(group) for group in taxonomy.groups]
    return {"mode": config["mode"], "trained_heads": list(model.trained_heads),
            "taxonomy": {"name": taxonomy.name, "groups": groups},
            "protocol_sha256": config["protocol_sha256"], "split": split,
            "selection_exposure": "used_each_epoch_for_checkpoint_selection" if split == "val" else "locked_test_not_used_for_selection",
            "eval_metadata": {"backbone": config["backbone"], "preprocessing": config["preprocessing"],
                              "cache_frames": 36, "cache_size": 224, "probe": probe,
                              "repeated_frame_fraction_basis": f"pixel equality among the selected {config['preprocessing']['frames']} RGB cached frames; no verified PTS or motion annotation",
                              "probe_seed": config["seed"], "temporal_positions": "uniform rounded indices in 36 cached positions; not verified PTS"}}


@torch.no_grad()
def evaluate_model(model: Any, loader: Any, preprocess: Any, device: torch.device,
                   taxonomy: Any, mode: str, main_decoder: str, *, amp: str = "none",
                   record_lookup: dict[str, Any] | None = None, probe: str = "none", seed: int = 42,
                   stop: StopRequest | None = None, repeat_lookup: dict[str, float | None] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    from cvm.models import prediction_probabilities
    model.eval()
    prediction_rows: list[dict[str, Any]] = []
    scores, targets_all, metadata = [], [], []
    started = time.perf_counter()
    for batch in loader:
        if stop and stop.requested:
            raise InterruptedError("evaluation interrupted; partial predictions are not reported as complete")
        frames = batch["video"].to(device, non_blocking=True)
        inputs = preprocess(frames, training=False)
        if probe != "none":
            from cvm.probes import apply_probe
            inputs = apply_probe(inputs.permute(0, 2, 1, 3, 4), batch["clip_id"], probe, seed).permute(0, 2, 1, 3, 4).contiguous()
        with autocast_context(device, amp):
            output = model(inputs)
        leaf = prediction_probabilities(output, mode, taxonomy, decoding=main_decoder if mode == "hierarchy" else "soft")
        # Decoder contract is probabilities [B,7], not class indices.
        if leaf.ndim != 2 or leaf.shape[1] != 7 or not torch.isfinite(leaf).all().item():
            raise ValueError("decoder must return finite seven-class probabilities")
        leaf = leaf.float().cpu()
        labels = batch["label"].cpu()
        scores.append(leaf)
        targets_all.append(labels)
        trained = set(model.trained_heads)
        flat = output.get("flat_logits") if "flat" in trained or "flat_logits" in trained else None
        group = output.get("group_logits") if "group" in trained or "group_logits" in trained else None
        conditional = output.get("conditional_logits") if "conditional" in trained or "conditional_logits" in trained else None
        for index, clip_id in enumerate(batch["clip_id"]):
            record = record_lookup[clip_id] if record_lookup else None
            row = {"clip_id": clip_id, "group_id": record.group_id if record else "toy",
                   "source": batch["source"][index], "duration": float(batch["duration"][index]),
                   "target": int(labels[index]), "prediction": int(leaf[index].argmax()),
                   "repeated_frame_fraction": repeat_lookup.get(clip_id) if repeat_lookup is not None else None,
                   "leaf_probs": leaf[index].tolist(),
                   "flat_logits": flat[index].float().cpu().tolist() if flat is not None else None,
                   "group_logits": group[index].float().cpu().tolist() if group is not None else None,
                   "conditional_logits": [part[index].float().cpu().tolist() for part in conditional] if conditional is not None else None}
            prediction_rows.append(row)
            metadata.append({"source": row["source"], "duration": row["duration"]})
    if not prediction_rows:
        raise ValueError("evaluation split is empty")
    metrics = compute_classification_metrics(torch.cat(scores), torch.cat(targets_all), metadata)
    return metrics, prediction_rows, time.perf_counter() - started


def make_optimizer(model: Any, args: argparse.Namespace, warmup: bool) -> torch.optim.Optimizer:
    model.configure_trainable(backbone_trainable=not warmup and getattr(args, "backbone_training", "finetune") != "frozen")
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    backbone = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) in backbone_ids]
    heads = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in backbone_ids]
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": args.backbone_lr, "name": "backbone"})
    if heads:
        groups.append({"params": heads, "lr": args.warmup_lr if warmup else args.head_lr, "name": "trained_heads"})
    if not groups:
        raise ValueError("model has no trainable parameters")
    if args.optimizer == "adamw":
        return torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    return torch.optim.SGD(groups, momentum=args.momentum, weight_decay=args.weight_decay)


def model_arguments(config: dict[str, Any], *, pretrained: bool) -> dict[str, Any]:
    from cvm.taxonomy import Taxonomy, get_taxonomy
    taxonomy = Taxonomy.from_dict(config["taxonomy_definition"]) if config.get("taxonomy_definition") else get_taxonomy(config["taxonomy"])
    arguments = {"backbone": config["backbone"], "mode": config["mode"], "taxonomy": taxonomy,
                 "weights": "DEFAULT" if pretrained else None, "allow_download": config.get("allow_download", False) if pretrained else False,
                 "seed": config["seed"]}
    if (pretrained or config["backbone"] in ("videomamba_tiny16", "videomae_base16")) and config.get("checkpoint"):
        arguments["weights_path"] = Path(config["checkpoint"])
    if config.get("videomamba_root"):
        arguments["videomamba_root"] = Path(config["videomamba_root"])
    if config.get("videomae_root"):
        arguments["videomae_root"] = Path(config["videomae_root"])
    return arguments


def require_device(name: str, amp: str) -> torch.device:
    device = torch.device(name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal training/evaluation requires available CUDA; CPU toy tests use train_epoch directly")
    if amp == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("requested bf16 is unsupported; no silent precision fallback")
    return device


def json_safe_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def run_train(args: argparse.Namespace) -> None:
    from cvm.protocol import load_protocol
    from cvm.models import build_model, compute_loss
    from cvm.analysis import metric_report
    protocol = load_protocol(args.protocol)
    if args.expected_protocol_sha256 and protocol.protocol_sha256 != args.expected_protocol_sha256:
        raise RuntimeError("protocol SHA changed after planning")
    if args.expected_code_hash and source_digest() != args.expected_code_hash:
        raise RuntimeError("runnable source changed after planning")
    if args.checkpoint is None and not args.allow_download:
        raise ValueError("formal training requires --checkpoint or explicit --allow-download")
    device = require_device(args.device, args.amp)
    if args.batch_size < 1 or args.accum_steps < 1 or args.warmup_epochs < 0 or args.epochs < 1 or args.patience < 1:
        raise ValueError("invalid batch/epoch/early-stopping settings")
    if args.mode != "hierarchy" and args.main_decoder != "flat":
        raise ValueError("flat/capacity_control/aux_flat require --main-decoder flat")
    if args.mode == "hierarchy" and args.main_decoder not in ("hard", "soft"):
        raise ValueError("hierarchy requires predeclared --main-decoder hard or soft")
    frame_count = selected_frame_count(args.backbone, getattr(args, "frames", None))
    train_set = CachedProtocolDataset(protocol, "train", args.cache_root, frame_count)
    val_set = CachedProtocolDataset(protocol, "val", args.cache_root, frame_count)
    seed_all(args.seed)
    preprocess = Preprocessor(args.backbone, args.augmentation, frame_count)
    config = json_safe_arguments(args)
    if args.taxonomy_file:
        from cvm.taxonomy import Taxonomy
        definition = json.loads(args.taxonomy_file.read_text(encoding="utf-8"))
        if definition.get("protocol_sha256") != protocol.protocol_sha256 or definition.get("provenance", {}).get("split") != "train":
            raise ValueError("custom taxonomy must have matching frozen protocol and train-only provenance")
        if definition.get("training_manifest_sha256") != sha256_file(protocol.manifest_paths["train"]):
            raise ValueError("custom taxonomy training manifest SHA does not match")
        Taxonomy.from_dict(definition)
        config["taxonomy_definition"] = definition
        config["taxonomy_file_metadata"] = definition
        config["taxonomy_file_sha256"] = sha256_file(args.taxonomy_file)
    config.update({"protocol_sha256": protocol.protocol_sha256, "source_hashes": source_hashes(), "code_hash": source_digest(),
                   "preprocessing": preprocess.spec, "effective_batch_size": args.batch_size * args.accum_steps,
                   "cache_contract": "existing uniform36f224 only; no decode/rebuild",
                   "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint else None,
                   "checkpoint_selection": "seven-class validation Macro-F1 after warmup; official backbone remains frozen" if getattr(args, "backbone_training", "finetune") == "frozen" else "seven-class validation Macro-F1, finetune epochs only; warmup cannot become the full-finetune result",
                   "test_evaluated": False})
    try:
        config["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        config["git_commit"] = None
    model = build_model(**model_arguments(config, pretrained=True)).to(device)
    if model.load_report.get("sha256"):
        config["checkpoint_sha256"] = model.load_report["sha256"]
    import torchvision
    config["runtime"] = {"python": sys.version.split()[0], "torch": torch.__version__, "torchvision": torchvision.__version__,
                         "cuda": torch.version.cuda, "device": str(device),
                         "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    taxonomy = model.taxonomy
    config["taxonomy_definition"] = taxonomy.to_dict()
    config["trained_heads"] = list(model.trained_heads)
    config["taxonomy_groups"] = [list(group) for group in taxonomy.groups]
    config["parameters_total"] = sum(parameter.numel() for parameter in model.parameters())
    model.configure_trainable(backbone_trainable=getattr(args, "backbone_training", "finetune") != "frozen")
    config["parameters_trainable_finetune"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    config["parameter_report"] = model.parameter_report()
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "config.json", config)
    atomic_json(args.output / "protocol_summary.json", json.loads(args.protocol.read_text(encoding="utf-8")))
    atomic_json(args.output / "load_report.json", model.load_report)
    atomic_json(args.output / "history.json", [])
    weights = make_class_weights(protocol.records["train"], args.class_weights, device)
    config["class_weights_train_only"] = weights.cpu().tolist() if weights is not None else None
    atomic_json(args.output / "config.json", config)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, generator=generator, worker_init_fn=seed_worker, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True, worker_init_fn=seed_worker, drop_last=False)
    loss_fn = lambda outputs, target: compute_loss(outputs, target, mode=args.mode, taxonomy=taxonomy,
                                                  aux_weight=args.aux_weight, group_weight=args.group_weight,
                                                  conditional_weight=args.conditional_weight, class_weights=weights)
    loss_normalizer = (lambda target: weights[target].sum().item()) if weights is not None else None
    stop = StopRequest()
    old_handlers = {number: signal.signal(number, stop.handle) for number in (signal.SIGINT, signal.SIGTERM)}
    history: list[dict[str, Any]] = []
    best_score, best_epoch, stale = -math.inf, None, 0
    optimizer = make_optimizer(model, args, warmup=args.warmup_epochs > 0)
    scaler = make_scaler(device, args.amp)
    started = time.perf_counter()
    status, failure = "completed", None
    try:
        for epoch in range(args.warmup_epochs + args.epochs):
            if stop.requested:
                break
            warmup = epoch < args.warmup_epochs
            if epoch == args.warmup_epochs and args.warmup_epochs:
                optimizer = make_optimizer(model, args, warmup=False)
            if not warmup:
                # Fixed cosine schedule; all compared modes use the same settings.
                step = epoch - args.warmup_epochs
                factor = .5 * (1 + math.cos(math.pi * step / max(args.epochs, 1)))
                for group in optimizer.param_groups:
                    group["lr"] = (args.backbone_lr if group["name"] == "backbone" else args.head_lr) * factor
            train_result = train_epoch(model, train_loader, optimizer, loss_fn, preprocess, device,
                                       accum_steps=args.accum_steps, amp=args.amp, scaler=scaler,
                                       clip_grad=args.clip_grad, stop=stop,
                                       freeze_backbone=warmup or getattr(args, "backbone_training", "finetune") == "frozen",
                                       loss_normalizer=loss_normalizer)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                     "config": config, "epoch": epoch + 1,
                     "stage": "warmup" if warmup else "frozen_features" if getattr(args, "backbone_training", "finetune") == "frozen" else "finetune",
                     "interrupted": stop.requested, "best_epoch": best_epoch,
                     "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}}
            atomic_checkpoint(args.output / "last.pt", state)
            if stop.requested:
                history.append({"epoch": epoch + 1, "stage": state["stage"], "train": train_result,
                                "validation_complete": False})
                atomic_json(args.output / "history.json", history)
                break
            val_metrics, rows, eval_seconds = evaluate_model(model, val_loader, preprocess, device, taxonomy,
                                                             args.mode, args.main_decoder, amp=args.amp,
                                                             record_lookup={record.clip_id: record for record in protocol.records["val"]}, stop=stop,
                                                             repeat_lookup=val_set.repeat_fractions)
            score = val_metrics["all"]["macro_f1"]
            entry = {"epoch": epoch + 1, "stage": state["stage"], "train": train_result,
                     "val_macro_f1": score, "val_accuracy": val_metrics["all"]["accuracy"],
                     "eval_seconds": eval_seconds, "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
                     "validation_complete": True}
            history.append(entry)
            if not warmup and score > best_score + args.min_delta:
                best_score, best_epoch, stale = score, epoch + 1, 0
                state["best_epoch"] = best_epoch
                state["selected_val_macro_f1"] = score
                atomic_checkpoint(args.output / "best.pt", state)
                write_jsonl(args.output / "val_predictions.jsonl", rows)
                metadata = trained_head_metadata(model, taxonomy, config, "val")
                atomic_json(args.output / "prediction_metadata.json", metadata)
                atomic_json(args.output / "val_report.json", metric_report(rows, metadata=metadata))
            elif not warmup:
                stale += 1
            atomic_json(args.output / "history.json", history)
            print(json.dumps(entry, ensure_ascii=False, allow_nan=False), flush=True)
            if not warmup and stale >= args.patience:
                status = "early_stopped"
                break
        if stop.requested:
            status = "interrupted"
        if source_digest() != config["code_hash"] or load_protocol(args.protocol).protocol_sha256 != protocol.protocol_sha256:
            raise RuntimeError("source/protocol changed during run; result is invalid")
    except InterruptedError:
        status = "interrupted"
    except BaseException as exc:
        status = "failed"
        failure = type(exc).__name__
        atomic_checkpoint(args.output / "last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scaler": scaler.state_dict(), "config": config, "interrupted": True,
                          "failure_type": failure, "best_epoch": best_epoch})
        raise
    finally:
        if status == "interrupted":
            atomic_checkpoint(args.output / "last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "scaler": scaler.state_dict(), "config": config, "interrupted": True,
                              "best_epoch": best_epoch, "signal_number": stop.signal_number})
        for number, handler in old_handlers.items():
            signal.signal(number, handler)
        result = {"status": status, "training_completed": status in ("completed", "early_stopped"),
                  "best_epoch": best_epoch, "best_val_macro_f1": best_score if best_epoch is not None else None,
                  "best_stage": ("frozen_features" if getattr(args, "backbone_training", "finetune") == "frozen" else "finetune") if best_epoch is not None else None,
                  "seconds": time.perf_counter() - started, "failure_type": failure,
                  "signal_number": stop.signal_number, "test_evaluated": False,
                  "protocol_sha256": protocol.protocol_sha256, "parameters_total": config["parameters_total"],
                  "parameters_trainable_finetune": config["parameters_trainable_finetune"],
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
        atomic_json(args.output / "result.json", result)
    if status == "interrupted":
        raise SystemExit(130)


def run_evaluate(args: argparse.Namespace) -> None:
    from cvm.protocol import load_protocol
    from cvm.models import build_model
    from cvm.analysis import metric_report
    protocol = load_protocol(args.protocol)
    # Only our own trusted run checkpoints are accepted. They contain config.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["code_hash"] != source_digest():
        raise RuntimeError("runnable code differs from training; evaluation requires the frozen implementation")
    if config["protocol_sha256"] != protocol.protocol_sha256:
        raise RuntimeError("checkpoint and frozen evaluation protocol do not match")
    if args.split == "test":
        if not args.allow_test or args.locked_protocol_sha256 != protocol.protocol_sha256:
            raise RuntimeError("test requires explicit authorization and matching locked protocol SHA")
        if protocol.summary.get("test_history_status") != "verified_clean":
            raise RuntimeError("test history is not verified clean; it cannot be reported as independent test")
    device = require_device(args.device, args.amp)
    frame_count = selected_frame_count(config["backbone"], config["preprocessing"]["frames"])
    dataset = CachedProtocolDataset(protocol, args.split, args.cache_root, frame_count)
    model = build_model(**model_arguments(config, pretrained=False))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    preprocess = Preprocessor(config["backbone"], "none", frame_count)
    if preprocess.spec != config["preprocessing"]:
        raise RuntimeError("preprocessing changed since checkpoint training")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    seed_all(config["seed"])
    _, rows, seconds = evaluate_model(model, loader, preprocess, device, model.taxonomy,
                                      config["mode"], config["main_decoder"], amp=args.amp,
                                      record_lookup={record.clip_id: record for record in protocol.records[args.split]},
                                      probe=args.probe, seed=config["seed"], repeat_lookup=dataset.repeat_fractions)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = trained_head_metadata(model, model.taxonomy, config, args.split, args.probe)
    metadata["checkpoint_sha256"] = sha256_file(args.checkpoint)
    metadata["checkpoint_epoch"] = checkpoint.get("epoch")
    write_jsonl(args.output / "predictions.jsonl", rows)
    atomic_json(args.output / "prediction_metadata.json", metadata)
    atomic_json(args.output / "report.json", metric_report(rows, metadata=metadata))
    atomic_json(args.output / "result.json", {"status": "evaluation_complete", "seconds": seconds,
                                              "split": args.split, "probe": args.probe, "training_started": False})


@torch.no_grad()
def run_export_features(args: argparse.Namespace) -> None:
    """Export training features only, for a separately frozen visual taxonomy."""
    from cvm.protocol import load_protocol
    from cvm.models import build_model
    protocol = load_protocol(args.protocol)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["code_hash"] != source_digest():
        raise RuntimeError("runnable code differs from feature checkpoint training")
    if config["protocol_sha256"] != protocol.protocol_sha256 or config["mode"] != "flat":
        raise RuntimeError("feature export requires a matching frozen-protocol flat checkpoint")
    device = require_device(args.device, args.amp)
    model = build_model(**model_arguments(config, pretrained=False))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    frame_count = selected_frame_count(config["backbone"], config["preprocessing"]["frames"])
    preprocess = Preprocessor(config["backbone"], "none", frame_count)
    if preprocess.spec != config["preprocessing"]:
        raise RuntimeError("preprocessing changed since training")
    dataset = CachedProtocolDataset(protocol, "train", args.cache_root, frame_count)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    features, labels = [], []
    for batch in loader:
        inputs = preprocess(batch["video"].to(device, non_blocking=True), training=False)
        with autocast_context(device, args.amp):
            output = model(inputs)
        values = output["features"].float().cpu().numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError("feature export contains non-finite values")
        features.append(values)
        labels.append(batch["label"].numpy())
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(args.output / "features.npz", features=np.concatenate(features), labels=np.concatenate(labels))
    atomic_json(args.output / "features.meta.json", {
        "split": "train", "protocol_sha256": protocol.protocol_sha256,
        "training_manifest_sha256": sha256_file(protocol.manifest_paths["train"]),
        "checkpoint_sha256": sha256_file(args.checkpoint), "backbone": config["backbone"],
        "feature_encoder_selection_exposure": "checkpoint selected on validation; exported examples and taxonomy labels are training only",
        "features_sha256": sha256_file(args.output / "features.npz"), "rows": len(dataset),
        "preprocessing": preprocess.spec, "evaluation_metrics_computed": False,
        "private_artifact": True})


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--protocol", type=Path, required=True)
    train.add_argument("--cache-root", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--backbone", choices=BACKBONES, required=True)
    train.add_argument("--mode", choices=MODES, required=True)
    train.add_argument("--taxonomy", choices=("clinical", "random_17", "random_29", "random_43"), default="clinical")
    train.add_argument("--taxonomy-file", type=Path)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--checkpoint", type=Path)
    train.add_argument("--allow-download", action="store_true")
    train.add_argument("--videomamba-root", type=Path)
    train.add_argument("--videomae-root", type=Path)
    train.add_argument("--frames", type=int, choices=(16, 32), default=16)
    train.add_argument("--backbone-training", choices=("finetune", "frozen"), default="finetune")
    train.add_argument("--device", default="cuda")
    train.add_argument("--amp", choices=("none", "bfloat16", "float16"), default="bfloat16")
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--accum-steps", type=int, default=16)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--warmup-epochs", type=int, default=5)
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--min-delta", type=float, default=0.)
    train.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    train.add_argument("--backbone-lr", type=float, default=1e-5)
    train.add_argument("--head-lr", type=float, default=1e-4)
    train.add_argument("--warmup-lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=.05)
    train.add_argument("--momentum", type=float, default=.9)
    train.add_argument("--class-weights", choices=("none", "sqrt_inverse", "inverse"), default="none")
    train.add_argument("--clip-grad", type=float, default=5.)
    train.add_argument("--augmentation", choices=("none", "horizontal_flip"), default="horizontal_flip")
    train.add_argument("--main-decoder", choices=("flat", "hard", "soft"), required=True)
    train.add_argument("--aux-weight", type=float, default=1.)
    train.add_argument("--group-weight", type=float, default=1.)
    train.add_argument("--conditional-weight", type=float, default=1.)
    train.add_argument("--expected-protocol-sha256")
    train.add_argument("--expected-code-hash")
    evaluate = sub.add_parser("evaluate")
    for name in ("protocol", "cache-root", "checkpoint", "output"):
        evaluate.add_argument("--" + name, type=Path, required=True)
    evaluate.add_argument("--split", choices=("val", "test"), default="val")
    evaluate.add_argument("--allow-test", action="store_true")
    evaluate.add_argument("--locked-protocol-sha256")
    evaluate.add_argument("--probe", choices=("none", "static", "shuffle"), default="none")
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--amp", choices=("none", "bfloat16", "float16"), default="bfloat16")
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--workers", type=int, default=4)
    export = sub.add_parser("export-features")
    for name in ("protocol", "cache-root", "checkpoint", "output"):
        export.add_argument("--" + name, type=Path, required=True)
    export.add_argument("--device", default="cuda")
    export.add_argument("--amp", choices=("none", "bfloat16", "float16"), default="bfloat16")
    export.add_argument("--batch-size", type=int, default=2)
    export.add_argument("--workers", type=int, default=4)
    return command


def main() -> None:
    args = parser().parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        run_export_features(args)


if __name__ == "__main__":
    main()
