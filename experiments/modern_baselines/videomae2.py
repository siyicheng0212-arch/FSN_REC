"""Fail-closed adapter for OpenGVLab's distilled VideoMAE V2 ViT-B/16.

This module prepares and checks an independent seven-class visual baseline. It
does not train, download, alter the frozen manifests/cache, or read test. A
formal validation requires a separately trained and sealed FSN checkpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from experiments.aligned_data import ResolvedFullClipDataset
from experiments.aligned_protocol import (
    DATA_PROTOCOL, EXPECTED_COUNTS, EXPECTED_MANIFEST_SHA,
    validate_manifest_protocol,
)
from experiments.full_data import cache_paths
from experiments.metrics import DEFAULT_CLASS_NAMES, compute_classification_metrics


# The model zoo identifies this file as the K710 student distilled from the
# hybrid-pretrained, K710-finetuned ViT-giant teacher. This is a classification
# checkpoint, not a masked-autoencoder-only or randomly initialized backbone.
OFFICIAL_COMMIT = "29eab1e8a588d1b3ec0cdec7b03a86cca491b74b"
OFFICIAL_MODEL_SHA256 = "d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb"
OFFICIAL_CHECKPOINT_NAME = "vit_b_k710_dl_from_giant.pth"
OFFICIAL_CHECKPOINT_SHA256 = "8141a6955e0700d11bf15928fe6d61e5cfe482606fed8cfdddb1b922c0fd88ec"
OFFICIAL_TIMM_VERSION = "0.4.12"
EXPECTED_STATE_KEYS = 162
EXPECTED_BACKBONE_PARAMETERS = 86_227_200
INPUT_FRAMES = 16
CACHE_FRAMES = 36
INPUT_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
FRAME_INDICES = tuple(math.floor((index + 0.5) * CACHE_FRAMES / INPUT_FRAMES)
                      for index in range(INPUT_FRAMES))
FINETUNED_SCHEMA = "fsn-videomae2-distilled-vitb-finetuned-v1"
_SHA_RE = re.compile(r"[a-f0-9]{64}\Z")


class BaselinePreflightError(RuntimeError):
    """One required input or provenance check failed; do not report a baseline."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_official_source(source_root: Path) -> Path:
    """Accept only the audited OpenGVLab checkout and exact model source bytes."""
    source_root = Path(source_root).resolve()
    model_path = source_root / "models" / "modeling_finetune.py"
    if not model_path.is_file() or sha256_file(model_path) != OFFICIAL_MODEL_SHA256:
        raise BaselinePreflightError("official modeling_finetune.py is absent or modified")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BaselinePreflightError("cannot verify OpenGVLab checkout commit") from exc
    if revision != OFFICIAL_COMMIT:
        raise BaselinePreflightError("OpenGVLab checkout commit differs from pinned version")
    return model_path


def verify_official_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Hash before deserializing; no remote URL or arbitrary checkpoint accepted."""
    checkpoint_path = Path(checkpoint_path).resolve()
    if checkpoint_path.name != OFFICIAL_CHECKPOINT_NAME or not checkpoint_path.is_file():
        raise BaselinePreflightError("missing named official distilled ViT-B K710 checkpoint")
    digest = sha256_file(checkpoint_path)
    if digest != OFFICIAL_CHECKPOINT_SHA256:
        raise BaselinePreflightError("official distilled ViT-B checkpoint SHA256 mismatch")
    return {"checkpoint_sha256": digest, "checkpoint_bytes": checkpoint_path.stat().st_size}


def _official_constructor(model_path: Path):
    try:
        timm_version = importlib.metadata.version("timm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise BaselinePreflightError("official model requires timm==0.4.12") from exc
    if timm_version != OFFICIAL_TIMM_VERSION:
        raise BaselinePreflightError(f"official model requires timm=={OFFICIAL_TIMM_VERSION}")
    # Import exactly the bytes being checked. A path-based loader could reopen
    # a changed source file after verify_official_source() approved its hash.
    try:
        source_bytes = Path(model_path).read_bytes()
    except OSError as exc:
        raise BaselinePreflightError("cannot read pinned OpenGVLab model source") from exc
    if hashlib.sha256(source_bytes).hexdigest() != OFFICIAL_MODEL_SHA256:
        raise BaselinePreflightError("official modeling_finetune.py changed before import")
    spec = importlib.util.spec_from_file_location("_fsn_official_videomae2_finetune", model_path)
    if spec is None or spec.loader is None:
        raise BaselinePreflightError("cannot import pinned OpenGVLab model source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        exec(compile(source_bytes, str(model_path), "exec"), module.__dict__)
    except (ImportError, AttributeError, SyntaxError) as exc:
        sys.modules.pop(spec.name, None)
        raise BaselinePreflightError("pinned OpenGVLab model could not load with installed dependencies") from exc
    return module.vit_base_patch16_224


def _checkpoint_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint_path = Path(checkpoint_path).resolve()
    if checkpoint_path.name != OFFICIAL_CHECKPOINT_NAME or not checkpoint_path.is_file():
        raise BaselinePreflightError("missing named official distilled ViT-B K710 checkpoint")
    try:
        # Hash and deserialize through the same open descriptor, then hash it
        # again to detect in-place changes made while torch.load was reading.
        with checkpoint_path.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != OFFICIAL_CHECKPOINT_SHA256:
                raise BaselinePreflightError("official distilled ViT-B checkpoint SHA256 mismatch")
            stream.seek(0)
            payload = torch.load(stream, map_location="cpu", weights_only=True)
            stream.seek(0)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != OFFICIAL_CHECKPOINT_SHA256:
                raise BaselinePreflightError("official checkpoint changed during deserialization")
    except (OSError, RuntimeError, ValueError) as exc:
        raise BaselinePreflightError("verified checkpoint is not readable with safe tensor loading") from exc
    if not isinstance(payload, dict) or set(payload) != {"module"}:
        raise BaselinePreflightError("distilled checkpoint must have its official module container")
    state = payload["module"]
    if not isinstance(state, dict) or len(state) != EXPECTED_STATE_KEYS:
        raise BaselinePreflightError("distilled checkpoint has an unexpected state structure")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
           for key, value in state.items()):
        raise BaselinePreflightError("distilled checkpoint must contain only named tensors")
    return state


def load_strict_pretrained_state(model: torch.nn.Module,
                                 state: dict[str, torch.Tensor]) -> dict[str, int]:
    """Load every backbone tensor; only the K710 classifier is replaced by FSN-7.

    No position embedding interpolation, renamed keys, partial backbone load,
    or use of a randomly initialized backbone is allowed.
    """
    expected = model.state_dict()
    heads = {"head.weight", "head.bias"}
    if set(expected) - set(state) or set(state) - set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise BaselinePreflightError(
            f"checkpoint/model keys differ (missing={missing[:4]}, extra={extra[:4]})")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise BaselinePreflightError("checkpoint state contains a non-tensor value")
    if tuple(state["head.weight"].shape) != (710, 768) or \
            tuple(state["head.bias"].shape) != (710,) or \
            any(state[key].dtype != torch.float16 for key in heads):
        raise BaselinePreflightError("checkpoint is not the expected K710 ViT-B student")
    if tuple(expected["head.weight"].shape) != (7, 768) or \
            tuple(expected["head.bias"].shape) != (7,):
        raise BaselinePreflightError("model classifier is not FSN seven-class ViT-B")
    backbone = {key: value for key, value in state.items() if key not in heads}
    backbone_parameters = 0
    for key, value in backbone.items():
        if value.shape != expected[key].shape or value.dtype != torch.float16:
            raise BaselinePreflightError(f"pretrained backbone tensor mismatch: {key}")
        if not torch.isfinite(value).all().item():
            raise BaselinePreflightError(f"nonfinite pretrained backbone tensor: {key}")
        backbone_parameters += value.numel()
    if backbone_parameters != EXPECTED_BACKBONE_PARAMETERS:
        raise BaselinePreflightError("pretrained backbone parameter count differs from official ViT-B")
    if not all(torch.isfinite(state[key]).all().item() for key in heads):
        raise BaselinePreflightError("nonfinite K710 classifier in official checkpoint")
    incompatible = model.load_state_dict(backbone, strict=False)
    if set(incompatible.missing_keys) != heads or incompatible.unexpected_keys:
        raise BaselinePreflightError("not all pretrained backbone tensors were loaded")
    return {"loaded_backbone_tensors": len(backbone),
            "loaded_backbone_parameters": backbone_parameters,
            "reinitialized_fsn_head_parameters": sum(expected[key].numel() for key in heads)}


def build_pretrained_model(source_root: Path, checkpoint_path: Path):
    """Instantiate the official architecture and load the exact published weights."""
    model_path = verify_official_source(source_root)
    checkpoint_audit = verify_official_checkpoint(checkpoint_path)
    constructor = _official_constructor(model_path)
    model = constructor(
        pretrained=False, img_size=INPUT_SIZE, num_classes=7,
        all_frames=INPUT_FRAMES, tubelet_size=2, use_mean_pooling=True,
        init_scale=0.001, drop_path_rate=0.1, with_cp=False,
    )
    if model.patch_embed.patch_size != (16, 16) or model.patch_embed.tubelet_size != 2 \
            or model.patch_embed.num_patches != 8 * 14 * 14 \
            or model.get_num_layers() != 12 or model.num_features != 768:
        raise BaselinePreflightError("official constructor produced unexpected ViT-B/16 geometry")
    state = _checkpoint_state(Path(checkpoint_path))
    load_audit = load_strict_pretrained_state(model, state)
    report = {
        "architecture": "OpenGVLab VideoMAEv2 vit_base_patch16_224",
        "source_commit": OFFICIAL_COMMIT,
        "model_source_sha256": OFFICIAL_MODEL_SHA256,
        "pretraining": "Kinetics-710 distilled student from VideoMAEv2 ViT-giant teacher",
        "checkpoint_name": OFFICIAL_CHECKPOINT_NAME,
        **checkpoint_audit, **load_audit,
        "input_shape": [3, INPUT_FRAMES, INPUT_SIZE, INPUT_SIZE],
        "tubelet_size": 2,
        "normalization": {"mean": list(MEAN), "std": list(STD)},
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "formal_fsn_evaluation_ready": False,
    }
    model._fsn_videomae2_pretraining = report
    return model, report


def preprocess_cached_video(video: torch.Tensor) -> torch.Tensor:
    """One fixed 16-frame view from the existing 36f/224 RGB cache.

    Cached FFmpeg pixels are already center-cropped to 224. The original
    OpenGVLab 16x5x3 K710 benchmark used its own video decoding/views; this
    experiment records its deterministic cache-derived one-view difference.
    """
    if not isinstance(video, torch.Tensor) or video.dtype != torch.uint8 or \
            tuple(video.shape) != (CACHE_FRAMES, 3, INPUT_SIZE, INPUT_SIZE):
        raise BaselinePreflightError("expected existing uint8 [36,3,224,224] RGB cache")
    view = video[list(FRAME_INDICES)].permute(1, 0, 2, 3).contiguous().float().div_(255.0)
    mean = view.new_tensor(MEAN).view(3, 1, 1, 1)
    std = view.new_tensor(STD).view(3, 1, 1, 1)
    return (view - mean) / std


class SingleViewDataset(Dataset):
    def __init__(self, base: ResolvedFullClipDataset, split: str, manifest_sha256: str):
        self.base = base
        self.split = split
        self.manifest_sha256 = manifest_sha256

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.base[index]
        return {**row, "video": preprocess_cached_video(row["video"])}


def audit_frozen_data(manifest_dir: Path, cache_dir: Path):
    """Read only the byte-frozen train/val and validate every existing cache."""
    manifest_dir, cache_dir = Path(manifest_dir), Path(cache_dir)
    if not cache_dir.is_dir():
        raise BaselinePreflightError("existing 36f/224 cache directory is absent")
    bases, rows, manifest_sha = {}, {}, {}
    for split in ("train", "val"):
        path = manifest_dir / f"{split}.jsonl"
        if not path.is_file():
            raise BaselinePreflightError(f"frozen {split} manifest is absent")
        manifest_sha[split] = sha256_file(path)
        if manifest_sha[split] != EXPECTED_MANIFEST_SHA[split]:
            raise BaselinePreflightError(f"frozen {split} manifest SHA256 mismatch")
        try:
            base = ResolvedFullClipDataset(path, cache_dir, CACHE_FRAMES,
                                           INPUT_SIZE, expected_split=split)
        except (OSError, RuntimeError, ValueError) as exc:
            raise BaselinePreflightError(f"{split} manifest/cache validation failed") from exc
        bases[split], rows[split] = base, base.records
    counts = {split: len(bases[split]) for split in ("train", "val")}
    validate_manifest_protocol(counts, manifest_sha)
    for field in ("clip_id", "group_id"):
        train = {getattr(row, field) for row in rows["train"]}
        val = {getattr(row, field) for row in rows["val"]}
        if train & val:
            raise BaselinePreflightError(f"frozen train/val overlap in {field}")
    class_counts = {}
    for split in ("train", "val"):
        counts_for_split = Counter(row.label_id for row in rows[split])
        if set(counts_for_split) != set(range(7)):
            raise BaselinePreflightError(f"frozen {split} does not contain all seven classes")
        class_counts[split] = {str(key): counts_for_split[key] for key in range(7)}
    mapping, inventory = [], []
    for split in ("train", "val"):
        base = bases[split]
        for record, storage in zip(base.records, base.storage_records):
            array, metadata = cache_paths(cache_dir, storage)
            stat = array.stat()
            mapping.append({"split": split, "clip_id": record.clip_id,
                            "cache_split": storage.split})
            inventory.append({"split": split, "clip_id": record.clip_id,
                              "cache_split": storage.split,
                              "metadata_sha256": sha256_file(metadata),
                              "array_bytes": stat.st_size,
                              "array_mtime_ns": stat.st_mtime_ns})
    audit = {"data_protocol": DATA_PROTOCOL, "counts": counts,
             "manifest_sha256": manifest_sha, "class_counts": class_counts,
             "cache_mapping_sha256": _object_sha256(mapping),
             "cache_inventory_sha256": _object_sha256(inventory),
             "cache_inventory_scope": "metadata SHA256 plus array size/mtime, not full pixel SHA256",
             "cached_frames": CACHE_FRAMES, "model_frames": INPUT_FRAMES,
             "frame_indices_zero_based": list(FRAME_INDICES),
             "spatial_view": "existing FFmpeg 224-square center crop",
             "validation_role": "val823 development; no independent test",
             "test_read": False}
    datasets = {split: SingleViewDataset(bases[split], split, manifest_sha[split])
                for split in ("train", "val")}
    return datasets, audit


def _device_report(device: torch.device) -> dict[str, Any]:
    report = {"torch_version": torch.__version__, "device": str(device),
              "cuda_runtime": torch.version.cuda}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        report.update(gpu_name=properties.name, gpu_total_bytes=properties.total_memory)
    return report


def smoke_bf16_batch(model: torch.nn.Module, train_dataset: SingleViewDataset,
                     *, device: torch.device, batch_size: int = 4,
                     workers: int = 0) -> dict[str, Any]:
    """One actual cached train batch, forward/backward only; no optimizer step."""
    if train_dataset.split != "train" or len(train_dataset) != EXPECTED_COUNTS["train"]:
        raise BaselinePreflightError("bf16 smoke requires frozen full train7372")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise BaselinePreflightError("actual CUDA device with bf16 support is required")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise BaselinePreflightError("selected CUDA device lacks bf16 support")
    if batch_size < 2 or batch_size > len(train_dataset):
        raise BaselinePreflightError("bf16 smoke batch size must be between 2 and train size")
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    batch = next(iter(loader))
    videos = batch["video"].to(device, non_blocking=True)
    targets = batch["label"].to(device, non_blocking=True)
    model.to(device).train()
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(videos)
        if tuple(logits.shape) != (batch_size, 7):
            raise BaselinePreflightError("bf16 smoke logits are not [batch,7]")
        loss = F.cross_entropy(logits.float(), targets)
    if not torch.isfinite(loss).item() or not torch.isfinite(logits).all().item():
        raise BaselinePreflightError("bf16 smoke produced nonfinite loss/logits")
    loss.backward()
    head_grad = model.head.weight.grad
    backbone_grad = model.patch_embed.proj.weight.grad
    if head_grad is None or backbone_grad is None or \
            not torch.isfinite(head_grad).all().item() or \
            not torch.isfinite(backbone_grad).all().item():
        raise BaselinePreflightError("bf16 smoke lacks finite classifier/backbone gradients")
    torch.cuda.synchronize(device)
    report = {"batch_size": batch_size, "loss": float(loss.item()),
              "logits_shape": list(logits.shape), "autocast_dtype": "bfloat16",
              "backward_completed": True, "optimizer_steps": 0,
              "elapsed_seconds": time.perf_counter() - start,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
              **_device_report(device)}
    model.zero_grad(set_to_none=True)
    return report


def load_finetuned_fsn_checkpoint(model: torch.nn.Module, path: Path,
                                  expected_sha256: str) -> dict[str, Any]:
    """Require a complete FSN-trained seven-class state before formal eval."""
    if not getattr(model, "_fsn_videomae2_pretraining", None):
        raise BaselinePreflightError("load the verified official pretraining before FSN checkpoint")
    if not _SHA_RE.fullmatch(expected_sha256):
        raise BaselinePreflightError("a 64-digit FSN checkpoint SHA256 is required")
    path = Path(path)
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise BaselinePreflightError("FSN trained checkpoint is absent or its SHA256 changed")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise BaselinePreflightError("FSN checkpoint is not readable with safe tensor loading") from exc
    if not isinstance(payload, dict) or payload.get("schema") != FINETUNED_SCHEMA or \
            payload.get("training_completed") is not True or \
            payload.get("pretrained_checkpoint_sha256") != OFFICIAL_CHECKPOINT_SHA256 or \
            payload.get("official_source_commit") != OFFICIAL_COMMIT or \
            payload.get("data_protocol") != DATA_PROTOCOL or \
            payload.get("manifest_sha256") != EXPECTED_MANIFEST_SHA or \
            payload.get("class_names") != list(DEFAULT_CLASS_NAMES):
        raise BaselinePreflightError("FSN checkpoint lacks the exact trained-baseline provenance")
    steps = payload.get("optimizer_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise BaselinePreflightError("FSN checkpoint must record positive optimizer steps")
    finetune_steps = payload.get("finetune_optimizer_steps")
    if isinstance(finetune_steps, bool) or not isinstance(finetune_steps, int) or finetune_steps <= 0:
        raise BaselinePreflightError("FSN checkpoint must record actual backbone finetune steps")
    state = payload.get("model")
    if not isinstance(state, dict) or set(state) != set(model.state_dict()):
        raise BaselinePreflightError("FSN trained checkpoint model state is incomplete")
    for key, value in state.items():
        reference = model.state_dict()[key]
        if not isinstance(value, torch.Tensor) or value.shape != reference.shape or \
                not torch.isfinite(value).all().item():
            raise BaselinePreflightError(f"FSN trained checkpoint tensor is invalid: {key}")
    model.load_state_dict(state, strict=True)
    model._fsn_finetuned_checkpoint_sha256 = expected_sha256
    return {"finetuned_checkpoint_sha256": expected_sha256,
            "trained_epoch": payload.get("epoch"),
            "optimizer_steps": steps,
            "finetune_optimizer_steps": finetune_steps,
            "selection": payload.get("selection"),
            "training_completed": True}


@torch.inference_mode()
def evaluate_single_view(model: torch.nn.Module, val_dataset: SingleViewDataset,
                         *, device: torch.device, batch_size: int = 4,
                         workers: int = 0) -> dict[str, Any]:
    """Evaluate a sealed, trained checkpoint on each development clip once."""
    trained_sha = getattr(model, "_fsn_finetuned_checkpoint_sha256", None)
    if not trained_sha:
        raise BaselinePreflightError("FSN-trained seven-class checkpoint required before evaluation")
    return _evaluate_single_view_unsealed(model, val_dataset, device=device,
                                          batch_size=batch_size, workers=workers,
                                          checkpoint_sha256=trained_sha)


@torch.inference_mode()
def _evaluate_single_view_unsealed(model: torch.nn.Module, val_dataset: SingleViewDataset,
                                   *, device: torch.device, batch_size: int,
                                   workers: int, checkpoint_sha256: str | None = None) -> dict[str, Any]:
    """Internal selection pass; only the public wrapper represents formal eval."""
    if val_dataset.split != "val" or len(val_dataset) != EXPECTED_COUNTS["val"] or \
            val_dataset.manifest_sha256 != EXPECTED_MANIFEST_SHA["val"]:
        raise BaselinePreflightError("evaluation requires the frozen full val823")
    if batch_size <= 0:
        raise BaselinePreflightError("evaluation batch size must be positive")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise BaselinePreflightError("selected CUDA device is unavailable")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise BaselinePreflightError("selected CUDA device lacks bf16 support")
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")
    model.to(device).eval()
    scores, targets, metadata, seen = [], [], [], set()
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        if any(identifier in seen for identifier in batch["clip_id"]):
            raise BaselinePreflightError("duplicate clip in single-view evaluation")
        seen.update(batch["clip_id"])
        videos = batch["video"].to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(videos)
        if tuple(logits.shape) != (len(batch["clip_id"]), 7) or \
                not torch.isfinite(logits).all().item():
            raise BaselinePreflightError("invalid single-view validation logits")
        scores.append(logits.float().cpu())
        targets.append(batch["label"].cpu())
        metadata.extend({"source_collection": source, "duration": float(duration)}
                        for source, duration in zip(batch["source"], batch["duration"]))
    if len(seen) != EXPECTED_COUNTS["val"]:
        raise BaselinePreflightError("single-view evaluation did not cover all val823 clips")
    result = {"metrics": compute_classification_metrics(
                  torch.cat(scores), torch.cat(targets), metadata),
              "clips_evaluated": len(seen), "views_per_clip": 1,
              "validation_role": "val823 development; not independent test",
              "test_read": False, "finetuned_checkpoint_sha256": checkpoint_sha256,
              "elapsed_seconds": time.perf_counter() - start,
              **_device_report(device)}
    if device.type == "cuda":
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke", "evaluate"), required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--distilled-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--finetuned-checkpoint", type=Path)
    parser.add_argument("--finetuned-sha256")
    parser.add_argument("--report", type=Path,
                        help="optional new private JSON file; existing files are never overwritten")
    args = parser.parse_args()
    verify_official_source(args.official_source)
    weight_audit = verify_official_checkpoint(args.distilled_checkpoint)
    datasets, data_audit = audit_frozen_data(args.manifest_dir, args.cache_dir)
    result: dict[str, Any] = {"schema": "fsn-videomae2-preparation-v1",
                              "source_commit": OFFICIAL_COMMIT,
                              **weight_audit, "data": data_audit}
    if args.mode != "preflight":
        model, model_audit = build_pretrained_model(args.official_source,
                                                    args.distilled_checkpoint)
        result["model"] = model_audit
        device = torch.device(args.device)
        if args.mode == "smoke":
            result["smoke"] = smoke_bf16_batch(
                model, datasets["train"], device=device, batch_size=args.batch_size,
                workers=args.workers)
        else:
            if args.finetuned_checkpoint is None or args.finetuned_sha256 is None:
                raise BaselinePreflightError("evaluation requires a sealed FSN trained checkpoint and SHA256")
            result["finetuned"] = load_finetuned_fsn_checkpoint(
                model, args.finetuned_checkpoint, args.finetuned_sha256)
            result["evaluation"] = evaluate_single_view(
                model, datasets["val"], device=device, batch_size=args.batch_size,
                workers=args.workers)
    if args.report is not None:
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
