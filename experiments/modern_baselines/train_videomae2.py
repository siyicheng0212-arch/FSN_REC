"""Single-GPU, resumable FSN training for the official distilled VideoMAE V2.

Only the byte-frozen train7372/val823 development protocol is accepted. A
fresh run performs a real bf16 cached-batch smoke before its first optimizer
step. Resume starts after the last fully committed epoch, replaying an
interrupted epoch from that point; no test manifest is opened.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.aligned_protocol import DATA_PROTOCOL, EXPECTED_COUNTS, EXPECTED_MANIFEST_SHA
from experiments.metrics import DEFAULT_CLASS_NAMES
from experiments.modern_baselines import videomae2 as base


RUN_SCHEMA = "fsn-videomae2-distilled-vitb-train-v1"
RESUME_SCHEMA = "fsn-videomae2-distilled-vitb-resume-v1"
SELECTION = "best_finetune_val823_macro_f1_strict_improvement"
_CANDIDATE_RE = re.compile(r"best_candidate_epoch_[0-9]{3}\.pt\Z")


@dataclass(frozen=True)
class TrainConfig:
    # Common comparison protocol from train_aligned_context.py / run_aligned_context.py.
    seed: int = 42
    batch_size: int = 4
    accumulation_steps: int = 16
    warmup_epochs: int = 5
    epochs: int = 100
    patience: int = 10
    workers: int = 4
    head_warmup_lr: float = 0.001
    clip_grad: float = 20.0
    class_weight_mode: str = "sqrt_inverse"
    horizontal_flip_probability: float = 0.5
    # ViT-specific: upstream VideoMAEv2 K710 script uses AdamW + layer decay.
    base_lr: float = 0.000175  # upstream 7e-4 scaled from batch 256 to FSN effective 64
    weight_decay: float = 0.05
    layer_decay: float = 0.75

    @property
    def total_epochs(self) -> int:
        return self.warmup_epochs + self.epochs


FORMAL_CONFIG = TrainConfig()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile("wb", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _event(output_dir: Path, event: str, **fields: Any) -> None:
    row = {"time_utc": _utc(), "event": event, **fields}
    with (output_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)


def _status(output_dir: Path, status: str, *, exit_code: int | None,
            completed_epochs: int, resume_available: bool,
            error_type: str | None = None) -> None:
    _atomic_json(output_dir / "status.json", {
        "schema": RUN_SCHEMA, "status": status, "exit_code": exit_code,
        "completed_epochs": completed_epochs, "resume_available": resume_available,
        "training_completed": status == "completed", "error_type": error_type,
        "time_utc": _utc(), "test_read": False,
    })


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rng_state(shuffle: torch.Generator, augmentation: torch.Generator) -> dict[str, Any]:
    numpy = np.random.get_state()
    return {
        "python": random.getstate(), "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all(),
        "numpy": {"kind": numpy[0], "keys": numpy[1].tolist(),
                  "position": numpy[2], "has_gauss": numpy[3],
                  "cached_gaussian": numpy[4]},
        "shuffle": shuffle.get_state(), "augmentation": augmentation.get_state(),
    }


def _restore_rng(state: dict[str, Any], shuffle: torch.Generator,
                 augmentation: torch.Generator) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    numpy = state["numpy"]
    np.random.set_state((numpy["kind"], np.asarray(numpy["keys"], dtype=np.uint32),
                         numpy["position"], numpy["has_gauss"], numpy["cached_gaussian"]))
    shuffle.set_state(state["shuffle"])
    augmentation.set_state(state["augmentation"])


def _source_seal() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    files = [Path(__file__), Path(base.__file__),
             repo / "experiments" / "aligned_data.py",
             repo / "experiments" / "aligned_protocol.py",
             repo / "experiments" / "full_data.py",
             repo / "experiments" / "pilot_data.py",
             repo / "experiments" / "metrics.py"]
    return {str(path.relative_to(repo)): base.sha256_file(path) for path in files}


def _identity(config: TrainConfig, data_audit: dict[str, Any],
              device: torch.device) -> dict[str, Any]:
    identity = {
        "schema": RUN_SCHEMA, "config": asdict(config), "data": data_audit,
        "source_sha256": _source_seal(),
        "official_source_commit": base.OFFICIAL_COMMIT,
        "official_model_sha256": base.OFFICIAL_MODEL_SHA256,
        "pretrained_checkpoint_sha256": base.OFFICIAL_CHECKPOINT_SHA256,
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "timm_version": base.OFFICIAL_TIMM_VERSION,
        "device": base._device_report(device),
    }
    return {**identity, "identity_sha256": base._object_sha256(identity)}


def _layer_id(name: str) -> int:
    if name.startswith("patch_embed."):
        return 0
    if name.startswith("blocks."):
        pieces = name.split(".")
        if len(pieces) < 3 or not pieces[1].isdigit() or not 0 <= int(pieces[1]) < 12:
            raise base.BaselinePreflightError(f"unexpected transformer parameter: {name}")
        return int(pieces[1]) + 1
    if name.startswith(("fc_norm.", "head.")):
        return 13
    raise base.BaselinePreflightError(f"unexpected ViT-B parameter group: {name}")


def make_optimizer_scheduler(model: torch.nn.Module, config: TrainConfig):
    """AdamW with official-style ViT layer decay and exact parameter coverage."""
    grouped: dict[tuple[int, bool, bool], dict[str, Any]] = {}
    seen = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            raise base.BaselinePreflightError("duplicate optimizer parameter")
        seen.add(id(parameter))
        layer = _layer_id(name)
        no_decay = parameter.ndim == 1 or name.endswith(".bias")
        is_head = name.startswith("head.")
        key = (layer, no_decay, is_head)
        scale = config.layer_decay ** (13 - layer)
        if key not in grouped:
            finetune_decay = 0.0 if no_decay else config.weight_decay
            grouped[key] = {"params": [], "lr": config.base_lr * scale,
                            "weight_decay": finetune_decay,
                            "finetune_weight_decay": finetune_decay,
                            "name": f"layer{layer}_{'head' if is_head else 'backbone'}_"
                                    f"{'nodecay' if no_decay else 'decay'}",
                            "layer_id": layer, "lr_scale": scale,
                            "head": is_head}
        grouped[key]["params"].append(parameter)
    if seen != {id(parameter) for parameter in model.parameters()}:
        raise base.BaselinePreflightError("optimizer parameter coverage mismatch")
    groups = [grouped[key] for key in sorted(grouped)]
    optimizer = torch.optim.AdamW(groups, lr=config.base_lr, betas=(0.9, 0.999))

    def schedule_for(group: dict[str, Any]):
        def multiplier(epoch: int) -> float:
            if epoch < config.warmup_epochs:
                return config.head_warmup_lr / (config.base_lr * group["lr_scale"]) \
                    if group["head"] else 0.0
            finetune_epoch = min(epoch - config.warmup_epochs, config.epochs)
            return 0.5 * (1.0 + math.cos(math.pi * finetune_epoch / config.epochs))
        return multiplier

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[schedule_for(group) for group in groups])
    summary = [{"name": group["name"], "layer_id": group["layer_id"],
                "lr_scale": group["lr_scale"], "weight_decay": group["weight_decay"],
                "parameter_tensors": len(group["params"]),
                "parameters": sum(value.numel() for value in group["params"])}
               for group in groups]
    return optimizer, scheduler, summary


def _set_phase(model: torch.nn.Module, phase: str,
               optimizer: torch.optim.Optimizer | None = None) -> None:
    if phase not in {"head_warmup", "finetune"}:
        raise ValueError("unknown training phase")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(phase == "finetune" or name.startswith("head."))
    if optimizer is not None:
        for group in optimizer.param_groups:
            group["weight_decay"] = (0.0 if phase == "head_warmup"
                                     else group["finetune_weight_decay"])


def _class_weights(train_dataset: base.SingleViewDataset, device: torch.device,
                   mode: str) -> torch.Tensor:
    counts = Counter(record.label_id for record in train_dataset.base.records)
    if set(counts) != set(range(7)) or sum(counts.values()) != len(train_dataset):
        raise base.BaselinePreflightError("invalid frozen train class distribution")
    inverse = torch.tensor([len(train_dataset) / (7 * counts[index]) for index in range(7)],
                           dtype=torch.float32, device=device)
    if mode == "sqrt_inverse":
        return inverse.sqrt()
    if mode == "none":
        return torch.ones_like(inverse)
    raise base.BaselinePreflightError("unsupported class weight mode")


def train_epoch(model: torch.nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer, weights: torch.Tensor,
                device: torch.device, config: TrainConfig,
                augmentation: torch.Generator) -> dict[str, Any]:
    """Each clip once; divide accumulated weighted CE by the full window mass."""
    model.train()
    iterator = iter(loader)
    visited: set[str] = set()
    samples = steps = 0
    weighted_ce_sum = weight_mass_sum = 0.0
    first_step_gradients = None
    effective_batches = []
    started = time.perf_counter()
    while True:
        window = []
        for _ in range(config.accumulation_steps):
            item = next(iterator, None)
            if item is None:
                break
            identifiers = item["clip_id"]
            if any(identifier in visited for identifier in identifiers):
                raise base.BaselinePreflightError("duplicate clip during train epoch")
            visited.update(identifiers)
            window.append(item)
        if not window:
            break
        window_samples = sum(len(item["label"]) for item in window)
        window_mass = sum(float(weights[item["label"].to(device)].sum().item())
                          for item in window)
        if window_mass <= 0 or not math.isfinite(window_mass):
            raise base.BaselinePreflightError("nonpositive class-weight mass")
        optimizer.zero_grad(set_to_none=True)
        for item in window:
            videos = item["video"]
            if torch.rand((), generator=augmentation).item() < config.horizontal_flip_probability:
                videos = videos.flip(-1)
            videos = videos.to(device, non_blocking=device.type == "cuda")
            targets = item["label"].to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(videos)
                if tuple(logits.shape) != (len(targets), 7):
                    raise base.BaselinePreflightError("train logits are not [batch,7]")
                numerator = F.cross_entropy(logits.float(), targets, weight=weights,
                                            reduction="sum")
                loss = numerator / window_mass
            if not torch.isfinite(loss).item() or not torch.isfinite(logits).all().item():
                raise base.BaselinePreflightError("nonfinite training loss/logits")
            loss.backward()
            weighted_ce_sum += float(numerator.detach().item())
            samples += len(targets)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        if not torch.isfinite(gradient_norm).item():
            raise base.BaselinePreflightError("nonfinite full-model gradient norm")
        if first_step_gradients is None:
            first_step_gradients = {
                "head": None if model.head.weight.grad is None else
                float(model.head.weight.grad.float().norm().item()),
                "patch_embed": None if model.patch_embed.proj.weight.grad is None else
                float(model.patch_embed.proj.weight.grad.float().norm().item()),
            }
        optimizer.step()
        weight_mass_sum += window_mass
        steps += 1
        effective_batches.append(window_samples)
    if samples != len(loader.dataset) or len(visited) != len(loader.dataset):
        raise base.BaselinePreflightError("training epoch did not cover all frozen train clips")
    expected_steps = math.ceil(math.ceil(len(loader.dataset) / config.batch_size) /
                               config.accumulation_steps)
    if steps != expected_steps:
        raise base.BaselinePreflightError("training optimizer step count differs from full coverage")
    return {"train_weighted_ce": weighted_ce_sum / weight_mass_sum,
            "samples_seen": samples, "optimizer_steps": steps,
            "effective_batch_min": min(effective_batches),
            "effective_batch_max": max(effective_batches),
            "first_step_gradients": first_step_gradients,
            "train_seconds": time.perf_counter() - started}


def _checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LambdaLR,
                shuffle: torch.Generator, augmentation: torch.Generator,
                identity: dict[str, Any], *, completed_epochs: int,
                optimizer_steps: int, finetune_optimizer_steps: int,
                best_score: float | None, best_epoch: int | None,
                best_candidate: str | None, best_candidate_sha256: str | None,
                stale: int, history: list[dict[str, Any]], peak_cuda_bytes: int) -> dict[str, Any]:
    return {"schema": RESUME_SCHEMA, "identity_sha256": identity["identity_sha256"],
            "completed_epochs": completed_epochs, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": _rng_state(shuffle, augmentation),
            "optimizer_steps": optimizer_steps,
            "finetune_optimizer_steps": finetune_optimizer_steps,
            "best_score": best_score, "best_epoch": best_epoch,
            "best_candidate": best_candidate,
            "best_candidate_sha256": best_candidate_sha256,
            "stale": stale, "history": history, "peak_cuda_bytes": peak_cuda_bytes,
            "training_completed": False}


def _restore_checkpoint(output_dir: Path, model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        scheduler: torch.optim.lr_scheduler.LambdaLR,
                        shuffle: torch.Generator, augmentation: torch.Generator,
                        identity: dict[str, Any], config: TrainConfig) -> dict[str, Any]:
    path = output_dir / "last.pt"
    if not path.is_file():
        raise base.BaselinePreflightError("resume checkpoint last.pt is absent")
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise base.BaselinePreflightError("resume checkpoint is unreadable") from exc
    epoch = saved.get("completed_epochs") if isinstance(saved, dict) else None
    if not isinstance(saved, dict) or saved.get("schema") != RESUME_SCHEMA or \
            saved.get("identity_sha256") != identity["identity_sha256"] or \
            isinstance(epoch, bool) or not isinstance(epoch, int) or \
            not 0 <= epoch <= config.total_epochs or \
            not isinstance(saved.get("history"), list) or len(saved["history"]) != epoch:
        raise base.BaselinePreflightError("resume checkpoint protocol/epoch/history mismatch")
    candidate = saved.get("best_candidate")
    if candidate is not None:
        if not isinstance(candidate, str) or not _CANDIDATE_RE.fullmatch(candidate):
            raise base.BaselinePreflightError("unsafe best-candidate reference in resume checkpoint")
        if base.sha256_file(output_dir / candidate) != saved.get("best_candidate_sha256"):
            raise base.BaselinePreflightError("saved best candidate changed or is absent")
    elif saved.get("best_epoch") is not None or saved.get("best_score") is not None:
        raise base.BaselinePreflightError("resume best selection state is incomplete")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    if scheduler.last_epoch != epoch:
        raise base.BaselinePreflightError("resume scheduler epoch disagrees with model checkpoint")
    _restore_rng(saved["rng"], shuffle, augmentation)
    return saved


def _assert_cuda(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise base.BaselinePreflightError("formal VideoMAE V2 training requires CUDA")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise base.BaselinePreflightError("selected CUDA GPU lacks bf16 support")


def _lock_run(output_dir: Path):
    """Return an exclusive OS lock handle, held until the caller closes it."""
    handle = (output_dir / ".trainer.lock").open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise base.BaselinePreflightError("another trainer already owns this output directory") from exc
    return handle


def run(*, official_source: Path, distilled_checkpoint: Path,
        manifest_dir: Path, cache_dir: Path, output_dir: Path,
        device: torch.device, resume: bool = False,
        config: TrainConfig = FORMAL_CONFIG) -> dict[str, Any]:
    """Run or explicitly resume one isolated seed-42 VideoMAE V2 experiment."""
    if config != FORMAL_CONFIG:
        raise base.BaselinePreflightError("formal run configuration is fixed; use only registered settings")
    _assert_cuda(device)
    official_source = Path(official_source).resolve()
    distilled_checkpoint = Path(distilled_checkpoint).resolve()
    manifest_dir, cache_dir = Path(manifest_dir).resolve(), Path(cache_dir).resolve()
    output_dir = Path(output_dir).resolve()
    repo = Path(__file__).resolve().parents[2]
    protected = (repo, official_source, distilled_checkpoint.parent, manifest_dir, cache_dir)
    if output_dir == Path(output_dir.anchor) or any(
            output_dir == root or root in output_dir.parents or output_dir in root.parents
            for root in protected):
        raise base.BaselinePreflightError("private training output must be outside source and inputs")
    base.verify_official_source(official_source)
    base.verify_official_checkpoint(distilled_checkpoint)
    datasets, data_audit = base.audit_frozen_data(manifest_dir, cache_dir)
    identity = _identity(config, data_audit, device)
    if resume:
        if not output_dir.is_dir():
            raise base.BaselinePreflightError("resume output directory is absent")
        stored = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
        if stored != identity:
            raise base.BaselinePreflightError("resume inputs, code, runtime or fixed settings changed")
        status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
        if status.get("status") == "completed":
            raise base.BaselinePreflightError("completed run cannot be resumed")
    elif output_dir.exists():
        raise base.BaselinePreflightError("fresh private output directory already exists")

    _seed_all(config.seed)
    model, load_report = base.build_pretrained_model(official_source, distilled_checkpoint)
    if not resume:
        output_dir.mkdir(parents=True, exist_ok=False)
    with _lock_run(output_dir):
        completed_epochs = 0
        _status(output_dir, "running", exit_code=None, completed_epochs=0,
                resume_available=resume)
        try:
            if not resume:
                _atomic_json(output_dir / "run_config.json", identity)
                _atomic_json(output_dir / "load_report.json", load_report)
                smoke = base.smoke_bf16_batch(model, datasets["train"], device=device,
                                              batch_size=config.batch_size, workers=0)
                _atomic_json(output_dir / "smoke.json", smoke)
                _event(output_dir, "smoke_passed", batch_size=config.batch_size,
                       peak_cuda_allocated_bytes=smoke["peak_cuda_allocated_bytes"])
                _seed_all(config.seed)  # smoke never changes formal initialization/RNG
            model.to(device)
            optimizer, scheduler, optimizer_groups = make_optimizer_scheduler(model, config)
            shuffle = torch.Generator().manual_seed(config.seed)
            augmentation = torch.Generator().manual_seed(config.seed + 1)
            weights = _class_weights(datasets["train"], device, config.class_weight_mode)
            loader = DataLoader(datasets["train"], batch_size=config.batch_size, shuffle=True,
                                num_workers=config.workers, pin_memory=True, drop_last=False,
                                persistent_workers=False, generator=shuffle)
            if resume:
                saved = _restore_checkpoint(output_dir, model, optimizer, scheduler,
                                            shuffle, augmentation, identity, config)
                completed_epochs = saved["completed_epochs"]
                history = saved["history"]
                optimizer_steps = saved["optimizer_steps"]
                finetune_steps = saved["finetune_optimizer_steps"]
                best_score, best_epoch, stale = saved["best_score"], saved["best_epoch"], saved["stale"]
                best_candidate, candidate_sha = saved["best_candidate"], saved["best_candidate_sha256"]
                peak_cuda_bytes = saved["peak_cuda_bytes"]
                _atomic_json(output_dir / "history.json", history)
                _event(output_dir, "resumed", completed_epochs=completed_epochs)
            else:
                history = []
                optimizer_steps = finetune_steps = stale = 0
                best_score = best_epoch = best_candidate = candidate_sha = None
                peak_cuda_bytes = 0
                initial = _checkpoint(model, optimizer, scheduler, shuffle, augmentation,
                                      identity, completed_epochs=0, optimizer_steps=0,
                                      finetune_optimizer_steps=0, best_score=None,
                                      best_epoch=None, best_candidate=None,
                                      best_candidate_sha256=None, stale=0, history=[],
                                      peak_cuda_bytes=0)
                _atomic_torch_save(output_dir / "last.pt", initial)
                _atomic_json(output_dir / "history.json", [])
                _event(output_dir, "training_started", seed=config.seed)
            _status(output_dir, "running", exit_code=None,
                    completed_epochs=completed_epochs, resume_available=True)
            remaining = (range(0) if stale >= config.patience and best_epoch is not None
                         else range(completed_epochs + 1, config.total_epochs + 1))
            for epoch in remaining:
                phase = "head_warmup" if epoch <= config.warmup_epochs else "finetune"
                _set_phase(model, phase, optimizer)
                row = train_epoch(model, loader, optimizer, weights, device, config, augmentation)
                val = base._evaluate_single_view_unsealed(
                    model, datasets["val"], device=device, batch_size=config.batch_size,
                    workers=config.workers)
                score = val["metrics"]["all"]["macro_f1"]
                if not math.isfinite(score):
                    raise base.BaselinePreflightError("nonfinite val823 macro-F1")
                optimizer_steps += row["optimizer_steps"]
                if phase == "finetune":
                    finetune_steps += row["optimizer_steps"]
                row.update(epoch=epoch, phase=phase, val_macro_f1=score,
                           val_accuracy=val["metrics"]["all"]["accuracy"],
                           val_seconds=val["elapsed_seconds"],
                           lr_by_group={group["name"]: group["lr"]
                                        for group in optimizer.param_groups})
                previous_candidate = best_candidate
                if phase == "finetune":
                    if best_score is None or score > best_score:
                        best_score, best_epoch, stale = score, epoch, 0
                        best_candidate = f"best_candidate_epoch_{epoch:03d}.pt"
                        _atomic_torch_save(output_dir / best_candidate, {
                            "model": {key: value.detach().cpu().clone()
                                      for key, value in model.state_dict().items()},
                            "epoch": epoch, "val_macro_f1": score,
                            "optimizer_steps": optimizer_steps,
                            "finetune_optimizer_steps": finetune_steps,
                            "identity_sha256": identity["identity_sha256"]})
                        candidate_sha = base.sha256_file(output_dir / best_candidate)
                    else:
                        stale += 1
                history.append(row)
                scheduler.step()
                peak_cuda_bytes = max(peak_cuda_bytes, torch.cuda.max_memory_allocated(device))
                payload = _checkpoint(model, optimizer, scheduler, shuffle, augmentation,
                                      identity, completed_epochs=epoch,
                                      optimizer_steps=optimizer_steps,
                                      finetune_optimizer_steps=finetune_steps,
                                      best_score=best_score, best_epoch=best_epoch,
                                      best_candidate=best_candidate,
                                      best_candidate_sha256=candidate_sha, stale=stale,
                                      history=history, peak_cuda_bytes=peak_cuda_bytes)
                _atomic_torch_save(output_dir / "last.pt", payload)
                completed_epochs = epoch
                _atomic_json(output_dir / "history.json", history)
                _status(output_dir, "running", exit_code=None,
                        completed_epochs=epoch, resume_available=True)
                _event(output_dir, "epoch_completed", epoch=epoch, phase=phase,
                       train_weighted_ce=row["train_weighted_ce"],
                       val_macro_f1=score, best_epoch=best_epoch,
                       optimizer_steps=optimizer_steps)
                if previous_candidate and previous_candidate != best_candidate:
                    (output_dir / previous_candidate).unlink(missing_ok=True)
                if phase == "finetune" and stale >= config.patience:
                    _event(output_dir, "early_stopping", epoch=epoch, stale=stale)
                    break
            if best_candidate is None or best_epoch is None or finetune_steps <= 0:
                raise base.BaselinePreflightError("no completed ViT backbone finetune epoch exists")
            if base.sha256_file(output_dir / best_candidate) != candidate_sha:
                raise base.BaselinePreflightError("saved best candidate changed before final replay")
            candidate = torch.load(output_dir / best_candidate, map_location="cpu", weights_only=True)
            if candidate.get("identity_sha256") != identity["identity_sha256"] or \
                    candidate.get("epoch") != best_epoch or \
                    candidate.get("val_macro_f1") != best_score or \
                    not 0 < candidate.get("finetune_optimizer_steps", 0) <= finetune_steps or \
                    not 0 < candidate.get("optimizer_steps", 0) <= optimizer_steps:
                raise base.BaselinePreflightError("best candidate provenance differs from resume state")
            model.load_state_dict(candidate["model"], strict=True)
            replay = base._evaluate_single_view_unsealed(
                model, datasets["val"], device=device, batch_size=config.batch_size,
                workers=config.workers)
            if not math.isclose(replay["metrics"]["all"]["macro_f1"],
                                best_score, rel_tol=0, abs_tol=1e-12):
                raise base.BaselinePreflightError("saved-best val823 replay differs from selection")
            best = {
                "schema": base.FINETUNED_SCHEMA, "model": candidate["model"],
                "training_completed": True, "epoch": best_epoch,
                "optimizer_steps": candidate["optimizer_steps"],
                "finetune_optimizer_steps": candidate["finetune_optimizer_steps"],
                "selection": SELECTION, "val_macro_f1": best_score,
                "pretrained_checkpoint_sha256": base.OFFICIAL_CHECKPOINT_SHA256,
                "official_source_commit": base.OFFICIAL_COMMIT,
                "data_protocol": DATA_PROTOCOL,
                "manifest_sha256": EXPECTED_MANIFEST_SHA,
                "class_names": list(DEFAULT_CLASS_NAMES),
                "run_identity_sha256": identity["identity_sha256"],
            }
            _atomic_torch_save(output_dir / "best.pt", best)
            best_sha = base.sha256_file(output_dir / "best.pt")
            replay["finetuned_checkpoint_sha256"] = best_sha
            evaluation = replay
            result = {
                "schema": RUN_SCHEMA, "training_completed": True,
                "data_protocol": DATA_PROTOCOL, "seed": config.seed,
                "identity_sha256": identity["identity_sha256"],
                "best_epoch": best_epoch, "best_val_macro_f1": best_score,
                "best_checkpoint_sha256": best_sha,
                "finetune_optimizer_steps": finetune_steps,
                "optimizer_steps": optimizer_steps,
                "best_checkpoint_optimizer_steps": candidate["optimizer_steps"],
                "epochs_completed": completed_epochs,
                "early_stopped": stale >= config.patience,
                "validation": evaluation,
                "parameters": {"total": sum(p.numel() for p in model.parameters()),
                               "trainable_finetune": sum(p.numel() for p in model.parameters()),
                               "optimizer_groups": optimizer_groups},
                "resources": {"train_seconds": sum(row["train_seconds"] for row in history),
                              "val_seconds": sum(row["val_seconds"] for row in history),
                              "peak_cuda_allocated_bytes": peak_cuda_bytes,
                              "device": base._device_report(device)},
                "validation_role": "val823 development; not independent test",
                "test_read": False,
            }
            _atomic_json(output_dir / "result.json", result)
            _status(output_dir, "completed", exit_code=0,
                    completed_epochs=completed_epochs, resume_available=False)
            _event(output_dir, "completed", exit_code=0, best_epoch=best_epoch,
                   best_val_macro_f1=best_score)
            return result
        except BaseException as error:
            exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
            _status(output_dir, "interrupted" if exit_code == 130 else "failed",
                    exit_code=exit_code, completed_epochs=completed_epochs,
                    resume_available=(output_dir / "last.pt").is_file(),
                    error_type=type(error).__name__)
            _event(output_dir, "failed", exit_code=exit_code,
                   error_type=type(error).__name__)
            raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--distilled-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        result = run(official_source=args.official_source,
                     distilled_checkpoint=args.distilled_checkpoint,
                     manifest_dir=args.manifest_dir, cache_dir=args.cache_dir,
                     output_dir=args.output_dir, device=torch.device(args.device),
                     resume=args.resume)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    print(json.dumps({"status": "completed", "exit_code": 0,
                      "best_epoch": result["best_epoch"],
                      "best_val_macro_f1": result["best_val_macro_f1"]},
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
