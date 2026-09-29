"""Atomic epoch-boundary checkpoints for the formal FSN trainer.

Only trusted checkpoints written by this project should be loaded: PyTorch
optimizer and Python/NumPy RNG states require pickle deserialization.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


RESUME_SCHEMA = "fsn-formal-epoch-resume-1.0"
PROTOCOL_FIELDS = (
    "variant", "seed", "sampling", "head_warmup_epochs", "head_warmup_lr",
    "epochs", "patience", "batch_size", "accumulation_steps", "workers",
    "lr", "weight_decay", "global_lr_ratio", "stn_lr_ratio",
    "temporal_lr_ratio", "fsn_module_lr_ratio", "class_weight_mode",
    "clip_grad", "pairwise_loss_weight", "test_after_training",
    "evidence_relation_mode", "disable_evidence_quality_gate",
    "disable_evidence_ambiguity_gate", "allow_random_init",
)


def protocol_fingerprint(
    args: Any, split_audit: Mapping[str, Any], checkpoint_sha256: str | None,
    code_commit: str | None = None,
) -> str:
    """Reject a resumed run if its data, initialization, or training rule changed."""
    settings = {name: getattr(args, name, None) for name in PROTOCOL_FIELDS}
    settings.update({
        "manifest_sha256": split_audit["manifest_sha256"],
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "code_commit": code_commit,
        "torch_version": torch.__version__,
    })
    encoded = json.dumps(settings, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_rng_state(
    sampler_generator: torch.Generator,
    worker_generators: Mapping[str, torch.Generator],
    augmentation_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "sampler": sampler_generator.get_state(),
        "workers": {key: value.get_state() for key, value in worker_generators.items()},
        "augmentation": augmentation_generator.get_state(),
    }


def restore_rng_state(
    state: Mapping[str, Any],
    sampler_generator: torch.Generator,
    worker_generators: Mapping[str, torch.Generator],
    augmentation_generator: torch.Generator,
) -> None:
    if set(state["workers"]) != set(worker_generators):
        raise ValueError("resume worker-generator splits do not match")
    if torch.cuda.is_available() and len(state["torch_cuda"]) != torch.cuda.device_count():
        raise ValueError("resume CUDA-visible device count differs")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    sampler_generator.set_state(state["sampler"])
    for key, generator in worker_generators.items():
        generator.set_state(state["workers"][key])
    augmentation_generator.set_state(state["augmentation"])


def atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    """Keep the preceding checkpoint intact if a save is interrupted."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some filesystems reject directory fsync; atomic replace still
            # protects against a partially written checkpoint file.
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_epoch_checkpoint(path: Path, expected_fingerprint: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("schema_version") != RESUME_SCHEMA:
        raise ValueError(
            "checkpoint is not a resumable last.pt; older best.pt files omit "
            "optimizer, scheduler, and RNG state"
        )
    if state.get("protocol_fingerprint") != expected_fingerprint:
        raise ValueError("resume protocol mismatch: data, weights, seed, or hyperparameters changed")
    required = {
        "model", "optimizer", "phase", "epoch", "best_epoch",
        "best_val_macro_f1", "stale", "history", "rng", "load_report",
        "optimizer_groups", "best_is_current",
    }
    if not required <= set(state):
        raise ValueError(f"resume checkpoint missing: {sorted(required - set(state))}")
    if state["phase"] not in ("head_warmup", "finetune") or not isinstance(state["epoch"], int):
        raise ValueError("resume checkpoint has invalid phase or epoch")
    if state["phase"] == "finetune" and state.get("scheduler") is None:
        raise ValueError("finetune checkpoint is missing scheduler state")
    return state
