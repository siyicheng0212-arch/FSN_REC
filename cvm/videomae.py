"""Pinned official VideoMAE ViT-B/16-frame feature adapter.

The intended artifact is the original K400-800e *classification-finetuned*
checkpoint, not the masked-pretraining encoder or a VideoMAEv2 distilled
model. Imports and weights are explicit/local; failures never create a random
baseline. Official fixed sinusoidal positions are not state_dict parameters,
so key/shape validation alone cannot authenticate the training frame count.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
from importlib import metadata
import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch
from torch import nn


OFFICIAL_COMMIT = "14ef8d856287c94ef1f985fe30f958eb4ec2c55d"
OFFICIAL_MODEL_BLOB = "66dbe940cb559b06ccc7a5b616e41080fa43b2b9"
OFFICIAL_MODEL_PATH = "modeling_finetune.py"
OFFICIAL_CHECKPOINT_URL = "https://drive.google.com/file/d/18EEgdXY9347yK3Yb28O-GxFMbk41F6Ne/view"
OFFICIAL_MODEL_ZOO_URL = (
    "https://github.com/MCG-NJU/VideoMAE/blob/" + OFFICIAL_COMMIT + "/MODEL_ZOO.md"
)
FEATURE_DIM = 768
NUM_FRAMES = 16
IMAGE_SIZE = 224
TUBELET_SIZE = 2
SUPPORTED_TIMM = ("0.4.8", "0.4.12")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                            text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"VideoMAE checkout verification failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _verify_checkout(repo: Path) -> dict[str, str]:
    if not repo.is_dir():
        raise FileNotFoundError(f"VideoMAE checkout unavailable: {repo}")
    if _git(repo, "rev-parse", "HEAD") != OFFICIAL_COMMIT:
        raise ValueError(f"VideoMAE must be checked out at {OFFICIAL_COMMIT}")
    source = repo / OFFICIAL_MODEL_PATH
    if not source.is_file() or _git(repo, "hash-object", str(source)) != OFFICIAL_MODEL_BLOB:
        raise ValueError("VideoMAE model source differs from the inspected official blob")
    if _git(repo, "status", "--porcelain", "--untracked-files=all", "--", OFFICIAL_MODEL_PATH):
        raise ValueError("VideoMAE imported model source must be clean")
    return {"repo": str(repo), "commit": OFFICIAL_COMMIT,
            "model_blob": OFFICIAL_MODEL_BLOB, "model_source_sha256": _sha256(source)}


def _dependency_report() -> dict[str, Any]:
    try:
        version = metadata.version("timm")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("VideoMAE requires timm==0.4.8 or timm==0.4.12; no random fallback") from exc
    if version not in SUPPORTED_TIMM:
        raise RuntimeError(f"Unsupported VideoMAE timm version {version}; use one of {SUPPORTED_TIMM}")
    return {"timm": version, "torch": torch.__version__,
            "custom_cuda_extension_required": False}


def _import_official(repo: Path):
    # This source depends only on NumPy/PyTorch/timm; the original decoding,
    # DeepSpeed and training modules are deliberately not imported.
    name = "_fsn_cvm_official_videomae_" + OFFICIAL_COMMIT
    spec = importlib.util.spec_from_file_location(name, repo / OFFICIAL_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import pinned VideoMAE model source")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(name)
    sys.modules[name] = module  # required by timm's @register_model
    try:
        spec.loader.exec_module(module)
    except BaseException as exc:
        if previous_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous_module
        if isinstance(exc, (ImportError, OSError, AttributeError)):
            raise RuntimeError("Pinned VideoMAE dependencies could not be imported; install compatible NumPy/timm/PyTorch") from exc
        raise
    return module


def _extract_state(payload: Any) -> tuple[dict[str, torch.Tensor], str]:
    if not isinstance(payload, Mapping):
        raise ValueError("VideoMAE checkpoint must contain a tensor state_dict")
    container = "raw"
    if not payload or not all(isinstance(key, str) and isinstance(value, torch.Tensor)
                              for key, value in payload.items()):
        candidates = [key for key in ("model", "module", "state_dict")
                      if key in payload and isinstance(payload[key], Mapping)]
        if len(candidates) != 1:
            raise ValueError("Checkpoint requires exactly one model/module/state_dict mapping")
        container = candidates[0]
        payload = payload[container]
    if not payload or not all(isinstance(key, str) and isinstance(value, torch.Tensor)
                              for key, value in payload.items()):
        raise ValueError("VideoMAE model state must contain only named tensors")
    state = dict(payload)
    for prefix in ("module.", "backbone.", "encoder."):
        if all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items()}
    return state, container


def _checkpoint_metadata(payload: Any) -> dict[str, Any]:
    args = payload.get("args") if isinstance(payload, Mapping) else None
    args = vars(args) if isinstance(args, argparse.Namespace) else args
    args = args if isinstance(args, Mapping) else {}
    expected = {"model": "vit_base_patch16_224", "num_frames": NUM_FRAMES,
                "num_segments": 1, "tubelet_size": TUBELET_SIZE, "input_size": IMAGE_SIZE,
                "use_mean_pooling": True, "nb_classes": 400, "data_set": "Kinetics-400"}
    observed = {}
    for key, value in expected.items():
        if key in args:
            # Avoid accepting True as the integer 1 or False as the integer 0.
            if type(args[key]) is not type(value) or args[key] != value:
                raise ValueError(f"VideoMAE checkpoint metadata {key}={args[key]!r}, expected {value!r}")
            observed[key] = args[key]
    temporal_keys = ("num_frames", "num_segments", "tubelet_size", "input_size")
    return {"checked_training_args": observed,
            "checkpoint_input_configuration_verified_from_metadata": all(key in observed for key in temporal_keys),
            "checkpoint_input_configuration_note": "Fixed sinusoidal positions are absent from native state_dict; "
                                                   "parameter shapes do not prove source training frame count/resolution."}


def _load_checkpoint(core: nn.Module, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Official VideoMAE local finetuned checkpoint unavailable: {path}")
    safe_context = (torch.serialization.safe_globals([argparse.Namespace])
                    if hasattr(torch.serialization, "safe_globals") else nullcontext())
    try:
        with safe_context:
            payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("Could not load VideoMAE with weights_only=True. Supply tensor-only weights or "
                         "a PyTorch version supporting safe_globals for official Namespace metadata.") from exc
    checkpoint_metadata = _checkpoint_metadata(payload)
    state, container = _extract_state(payload)
    expected = core.state_dict()
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    mismatched = {key: {"checkpoint": list(state[key].shape), "model": list(expected[key].shape)}
                  for key in set(expected) & set(state) if state[key].shape != expected[key].shape}
    if missing or unexpected or mismatched:
        raise ValueError(f"VideoMAE checkpoint is not exact native ViT-B/K400 finetuned: "
                         f"missing={missing}, unexpected={unexpected}, shapes={mismatched}")
    if tuple(state.get("head.weight", torch.empty(0)).shape) != (400, FEATURE_DIM):
        raise ValueError("VideoMAE requires the original 400-class K400 head")
    if any(not bool(torch.isfinite(value).all()) for value in state.values() if value.is_floating_point()):
        raise ValueError("VideoMAE checkpoint contains non-finite tensors")
    core.load_state_dict(state, strict=True)
    core.head = nn.Identity()
    return {"pretrained": True, "source": str(path), "sha256": _sha256(path),
            "checkpoint_container": container, "strict": True, "loaded_tensor_count": len(state),
            "missing_keys": [], "unexpected_keys": [],
            "discarded_pretrained_classifier": ["head.weight", "head.bias"],
            "official_checkpoint_url": OFFICIAL_CHECKPOINT_URL,
            "official_model_zoo_url": OFFICIAL_MODEL_ZOO_URL,
            "declared_checkpoint_route": "K400 self-supervised VideoMAE 800 epochs -> K400 supervised finetuning; no extra data",
            "publisher_checksum_verified": False,
            "checkpoint_identity_note": "Local SHA256 recorded; publisher checksum unavailable. "
                                        "Strict loading proves architecture compatibility, not training history.",
            **checkpoint_metadata}


class VideoMAEFeatureEncoder(nn.Module):
    feature_dim = FEATURE_DIM

    def __init__(self, encoder: nn.Module, load_report: Mapping[str, Any]):
        super().__init__()
        if getattr(encoder, "num_features", None) != FEATURE_DIM or getattr(encoder, "fc_norm", None) is None:
            raise ValueError("Expected official VideoMAE-B 768-dimensional mean-pooled fc_norm features")
        self.encoder = encoder
        self.load_report = dict(load_report)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or tuple(video.shape[1:]) != (3, NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("VideoMAE-B requires normalized [B,3,16,224,224] input")
        features = self.encoder.forward_features(video)
        if tuple(features.shape) != (video.shape[0], FEATURE_DIM):
            raise RuntimeError("Official VideoMAE forward_features did not return [B,768]")
        return features


def build_videomae_base16(*, repo_path: str | Path, checkpoint_path: str | Path,
                         seed: int | None = None) -> VideoMAEFeatureEncoder:
    repo, checkpoint = Path(repo_path).expanduser().resolve(), Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Official VideoMAE local finetuned checkpoint unavailable: {checkpoint}")
    source_report = _verify_checkout(repo)
    dependency_report = _dependency_report()
    official = _import_official(repo)
    context = torch.random.fork_rng(devices=[]) if seed is not None else nullcontext()
    with context:
        if seed is not None:
            # Official factory initializes on CPU. Avoid manual_seed's CUDA
            # side effect, since fork_rng(devices=[]) only restores CPU state.
            torch.default_generator.manual_seed(seed)
        core = official.vit_base_patch16_224(pretrained=False, num_classes=400, all_frames=NUM_FRAMES,
                                            img_size=IMAGE_SIZE, tubelet_size=TUBELET_SIZE,
                                            use_mean_pooling=True, use_learnable_pos_emb=False,
                                            use_checkpoint=False, drop_path_rate=.1, init_scale=.001)
        report = _load_checkpoint(core, checkpoint)
    report.update({"backbone": "videomae_base16", "feature_dim": FEATURE_DIM,
                   "frames": NUM_FRAMES, "crop_size": IMAGE_SIZE, "tubelet_size": TUBELET_SIZE,
                   "feature_readout": "mean of all spatiotemporal tokens, then fc_norm; no CLS token",
                   "drop_path_rate": .1, "activation_checkpointing": False,
                   "external_source": source_report, "dependencies": dependency_report})
    return VideoMAEFeatureEncoder(core, report)
