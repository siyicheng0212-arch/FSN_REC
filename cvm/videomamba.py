"""Strict feature adapter for the official ECCV 2024 VideoMamba-Ti.

This module imports an explicitly supplied, pinned external checkout. It does
not vendor code, download weights, or replace failed pretrained loading with a
random model. Real inference needs the official CUDA extensions; CPU tests use
a small injected encoder to test the loading contract only.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch
from torch import nn


OFFICIAL_COMMIT = "37355c26d0ae99ca2459f6d4044a5f509031a79f"
OFFICIAL_MODEL_BLOB = "2625505bd72cb40842d1b25abd7f08bccc0a9bfe"
OFFICIAL_MODEL_PATH = "videomamba/video_sm/models/videomamba.py"
OFFICIAL_CHECKPOINT_URL = (
    "https://huggingface.co/OpenGVLab/VideoMamba/resolve/main/"
    "videomamba_t16_k400_f16_res224.pth"
)
OFFICIAL_CHECKPOINT_MIRROR = (
    "https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/"
    "videomamba_t16_k400_f16_res224.pth"
)
FEATURE_DIM = 192
NUM_FRAMES = 16
IMAGE_SIZE = 224


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=False,
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"VideoMamba checkout verification failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _verify_checkout(repo: Path) -> dict[str, str]:
    if not repo.is_dir():
        raise FileNotFoundError(f"VideoMamba checkout unavailable: {repo}")
    if _git(repo, "rev-parse", "HEAD") != OFFICIAL_COMMIT:
        raise ValueError(f"VideoMamba must be checked out at {OFFICIAL_COMMIT}")
    source = repo / OFFICIAL_MODEL_PATH
    if not source.is_file() or _git(repo, "hash-object", str(source)) != OFFICIAL_MODEL_BLOB:
        raise ValueError("VideoMamba model source differs from the inspected official blob")
    # Check the imported model and custom dependency sources, including staged
    # edits and untracked Python files that could shadow the official packages.
    paths = [OFFICIAL_MODEL_PATH, "mamba", "causal-conv1d"]
    if _git(repo, "status", "--porcelain", "--untracked-files=all", "--", *paths):
        raise ValueError("VideoMamba model/custom dependency sources must be clean")
    return {"repo": str(repo), "commit": OFFICIAL_COMMIT,
            "model_blob": OFFICIAL_MODEL_BLOB, "model_source_sha256": _sha256(source)}


def _verify_loaded_dependencies(repo: Path) -> dict[str, Any]:
    """Accept installed copies only if their loaded Python files match the pin.

    This verifies Python implementation provenance, not compiled binary build
    provenance. The CUDA preflight on the training server remains necessary.
    """
    package_roots = {"mamba_ssm": repo / "mamba/mamba_ssm",
                     "causal_conv1d": repo / "causal-conv1d/causal_conv1d"}
    checked = {}
    for name, module in tuple(sys.modules.items()):
        root_name = name.split(".")[0]
        if root_name not in package_roots or module is None:
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        actual = Path(origin).resolve()
        if actual.suffix != ".py":
            continue
        relative = Path(*name.split(".")[1:])
        expected = package_roots[root_name] / relative
        expected = expected / "__init__.py" if actual.name == "__init__.py" else expected.with_suffix(".py")
        if not expected.is_file() or _sha256(actual) != _sha256(expected):
            raise RuntimeError(f"Loaded {name} does not match the pinned VideoMamba dependency source")
        checked[name] = {"path": str(actual), "sha256": _sha256(actual)}
    if "mamba_ssm.modules.mamba_simple" not in checked:
        raise RuntimeError("Official VideoMamba custom Mamba implementation was not imported")
    return {"python_modules": checked,
            "compiled_extension_provenance_verified": False}


def _import_official(repo: Path):
    source = repo / OFFICIAL_MODEL_PATH
    # Loading just this file avoids unrelated model registrations/imports in
    # the official models/__init__.py. CUDA extensions must already be built.
    name = "_fsn_cvm_official_videomamba_" + OFFICIAL_COMMIT
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import VideoMamba source: {source}")
    module = importlib.util.module_from_spec(spec)
    inserted = [str(repo / "mamba"), str(repo / "causal-conv1d")]
    original_path = list(sys.path)
    previous_module = sys.modules.get(name)
    try:
        sys.path[:0] = inserted
        # timm's @register_model consults sys.modules[function.__module__].
        sys.modules[name] = module
        spec.loader.exec_module(module)
    except (ImportError, OSError, AssertionError, ValueError) as exc:
        if previous_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous_module
        raise RuntimeError(
            "Official VideoMamba dependencies/CUDA extensions are unavailable. "
            "Build the pinned checkout's causal-conv1d and mamba packages and "
            "install its compatible timm/einops dependencies; no random fallback is used."
        ) from exc
    except BaseException:
        if previous_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous_module
        raise
    finally:
        sys.path[:] = original_path
    return module


def _extract_state(payload: Any) -> tuple[dict[str, torch.Tensor], str]:
    if not isinstance(payload, Mapping):
        raise ValueError("VideoMamba checkpoint must contain a tensor state_dict")
    container = "raw"
    if not payload or not all(isinstance(key, str) and isinstance(value, torch.Tensor)
                              for key, value in payload.items()):
        candidates = [key for key in ("model", "module", "state_dict")
                      if key in payload and isinstance(payload[key], Mapping)]
        if len(candidates) != 1:
            raise ValueError("Checkpoint requires exactly one model/module/state_dict tensor mapping")
        container = candidates[0]
        payload = payload[container]
    if not payload or not all(isinstance(key, str) and isinstance(value, torch.Tensor)
                              for key, value in payload.items()):
        raise ValueError("Checkpoint model state must contain only named tensors")
    state = dict(payload)
    # Uniform wrapper prefixes only; mixed namespaces are an error below.
    for prefix in ("module.", "backbone.", "encoder."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items()}
    return state, container


def _load_checkpoint(core: nn.Module, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Official VideoMamba local checkpoint unavailable: {path}")
    # Official training checkpoints can include argparse.Namespace. Newer
    # PyTorch can allow that specific metadata class while keeping weights-only
    # loading. Older PyTorch works with tensor-only published model files.
    safe_context = (torch.serialization.safe_globals([argparse.Namespace])
                    if hasattr(torch.serialization, "safe_globals") else nullcontext())
    try:
        with safe_context:
            payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            "Could not load the checkpoint with weights_only=True. Use the official tensor-only "
            "model file, or a PyTorch version supporting safe_globals for official Namespace metadata."
        ) from exc
    state, container = _extract_state(payload)
    expected = core.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = {key: {"checkpoint": list(state[key].shape), "model": list(expected[key].shape)}
                  for key in set(expected) & set(state) if state[key].shape != expected[key].shape}
    if missing or unexpected or mismatched:
        raise ValueError(f"VideoMamba checkpoint is not exact Ti/K400/16-frame/224: "
                         f"missing={missing}, unexpected={unexpected}, shapes={mismatched}")
    if tuple(state.get("head.weight", torch.empty(0)).shape) != (400, FEATURE_DIM):
        raise ValueError("VideoMamba checkpoint must retain the original 400-class K400 head")
    if any(not bool(torch.isfinite(value).all()) for value in state.values() if value.is_floating_point()):
        raise ValueError("VideoMamba checkpoint contains non-finite tensors")
    # First load EVERY original parameter strictly, then remove the pretrained
    # task head. No missing backbone keys, inflation, interpolation or skipping.
    core.load_state_dict(state, strict=True)
    core.head = nn.Identity()
    return {"pretrained": True, "source": str(path), "sha256": _sha256(path),
            "checkpoint_container": container, "strict": True,
            "loaded_tensor_count": len(state), "missing_keys": [], "unexpected_keys": [],
            "discarded_pretrained_classifier": ["head.weight", "head.bias"],
            "pretraining": "ImageNet-1K -> supervised Kinetics-400 (official model-zoo declaration)",
            "official_checkpoint_url": OFFICIAL_CHECKPOINT_URL,
            "publisher_checksum_verified": False,
            "checkpoint_identity_note": "Local SHA256 recorded; publisher checksum unavailable. "
                                        "Shape validation proves compatibility, not training history."}


class VideoMambaFeatureEncoder(nn.Module):
    feature_dim = FEATURE_DIM

    def __init__(self, encoder: nn.Module, load_report: Mapping[str, Any]):
        super().__init__()
        if getattr(encoder, "num_features", None) != FEATURE_DIM:
            raise ValueError("Expected official VideoMamba-Ti feature dimension 192")
        self.encoder = encoder
        self.load_report = dict(load_report)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or tuple(video.shape[1:]) != (3, NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("VideoMamba-Ti requires normalized [B,3,16,224,224] input")
        features = self.encoder.forward_features(video)
        if tuple(features.shape) != (video.shape[0], FEATURE_DIM):
            raise RuntimeError("Official VideoMamba forward_features did not return [B,192]")
        return features


def build_videomamba_tiny16(*, repo_path: str | Path, checkpoint_path: str | Path,
                           seed: int | None = None) -> VideoMambaFeatureEncoder:
    repo = Path(repo_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Official VideoMamba local checkpoint unavailable: {checkpoint}")
    source_report = _verify_checkout(repo)
    official = _import_official(repo)
    dependency_report = _verify_loaded_dependencies(repo)
    context = torch.random.fork_rng(devices=[]) if seed is not None else nullcontext()
    with context:
        if seed is not None:
            torch.manual_seed(seed)
        core = official.videomamba_tiny(pretrained=False, num_classes=400,
                                       num_frames=NUM_FRAMES, img_size=IMAGE_SIZE,
                                       kernel_size=1)
        report = _load_checkpoint(core, checkpoint)
    dependency_report = _verify_loaded_dependencies(repo)
    report.update({"backbone": "videomamba_tiny16", "feature_dim": FEATURE_DIM,
                   "frames": NUM_FRAMES, "crop_size": IMAGE_SIZE,
                   "external_source": source_report, "dependencies": dependency_report})
    return VideoMambaFeatureEncoder(core, report)
