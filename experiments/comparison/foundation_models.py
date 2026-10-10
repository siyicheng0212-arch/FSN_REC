"""Frozen linear probes for newer self-supervised video foundation encoders.

The adapters intentionally support linear-probe screening only. Full fine-tuning
must be implemented and validated against each upstream repository's training
recipe before it is advertised as supported.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.modern_baselines import videomae2 as video_data

FOUNDATION_BACKBONES = ("vjepa2_vitl", "internvideo_next_base")
INTERNVIDEO_NEXT_MODEL_ID = "revliter/internvideo_next_base_p14_res224_f16"
INTERNVIDEO_NEXT_REVISION = "f673b6016189b6dfc4e20ba92e7f01c5a18eb724"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_tensor(value):
    """Unwrap the one-view output structure used by the official V-JEPA2 API."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _first_tensor(value[0])
    raise TypeError(
        "upstream video encoder returned an unexpected output structure; "
        "expected a single token tensor"
    )


def _processor_crop_size(processor) -> tuple[int, int]:
    crop = getattr(processor, "crop_size", {"height": 224, "width": 224})
    if isinstance(crop, int):
        return crop, crop
    if isinstance(crop, dict):
        if "height" in crop and "width" in crop:
            return int(crop["height"]), int(crop["width"])
        if "shortest_edge" in crop:
            value = int(crop["shortest_edge"])
            return value, value
    raise ValueError(f"unsupported InternVideo-Next crop_size from processor: {crop!r}")


class FrozenFoundationLinearProbe(nn.Module):
    """Frozen upstream encoder plus the same seven-class linear head.

    Input is the immutable cache format [B,36,3,224,224] in uint8 RGB.
    The same 16 source-cache positions are selected for every model.
    """

    def __init__(
        self,
        name: str,
        core: nn.Module,
        feature_dim: int,
        model_kind: str,
        mean,
        std,
        input_size: tuple[int, int],
        crop_size: tuple[int, int],
        rescale_factor: float,
        audit: dict,
    ):
        super().__init__()
        self.name = name
        self.core = core
        self.head = nn.Linear(int(feature_dim), 7)
        self.model_kind = model_kind
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)
        self.input_size = tuple(int(value) for value in input_size)
        self.crop_size = tuple(int(value) for value in crop_size)
        self.rescale_factor = float(rescale_factor)
        self.frame_indices = tuple(video_data.FRAME_INDICES)
        self.audit = audit
        self.preprocess = {
            "input_frames": 16,
            "source_cache_frames": 36,
            "sample_indices": list(self.frame_indices),
            "resize_size": list(self.input_size),
            "center_crop_size": list(self.crop_size),
            "mean": list(self.mean),
            "std": list(self.std),
            "rescale_factor": self.rescale_factor,
        }

    def classifier_layer(self) -> nn.Linear:
        return self.head

    def set_mode(self, mode: str) -> None:
        if mode != "linear_probe":
            raise ValueError(
                f"{self.name} currently supports frozen linear-probe screening only; "
                "use --mode linear_probe. Full fine-tuning is not implemented."
            )
        for parameter in self.core.parameters():
            parameter.requires_grad_(False)
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)

    def _prepare_video(self, raw_video: torch.Tensor) -> torch.Tensor:
        if not isinstance(raw_video, torch.Tensor) or raw_video.ndim != 5:
            raise ValueError("raw_video must have shape [B,36,3,224,224]")
        if raw_video.dtype != torch.uint8 or tuple(raw_video.shape[1:]) != (36, 3, 224, 224):
            raise ValueError("expected the sealed 36-frame, 224-square uint8 RGB cache")
        indices = torch.as_tensor(self.frame_indices, device=raw_video.device)
        frames = raw_video.index_select(1, indices).float().mul_(self.rescale_factor)
        batch, count = frames.shape[:2]
        flattened = frames.reshape(batch * count, 3, 224, 224)
        if tuple(flattened.shape[-2:]) != self.input_size:
            flattened = F.interpolate(
                flattened,
                size=self.input_size,
                mode="bicubic",
                align_corners=False,
            )
        crop_h, crop_w = self.crop_size
        height, width = flattened.shape[-2:]
        if crop_h > height or crop_w > width:
            raise ValueError(
                f"configured center crop {self.crop_size} exceeds resized frame {(height, width)}"
            )
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
        flattened = flattened[:, :, top : top + crop_h, left : left + crop_w]
        frames = flattened.reshape(batch, count, 3, crop_h, crop_w)
        mean = frames.new_tensor(self.mean).view(1, 1, 3, 1, 1)
        std = frames.new_tensor(self.std).view(1, 1, 3, 1, 1)
        if bool((std <= 0).any()):
            raise ValueError("processor standard deviations must be positive")
        return (frames - mean) / std

    def forward(self, raw_video: torch.Tensor) -> torch.Tensor:
        video = self._prepare_video(raw_video).permute(0, 2, 1, 3, 4).contiguous()
        # Upstream encoders are frozen; no activation graph is retained for the probe.
        with torch.no_grad():
            if self.model_kind == "vjepa2":
                # Official frozen-eval encoder takes [clip][view] nested video tensors.
                tokens = _first_tensor(self.core([[video]]))
            elif self.model_kind == "internvideo_next":
                parameter = next(self.core.parameters())
                tokens = self.core.extract_features(
                    pixel_values=video.to(dtype=parameter.dtype)
                )
                if isinstance(tokens, (tuple, list)):
                    tokens = _first_tensor(tokens)
            else:
                raise RuntimeError(f"unknown foundation model kind: {self.model_kind}")
        if tokens.ndim != 3 or tokens.shape[0] != raw_video.shape[0]:
            raise ValueError(
                f"{self.name} must return [B,num_tokens,feature_dim], got {tuple(tokens.shape)}"
            )
        features = tokens.float().mean(dim=1)
        logits = self.head(features)
        if logits.shape != (raw_video.shape[0], 7) or not bool(torch.isfinite(logits).all()):
            raise FloatingPointError(f"{self.name} produced invalid logits")
        return logits


def _build_vjepa2(repo_path: Path | None, checkpoint_path: Path | None):
    if repo_path is None or checkpoint_path is None:
        raise ValueError(
            "vjepa2_vitl requires both --vjepa2-repo (official facebookresearch/vjepa2 "
            "checkout root) and --vjepa2-checkpoint (local ViT-L checkpoint)"
        )
    repo_path = repo_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not (repo_path / "src" / "models" / "vision_transformer.py").is_file():
        raise FileNotFoundError(f"{repo_path} is not a V-JEPA2 repository root")
    if not (repo_path / "evals" / "video_classification_frozen" / "modelcustom" / "vit_encoder_multiclip.py").is_file():
        raise FileNotFoundError("V-JEPA2 checkout is missing its official frozen-evaluation encoder module")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"V-JEPA2 checkpoint not found: {checkpoint_path}")

    repo_string = str(repo_path)
    if repo_string in sys.path:
        sys.path.remove(repo_string)
    sys.path.insert(0, repo_string)
    try:
        module = importlib.import_module(
            "evals.video_classification_frozen.modelcustom.vit_encoder_multiclip"
        )
    except Exception as error:
        raise RuntimeError(
            "could not import the official V-JEPA2 frozen-evaluation encoder. "
            "Install the dependencies declared by facebookresearch/vjepa2 in the "
            "active environment (commonly including xformers)."
        ) from error

    model_kwargs = {
        "encoder": {
            "checkpoint_key": "target_encoder",
            "model_name": "vit_large",
            "patch_size": 16,
            "tubelet_size": 2,
            "uniform_power": True,
            "use_rope": True,
        }
    }
    wrapper_kwargs = {"max_frames": 128, "use_pos_embed": False}
    try:
        encoder = module.init_module(
            resolution=256,
            frames_per_clip=16,
            checkpoint=str(checkpoint_path),
            model_kwargs=model_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )
    except Exception as error:
        raise RuntimeError(
            "failed to initialize V-JEPA2 ViT-L from the checkpoint. Use the official "
            "ViT-L checkpoint whose target_encoder matches the current V-JEPA2 code."
        ) from error

    try:
        commit = __import__("subprocess").check_output(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            text=True,
            stderr=__import__("subprocess").DEVNULL,
        ).strip()
    except Exception:
        commit = "unknown"
    embed_dim = int(getattr(encoder, "embed_dim"))
    audit = {
        "architecture": "V-JEPA 2 ViT-L/16",
        "pretraining": "self-supervised video representation learning",
        "pretrained": True,
        "source_repository": "https://github.com/facebookresearch/vjepa2",
        "source_repository_commit": commit,
        "checkpoint_filename": checkpoint_path.name,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_key": "target_encoder",
        "feature_pooling": "mean over encoder tokens",
        "probe": "new seven-class linear head; encoder frozen",
        "input_resolution": 256,
        "input_frames": 16,
        "normalization": {"mean": [0.45, 0.45, 0.45], "std": [0.225, 0.225, 0.225]},
    }
    return FrozenFoundationLinearProbe(
        "vjepa2_vitl",
        encoder,
        embed_dim,
        "vjepa2",
        mean=(0.45, 0.45, 0.45),
        std=(0.225, 0.225, 0.225),
        input_size=(256, 256),
        crop_size=(256, 256),
        rescale_factor=1.0 / 255.0,
        audit=audit,
    ), audit


def _build_internvideo_next(model_id: str, revision: str):
    if not model_id.strip():
        raise ValueError("--internvideo-next-model-id must be non-empty")
    if not revision.strip():
        raise ValueError("--internvideo-next-revision must pin a model repository revision")
    try:
        from transformers import AutoModel, VideoMAEImageProcessor
    except ImportError as error:
        raise RuntimeError(
            "internvideo_next_base requires transformers with VideoMAEImageProcessor support"
        ) from error

    try:
        processor = VideoMAEImageProcessor.from_pretrained(model_id, revision=revision)
        encoder = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
    except Exception as error:
        raise RuntimeError(
            "failed to load the pinned InternVideo-Next Base checkpoint. Check Hugging Face "
            "network/cache access and install the model's declared Transformers dependencies."
        ) from error

    if not hasattr(encoder, "extract_features"):
        raise TypeError("loaded InternVideo-Next model does not expose extract_features")
    if not bool(getattr(processor, "do_rescale", True)):
        raise ValueError("InternVideo-Next adapter expects processor rescaling from uint8 RGB")
    if not bool(getattr(processor, "do_normalize", True)):
        raise ValueError("InternVideo-Next adapter expects processor normalization")

    mean = getattr(processor, "image_mean", None)
    std = getattr(processor, "image_std", None)
    if not mean or not std or len(mean) != 3 or len(std) != 3:
        raise ValueError("unsupported image_mean/image_std in InternVideo-Next processor config")
    size = getattr(processor, "size", {"shortest_edge": 224})
    if isinstance(size, int):
        input_size = (size, size)
    elif isinstance(size, dict) and "height" in size and "width" in size:
        input_size = (int(size["height"]), int(size["width"]))
    elif isinstance(size, dict) and "shortest_edge" in size:
        edge = int(size["shortest_edge"])
        input_size = (edge, edge)
    else:
        raise ValueError(f"unsupported InternVideo-Next resize config: {size!r}")
    # The FSN cache is already square; common VideoMAE processor settings use
    # the same 224x224 crop. Refuse silent geometric changes outside that contract.
    crop_size = _processor_crop_size(processor)
    if input_size != (224, 224) or crop_size != (224, 224):
        raise ValueError(
            f"cached-clip adapter expects the pinned processor's 224x224 geometry; "
            f"got resize={input_size}, crop={crop_size}"
        )
    feature_dim = int(
        getattr(
            encoder.config,
            "hidden_size",
            getattr(encoder.config, "embed_dim", 768),
        )
    )
    audit = {
        "architecture": "InternVideo-Next Base, patch-14, 16 frames",
        "pretraining": "self-supervised video representation learning without video-text supervision",
        "pretrained": True,
        "source_repository": "https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo-Next",
        "checkpoint_model_id": model_id,
        "checkpoint_revision": revision,
        "checkpoint_provenance_note": (
            "The OpenGVLab InternVideo-Next Hugging Face collection lists this model ID, "
            "but the hosting account is revliter; cite and report the exact checkpoint owner."
        ),
        "feature_pooling": "mean over extract_features tokens",
        "probe": "new seven-class linear head; encoder frozen",
        "input_resolution": list(input_size),
        "input_frames": 16,
        "normalization": {"mean": list(mean), "std": list(std)},
    }
    return FrozenFoundationLinearProbe(
        "internvideo_next_base",
        encoder,
        feature_dim,
        "internvideo_next",
        mean=mean,
        std=std,
        input_size=input_size,
        crop_size=crop_size,
        rescale_factor=float(getattr(processor, "rescale_factor", 1.0 / 255.0)),
        audit=audit,
    ), audit


def build_foundation_backbone(
    name: str,
    *,
    vjepa2_repo: Path | None = None,
    vjepa2_checkpoint: Path | None = None,
    internvideo_next_model_id: str = INTERNVIDEO_NEXT_MODEL_ID,
    internvideo_next_revision: str = INTERNVIDEO_NEXT_REVISION,
):
    if name == "vjepa2_vitl":
        return _build_vjepa2(vjepa2_repo, vjepa2_checkpoint)
    if name == "internvideo_next_base":
        return _build_internvideo_next(internvideo_next_model_id, internvideo_next_revision)
    raise ValueError(f"unsupported foundation backbone: {name}")
