"""Kinetics-400-pretrained torchvision video backbones for FSN comparison."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from experiments.modern_baselines import videomae2 as video_data


VISION_BACKBONES = ("mvit_v2_s", "swin3d_t", "r2plus1d_18", "vjepa2_vitl", "internvideo_next_base")
FOUNDATION_BACKBONES = ("vjepa2_vitl", "internvideo_next_base")


def _replace_last_linear(module: nn.Module, num_classes: int):
    if isinstance(module, nn.Linear):
        if module.out_features != 400:
            raise ValueError(f"expected a Kinetics-400 linear head, got {module.out_features} outputs")
        return nn.Linear(module.in_features, num_classes), True
    if isinstance(module, nn.Sequential):
        children = list(module.children())
        for index in range(len(children) - 1, -1, -1):
            replacement, found = _replace_last_linear(children[index], num_classes)
            if found:
                children[index] = replacement
                return nn.Sequential(*children), True
    return module, False


def _find_last_linear(module: nn.Module) -> nn.Linear:
    if isinstance(module, nn.Linear):
        return module
    if isinstance(module, nn.Sequential):
        for child in reversed(list(module.children())):
            try:
                return _find_last_linear(child)
            except ValueError:
                continue
    raise ValueError("cannot find a linear classifier in selected backbone head")


class KineticsVideoClassifier(nn.Module):
    """Model wrapper for raw RGB caches shaped [B,36,3,224,224], uint8."""

    PREPROCESS = {
        "mvit_v2_s": {
            "mean": (0.45, 0.45, 0.45),
            "std": (0.225, 0.225, 0.225),
            "resize_size": (256, 256),
            "crop_size": 224,
            "geometry_note": "16 uniform frames; resize cached square crop to 256 then center-crop 224; Kinetics MViT normalization",
        },
        "swin3d_t": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "resize_size": (256, 256),
            "crop_size": 224,
            "geometry_note": "16 uniform frames; resize cached square crop to 256 then center-crop 224; Swin3D-T Kinetics normalization",
        },
        "r2plus1d_18": {
            "mean": (0.43216, 0.394666, 0.37645),
            "std": (0.22803, 0.22145, 0.216989),
            "resize_size": (128, 171),
            "crop_size": 112,
            "geometry_note": "16 uniform frames; resize to 128x171 then center-crop 112x112; R(2+1)D Kinetics normalization",
        },
    }

    def __init__(self, name: str, core: nn.Module, classifier_attribute: str, weight_url: str):
        super().__init__()
        if name not in VISION_BACKBONES:
            raise ValueError(f"unsupported backbone: {name}")
        self.name = name
        self.core = core
        self.classifier_attribute = classifier_attribute
        self.weight_url = weight_url
        self.preprocess = self.PREPROCESS[name]
        self.frame_indices = tuple(video_data.FRAME_INDICES)

    def classifier_layer(self) -> nn.Linear:
        return _find_last_linear(getattr(self.core, self.classifier_attribute))

    def set_mode(self, mode: str) -> None:
        if mode not in {"linear_probe", "finetune"}:
            raise ValueError("mode must be linear_probe or finetune")
        if mode == "finetune":
            for parameter in self.core.parameters():
                parameter.requires_grad_(True)
            return
        for parameter in self.core.parameters():
            parameter.requires_grad_(False)
        for parameter in self.classifier_layer().parameters():
            parameter.requires_grad_(True)

    def forward(self, raw_video: torch.Tensor) -> torch.Tensor:
        if not isinstance(raw_video, torch.Tensor) or raw_video.ndim != 5:
            raise ValueError("raw_video must have shape [B,36,3,224,224]")
        if raw_video.dtype != torch.uint8 or tuple(raw_video.shape[1:]) != (36, 3, 224, 224):
            raise ValueError("expected the sealed 36-frame, 224-square uint8 RGB cache")
        indices = torch.as_tensor(self.frame_indices, device=raw_video.device)
        frames = raw_video.index_select(1, indices).float().div_(255.0)
        batch, count = frames.shape[:2]
        resized_height, resized_width = self.preprocess["resize_size"]
        crop_size = self.preprocess["crop_size"]
        flattened = frames.reshape(batch * count, 3, 224, 224)
        flattened = F.interpolate(
            flattened,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        top = (resized_height - crop_size) // 2
        left = (resized_width - crop_size) // 2
        flattened = flattened[:, :, top:top + crop_size, left:left + crop_size]
        frames = flattened.reshape(batch, count, 3, crop_size, crop_size)
        mean = frames.new_tensor(self.preprocess["mean"]).view(1, 1, 3, 1, 1)
        std = frames.new_tensor(self.preprocess["std"]).view(1, 1, 3, 1, 1)
        frames = (frames - mean) / std
        video = frames.permute(0, 2, 1, 3, 4).contiguous()
        logits = self.core(video)
        if logits.ndim != 2 or logits.shape != (raw_video.shape[0], 7):
            raise ValueError(f"{self.name} produced unexpected logits shape {tuple(logits.shape)}")
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError(f"{self.name} produced nonfinite logits")
        return logits


def build_backbone(
    name: str,
    pretrained: bool = True,
    *,
    vjepa2_repo=None,
    vjepa2_checkpoint=None,
    internvideo_next_model_id: str = "revliter/internvideo_next_base_p14_res224_f16",
    internvideo_next_revision: str = "f673b6016189b6dfc4e20ba92e7f01c5a18eb724",
):
    """Construct a comparison backbone with a documented initialization recipe.

    The Kinetics alternatives support linear probing and full fine-tuning.
    V-JEPA 2 and InternVideo-Next are exposed as frozen-encoder linear probes
    only; their upstream training recipes are not reproduced by this runner.
    """
    if name not in VISION_BACKBONES:
        raise ValueError(f"name must be one of {VISION_BACKBONES}")
    if name in FOUNDATION_BACKBONES:
        if not pretrained:
            raise ValueError(f"{name} requires its published pretrained weights")
        from experiments.comparison.foundation_models import build_foundation_backbone

        return build_foundation_backbone(
            name,
            vjepa2_repo=vjepa2_repo,
            vjepa2_checkpoint=vjepa2_checkpoint,
            internvideo_next_model_id=internvideo_next_model_id,
            internvideo_next_revision=internvideo_next_revision,
        )
    try:
        from torchvision.models.video import (
            MViT_V2_S_Weights, R2Plus1D_18_Weights, Swin3D_T_Weights,
            mvit_v2_s, r2plus1d_18, swin3d_t,
        )
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "installed torchvision lacks MViT-V2-S, Swin3D-T, or R(2+1)D-18 video models"
        ) from error

    registry = {
        "mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights.KINETICS400_V1),
        "swin3d_t": (swin3d_t, Swin3D_T_Weights.KINETICS400_V1),
        "r2plus1d_18": (r2plus1d_18, R2Plus1D_18_Weights.KINETICS400_V1),
    }
    constructor, enum_value = registry[name]
    weights = enum_value if pretrained else None
    core = constructor(weights=weights)
    classifier_attribute = next(
        (attribute for attribute in ("fc", "head", "classifier")
         if hasattr(core, attribute)),
        None,
    )
    if classifier_attribute is None:
        raise RuntimeError(f"{name}: no known classifier attribute")
    old_head = getattr(core, classifier_attribute)
    new_head, replaced = _replace_last_linear(old_head, 7)
    if not replaced:
        raise RuntimeError(f"{name}: could not safely replace the 400-class head")
    setattr(core, classifier_attribute, new_head)
    url = enum_value.url if pretrained else "random_initialization"
    model = KineticsVideoClassifier(name, core, classifier_attribute, url)
    audit = {
        "architecture": name,
        "pretrained": pretrained,
        "pretraining": "Kinetics-400 supervised" if pretrained else "random initialization",
        "weights_enum": str(enum_value) if pretrained else None,
        "weights_url": url,
        "classifier_attribute": classifier_attribute,
        "input_frames": 16,
        "source_cache_frames": 36,
        "sample_indices": list(video_data.FRAME_INDICES),
        "preprocessing": model.preprocess,
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model, audit
