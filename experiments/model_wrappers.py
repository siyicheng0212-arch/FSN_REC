"""Unified interfaces for the three FSN comparison models."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
ADAFOCUS_CANDIDATES = (
    ROOT
    / "third_party"
    / "Uni-AdaFocus"
    / "Uni-AdaFocus-TSM with Experiments on Sth-Sth V1&V2 and Jester",
    ROOT / "models" / "Uni-AdaFocus-TSM-FSN",
)
ADAFOCUS_ROOT = next((path for path in ADAFOCUS_CANDIDATES if path.is_dir()), None)
if ADAFOCUS_ROOT is None:
    raise RuntimeError("could not locate the Uni-AdaFocus-TSM-FSN source tree")
if str(ADAFOCUS_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAFOCUS_ROOT))

from archs.uni_adafocus_tsm import AdaFocus  # noqa: E402


class FSNModel(nn.Module):
    """Small contract used by the unified train/evaluation runner."""

    model_name: str

    def compute_loss(self, output: dict[str, Any], target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(output["logits"], target)


def _adafocus_args(
    device: torch.device,
    modified: bool,
    num_glance_segments: int,
    num_input_focus_segments: int,
    num_focus_segments: int,
    patch_size: int,
    mc_sample_times: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        num_glance_segments=num_glance_segments,
        num_input_focus_segments=num_input_focus_segments,
        num_focus_segments=num_focus_segments,
        input_size=224,
        patch_size=patch_size,
        feature_map_channels=1280,
        device=device,
        modality="RGB",
        consensus_type="avg",
        dropout=0.5,
        img_feature_dim=256,
        no_partialbn=False,
        pretrain="none",
        shift=True,
        shift_div=8,
        shift_place="blockres",
        tune_from=None,
        dataset="fsn",
        temporal_pool=False,
        non_local=False,
        stn_hidden_dim=32,
        temporal_hidden_dim=32,
        fsn_local_adapter="temporal" if modified else "none",
        fsn_interaction="cross_attention" if modified else "none",
        fsn_interaction_weight=0.2,
        fsn_local_grid_size=3,
        fsn_adapter_dim=64,
        fsn_interaction_dim=64,
        fsn_interaction_heads=4,
        fsn_interaction_dropout=0.1,
        fsn_global_grid_size=3,
        fsn_module_lr_ratio=1.0,
        mc_sample_times=mc_sample_times,
        global_lr_ratio=0.5,
        stn_lr_ratio=0.2,
        temporal_lr_ratio=0.2,
    )


class AdaFocusFSN(FSNModel):
    """Official Uni-AdaFocus path or the opt-in FSN candidate path."""

    def __init__(
        self,
        num_classes: int = 7,
        modified: bool = False,
        device: torch.device | None = None,
        num_glance_segments: int = 4,
        num_input_focus_segments: int = 8,
        num_focus_segments: int = 4,
        patch_size: int = 96,
        mc_sample_times: int = 4,
    ) -> None:
        super().__init__()
        device = device or torch.device("cpu")
        self.model_name = "adafocus_fsn" if modified else "adafocus_original"
        self.num_glance_segments = num_glance_segments
        self.num_input_focus_segments = num_input_focus_segments
        self.num_focus_segments = num_focus_segments
        self.register_buffer("class_weights", None)
        self.core = AdaFocus(
            num_classes,
            _adafocus_args(
                device,
                modified,
                num_glance_segments,
                num_input_focus_segments,
                num_focus_segments,
                patch_size,
                mc_sample_times,
            ),
        )

    def set_class_weights(self, weights: torch.Tensor | None) -> None:
        self.class_weights = None if weights is None else weights.detach().clone()

    def _ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, weight=self.class_weights)

    @staticmethod
    def _take_uniform(frames: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        total = frames.shape[1]
        indices = torch.linspace(0, total - 1, count, device=frames.device).round().long()
        positions = indices.float() / max(total - 1, 1)
        positions = positions.unsqueeze(0).expand(frames.shape[0], -1)
        return frames.index_select(1, indices), positions

    def forward(self, frames: torch.Tensor) -> dict[str, Any]:
        if frames.ndim != 5 or frames.shape[2:] != (3, 224, 224):
            raise ValueError("AdaFocus expects frames [B,T,3,224,224]")
        mean = frames.new_tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
        std = frames.new_tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
        frames = (frames - mean) / std
        glance, glance_positions = self._take_uniform(frames, self.num_glance_segments)
        focus_input, input_positions = self._take_uniform(frames, self.num_input_focus_segments)
        batch = frames.shape[0]
        glance = glance.reshape(batch, self.num_glance_segments * 3, 224, 224)
        focus_input = focus_input.reshape(batch, self.num_input_focus_segments * 3, 224, 224)
        output = self.core(
            images_glance=glance,
            images_input=focus_input,
            glance_positions=glance_positions,
            input_positions=input_positions,
        )
        if self.training:
            random_branch, policy_branch = output
            return {
                "logits": policy_branch[0],
                "random_branch": random_branch,
                "policy_branch": policy_branch,
            }
        return {"logits": output[0], "eval_outputs": output}

    def compute_loss(self, output: dict[str, Any], target: torch.Tensor) -> torch.Tensor:
        if "policy_branch" not in output:
            return self._ce(output["logits"], target)
        p1 = output["random_branch"]
        p2 = output["policy_branch"]
        loss_target = self._ce(p1[0], target) + self._ce(p2[0], target)
        loss_global = self._ce(p1[1], target) + self._ce(p2[1], target)
        loss_local = self._ce(p1[2], target) + self._ce(p2[2], target)
        loss_temporal = self._ce(p1[3], target) + self._ce(p2[3], target)
        loss_spatial = self._ce(p1[4], target) + self._ce(p2[4], target)
        target_scale = torch.ones_like(p1[6][:, 2:4])
        loss_norm = ((p1[6][:, 2:4] - target_scale) ** 2).mean()
        return (
            loss_target + loss_global + loss_local + loss_temporal + loss_spatial
        ) / 2 + 0.5 * loss_norm


class MViTV2Reference(FSNModel):
    """Pure-PyTorch modern reference that is runnable on this CPU host.

    The formal 2024 CUDA reference remains VideoMamba-Ti; this MViT-V2-S model
    is used for local pipeline validation because VideoMamba's selective scan
    extension is CUDA-only.
    """

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        from torchvision.models.video import mvit_v2_s

        self.model_name = "mvit_v2_s_reference"
        self.core = mvit_v2_s(weights=None)
        self.core.head[1] = nn.Linear(self.core.head[1].in_features, num_classes)

    def forward(self, frames: torch.Tensor) -> dict[str, Any]:
        if frames.ndim != 5 or frames.shape[2:] != (3, 224, 224):
            raise ValueError("MViT-V2-S expects frames [B,T,3,224,224]")
        mean = frames.new_tensor((0.45, 0.45, 0.45)).view(1, 1, 3, 1, 1)
        std = frames.new_tensor((0.225, 0.225, 0.225)).view(1, 1, 3, 1, 1)
        frames = (frames - mean) / std
        indices = torch.linspace(0, frames.shape[1] - 1, 16, device=frames.device).round().long()
        frames = frames.index_select(1, indices)
        return {"logits": self.core(frames.permute(0, 2, 1, 3, 4).contiguous())}


def build_model(name: str, device: torch.device, num_classes: int = 7) -> FSNModel:
    if name == "adafocus_original":
        model = AdaFocusFSN(num_classes=num_classes, modified=False, device=device)
    elif name == "adafocus_fsn":
        model = AdaFocusFSN(num_classes=num_classes, modified=True, device=device)
    elif name == "mvit_v2_s_reference":
        model = MViTV2Reference(num_classes=num_classes)
    else:
        raise ValueError(f"unknown model: {name}")
    return model.to(device)


def load_shared_adafocus_weights(modified: AdaFocusFSN, baseline_state: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    """Load every shape-compatible baseline tensor into the FSN model."""
    current = modified.state_dict()
    compatible = {
        key: value for key, value in baseline_state.items()
        if key in current and current[key].shape == value.shape
    }
    result = modified.load_state_dict(compatible, strict=False)
    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "loaded_keys": sorted(compatible),
    }


def load_official_adafocus_checkpoint(
    model: AdaFocusFSN,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Load shape-compatible official weights while resetting seven-class heads."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = checkpoint.get("state_dict", checkpoint)
    if not isinstance(source, dict):
        raise ValueError("checkpoint does not contain a state dict")
    target = model.state_dict()
    classifier_tokens = ("new_fc", "new_new_fc", "aux_fc")
    compatible: dict[str, torch.Tensor] = {}
    skipped_heads: list[str] = []
    skipped_shape: list[str] = []
    for raw_key, value in source.items():
        key = raw_key.removeprefix("module.")
        key = key if key.startswith("core.") else "core." + key
        if any(token in key for token in classifier_tokens):
            skipped_heads.append(raw_key)
            continue
        if key not in target or target[key].shape != value.shape:
            skipped_shape.append(raw_key)
            continue
        compatible[key] = value
    result = model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "loaded_tensors": len(compatible),
        "skipped_head_tensors": len(skipped_heads),
        "skipped_shape_or_unknown_tensors": len(skipped_shape),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }
def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
