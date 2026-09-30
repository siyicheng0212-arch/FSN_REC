"""Shared visual backbones for a controlled clinical hierarchy study.

Soft factorization and hard routing are established diagnostic alternatives,
not new architecture claims. Metadata and true labels never enter forward().
Weights are loaded strictly; a missing pretrained file never becomes a silent
random-initialization run. Downloads require explicit allow_download=True.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlparse

import torch
from torch import nn
import torch.nn.functional as F

from .taxonomy import NUM_CLASSES, Taxonomy, clinical_taxonomy, _validate_targets


BACKBONES = ("r2plus1d_18", "mvit_v2_s", "r3d_18", "videomamba_tiny16")
MODES = ("flat", "capacity_control", "aux_flat", "hierarchy")


def preprocessing_spec(backbone: str) -> dict[str, Any]:
    if backbone == "videomamba_tiny":
        backbone = "videomamba_tiny16"
    if backbone not in BACKBONES:
        raise ValueError(f"unsupported backbone: {backbone}")
    if backbone == "videomamba_tiny16":
        return {"frames": 16, "crop_size": [224, 224], "resize_size": [256],
                "mean": [.485, .456, .406], "std": [.229, .224, .225]}
    if backbone == "mvit_v2_s":
        return {"frames": 16, "crop_size": [224, 224], "resize_size": [256],
                "mean": [.45, .45, .45], "std": [.225, .225, .225]}
    return {"frames": 16, "crop_size": [112, 112], "resize_size": [128, 171],
            "mean": [.43216, .394666, .37645], "std": [.22803, .22145, .216989]}


def _torchvision_components(backbone: str):
    from torchvision.models import video

    names = {"r2plus1d_18": "R2Plus1D_18_Weights", "r3d_18": "R3D_18_Weights",
             "mvit_v2_s": "MViT_V2_S_Weights"}
    if backbone not in names:
        raise ValueError(f"unsupported backbone: {backbone}")
    return getattr(video, backbone), getattr(video, names[backbone])


def preprocessing_transform(backbone: str):
    """Native official spatial/normalization transform; does not download weights.

    Accepts [T,C,H,W] or [B,T,C,H,W] uint8 or float [0,1], returns [C,T,H,W]
    or [B,C,T,H,W]. Temporal subsampling is the trainer's declared protocol.
    """
    if backbone in ("videomamba_tiny", "videomamba_tiny16"):
        from torchvision.transforms._presets import VideoClassification
        spec = preprocessing_spec(backbone)
        return VideoClassification(crop_size=spec["crop_size"], resize_size=spec["resize_size"],
                                   mean=spec["mean"], std=spec["std"])
    _, enum = _torchvision_components(backbone)
    return enum.DEFAULT.transforms()


class CapacityAdapter(nn.Module):
    """Active generic nonlinear capacity control, not a taxonomy mechanism.

    2D+2 adapter parameters + 7(D+1) classifier parameters exactly equal the
    clinical group/fine heads' 9(D+1). No dummy weights or empty classes.
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(feature_dim))
        self.shift = nn.Parameter(torch.zeros(feature_dim))
        self.alpha = nn.Parameter(torch.tensor(.01))
        self.beta = nn.Parameter(torch.tensor(0.))
        self.classifier = nn.Linear(feature_dim, NUM_CLASSES)

    def forward(self, features):
        adapted = features + self.alpha * (features * self.scale + self.shift).tanh() + self.beta * features
        return self.classifier(adapted)


class ClinicalHierarchyModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, *, mode: str = "flat",
                 taxonomy: Taxonomy | None = None, load_report: Mapping[str, Any] | None = None):
        super().__init__()
        if mode not in MODES or feature_dim < 1:
            raise ValueError("invalid mode or feature dimension")
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.mode = mode
        self.taxonomy = taxonomy or clinical_taxonomy()
        self.load_report = dict(load_report or {"pretrained": False, "source": "injected_backbone"})
        self.flat_head = nn.Linear(feature_dim, NUM_CLASSES)
        # Additional constructors do not advance shared/backbone dropout RNG.
        with torch.random.fork_rng(devices=[]):
            self.group_head = nn.Linear(feature_dim, self.taxonomy.num_groups)
            self.conditional_heads = nn.ModuleList([
                nn.Linear(feature_dim, len(group)) if len(group) > 1 else nn.Identity()
                for group in self.taxonomy.groups
            ])
            self.capacity_head = CapacityAdapter(feature_dim) if mode == "capacity_control" else None
        if mode == "capacity_control" and self.taxonomy.auxiliary_rows != 9:
            raise ValueError("capacity control requires the declared (1,1,3,2) size profile")
        self.trained_heads = {
            "flat": ("flat",), "capacity_control": ("flat", "capacity"),
            "aux_flat": ("flat", "group", "conditional"),
            "hierarchy": ("group", "conditional"),
        }[mode]
        self.configure_trainable()

    def configure_trainable(self, backbone_trainable: bool = True) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = backbone_trainable
        for name, module in (("flat", self.flat_head), ("group", self.group_head),
                             ("conditional", self.conditional_heads), ("capacity", self.capacity_head)):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = name in self.trained_heads

    def forward(self, video: torch.Tensor) -> dict[str, Any]:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError("model input must be normalized [B,3,T,H,W]")
        features = self.backbone(video)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise RuntimeError("backbone did not return the declared [B,D] features")
        conditional = [head(features) if len(group) > 1 else features.new_zeros(features.shape[0], 1)
                       for group, head in zip(self.taxonomy.groups, self.conditional_heads)]
        return {"flat_logits": self.flat_head(features), "group_logits": self.group_head(features),
                "conditional_logits": conditional, "features": features,
                "capacity_logits": None if self.capacity_head is None else self.capacity_head(features),
                "trained_heads": self.trained_heads, "mode": self.mode}

    def parameter_report(self) -> dict[str, Any]:
        return parameter_report(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_native_weights(model: nn.Module, backbone: str, weights, weights_path, allow_download: bool):
    _, enum = _torchvision_components(backbone)
    if weights is None and weights_path is None:
        return {"pretrained": False, "source": "explicit_random_initialization"}
    selected = enum.DEFAULT if weights in (None, "DEFAULT") else enum.verify(weights)
    path = Path(weights_path).expanduser().resolve() if weights_path is not None else (
        Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(selected.url).path).name
    )
    if not path.is_file():
        if weights_path is not None or not allow_download:
            raise FileNotFoundError(f"Pretrained weights unavailable: {path}. Supply an official local file or explicitly allow download.")
        selected.get_state_dict(progress=True, check_hash=True)
    sha256 = _sha256(path)
    expected_prefix = re.search(r"-([0-9a-f]{8})\.pth$", selected.url)
    if expected_prefix is None or not sha256.startswith(expected_prefix.group(1)):
        raise ValueError("checkpoint does not match the declared official torchvision weight hash")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping) or not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("pretrained checkpoint must be a native torchvision tensor state_dict")
    if all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return {"pretrained": True, "source": str(path), "sha256": sha256,
            "weights": selected.name, "backbone": backbone, "strict": True,
            "official_hash_verified": True,
            "discarded_pretrained_classifier": "head.1" if backbone == "mvit_v2_s" else "fc"}


def build_model(backbone: str = "r2plus1d_18", mode: str = "flat", taxonomy: Taxonomy | None = None,
                *, weights="DEFAULT", weights_path=None, allow_download: bool = False,
                seed: int | None = None, external_repo=None, videomamba_root=None,
                videomamba_checkpoint=None) -> ClinicalHierarchyModel:
    if backbone in ("videomamba_tiny", "videomamba_tiny16"):
        from .videomamba import build_videomamba_tiny16
        repo = external_repo if external_repo is not None else videomamba_root
        checkpoint = weights_path if weights_path is not None else videomamba_checkpoint
        if repo is None or checkpoint is None:
            raise ValueError("VideoMamba requires explicit official repository and pretrained checkpoint paths")
        context = torch.random.fork_rng(devices=[]) if seed is not None else nullcontext()
        with context:
            if seed is not None:
                torch.random.default_generator.manual_seed(seed)
            core = build_videomamba_tiny16(repo_path=repo, checkpoint_path=checkpoint, seed=seed)
            return ClinicalHierarchyModel(core, core.feature_dim, mode=mode,
                                          taxonomy=taxonomy, load_report=core.load_report)
    constructor, _ = _torchvision_components(backbone)
    context = torch.random.fork_rng(devices=[]) if seed is not None else nullcontext()
    with context:
        if seed is not None:
            torch.random.default_generator.manual_seed(seed)
        core = constructor(weights=None)
        report = _load_native_weights(core, backbone, weights, weights_path, allow_download)
        if backbone == "mvit_v2_s":
            feature_dim = core.head[-1].in_features
            core.head[-1] = nn.Identity()  # preserve native dropout before shared features
        else:
            feature_dim = core.fc.in_features
            core.fc = nn.Identity()
        model = ClinicalHierarchyModel(core, feature_dim, mode=mode, taxonomy=taxonomy, load_report=report)
    return model


def parameter_report(model: ClinicalHierarchyModel) -> dict[str, Any]:
    groups = {"backbone": model.backbone, "flat": model.flat_head,
              "group": model.group_head, "conditional": model.conditional_heads,
              "capacity": model.capacity_head}
    by_component = {
        name: {"total": sum(parameter.numel() for parameter in module.parameters()),
               "trainable": sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)}
        for name, module in groups.items() if module is not None
    }
    return {"total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "feature_dim": model.feature_dim, "trained_heads": list(model.trained_heads),
            "components": by_component}


def _weighted_mean(loss: torch.Tensor, targets: torch.Tensor, class_weights: torch.Tensor | None):
    if class_weights is None:
        return loss.mean()
    if class_weights.shape != (NUM_CLASSES,) or bool((class_weights < 0).any()) or not bool(torch.isfinite(class_weights).all()):
        raise ValueError("class weights must be seven finite nonnegative values")
    weights = class_weights.to(device=loss.device, dtype=loss.dtype)[targets]
    denominator = weights.sum()
    if not bool(denominator > 0):
        raise ValueError("batch must have positive total class weight")
    return (loss * weights).sum() / denominator


def loss_components(outputs: Mapping[str, Any], targets: torch.Tensor, taxonomy: Taxonomy,
                    class_weights: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    _validate_targets(targets)
    if not targets.numel():
        raise ValueError("empty training batch")
    flat_loss = F.cross_entropy(outputs["flat_logits"], targets, reduction="none")
    group_targets = taxonomy.group_targets(targets)
    fine_targets = taxonomy.conditional_targets(targets)
    group_loss = F.cross_entropy(outputs["group_logits"], group_targets, reduction="none")
    # One fine loss per example, zero for singleton groups. All examples retain
    # the SAME denominator; averaging each group's losses separately is biased.
    fine_loss = flat_loss.new_zeros(targets.shape[0])
    for group_index, group in enumerate(taxonomy.groups):
        selected = group_targets == group_index
        if len(group) > 1 and bool(selected.any()):
            fine_loss[selected] = F.cross_entropy(outputs["conditional_logits"][group_index][selected],
                                                  fine_targets[selected], reduction="none")
    result = {"flat": _weighted_mean(flat_loss, targets, class_weights),
              "group": _weighted_mean(group_loss, targets, class_weights),
              "conditional": _weighted_mean(fine_loss, targets, class_weights)}
    result["hierarchy"] = result["group"] + result["conditional"]
    if outputs.get("capacity_logits") is not None:
        result["capacity"] = _weighted_mean(F.cross_entropy(outputs["capacity_logits"], targets, reduction="none"), targets, class_weights)
    return result


def compute_loss(outputs: Mapping[str, Any], targets: torch.Tensor, mode: str, taxonomy: Taxonomy,
                 aux_weight: float = 1., group_weight: float = 1., conditional_weight: float = 1.,
                 class_weights: torch.Tensor | None = None) -> torch.Tensor:
    if mode not in MODES or any(not math.isfinite(weight) or weight < 0 for weight in (aux_weight, group_weight, conditional_weight)):
        raise ValueError("invalid loss mode or weights")
    losses = loss_components(outputs, targets, taxonomy, class_weights)
    hierarchy = group_weight * losses["group"] + conditional_weight * losses["conditional"]
    if mode == "flat":
        return losses["flat"]
    if mode == "hierarchy":
        return hierarchy
    if mode == "capacity_control":
        return losses["flat"] + aux_weight * losses["capacity"]
    return losses["flat"] + aux_weight * hierarchy


def _require_heads(outputs: Mapping[str, Any], heads: tuple[str, ...]) -> None:
    trained = outputs.get("trained_heads")
    if trained is not None and not all(head in trained for head in heads):
        raise ValueError(f"requested predictions use untrained heads: {heads}")


def soft_leaf_probabilities(outputs: Mapping[str, Any], taxonomy: Taxonomy) -> torch.Tensor:
    _require_heads(outputs, ("group", "conditional"))
    group_probabilities = outputs["group_logits"].float().softmax(dim=1)
    probabilities = group_probabilities.new_zeros(group_probabilities.shape[0], NUM_CLASSES)
    for group_index, group in enumerate(taxonomy.groups):
        fine = outputs["conditional_logits"][group_index].float().softmax(dim=1)
        probabilities[:, list(group)] = group_probabilities[:, group_index, None] * fine
    return probabilities


def hard_leaf_probabilities(outputs: Mapping[str, Any], taxonomy: Taxonomy) -> torch.Tensor:
    _require_heads(outputs, ("group", "conditional"))
    routes = outputs["group_logits"].argmax(dim=1)
    probabilities = outputs["group_logits"].float().new_zeros(routes.shape[0], NUM_CLASSES)
    for group_index, group in enumerate(taxonomy.groups):
        selected = routes == group_index
        probabilities[:, list(group)] = outputs["conditional_logits"][group_index].float().softmax(dim=1) * selected[:, None]
    return probabilities


def decode_flat(outputs: Mapping[str, Any]) -> torch.Tensor:
    _require_heads(outputs, ("flat",))
    return outputs["flat_logits"].argmax(dim=1)


def decode_soft(outputs: Mapping[str, Any], taxonomy: Taxonomy) -> torch.Tensor:
    return soft_leaf_probabilities(outputs, taxonomy).argmax(dim=1)


def decode_hard(outputs: Mapping[str, Any], taxonomy: Taxonomy) -> torch.Tensor:
    return hard_leaf_probabilities(outputs, taxonomy).argmax(dim=1)


def decode_oracle(outputs: Mapping[str, Any], targets: torch.Tensor, taxonomy: Taxonomy) -> torch.Tensor:
    """Diagnostic ONLY: true-group route, never deployed or selected as main score."""
    _require_heads(outputs, ("group", "conditional"))
    routes = taxonomy.group_targets(targets)
    masked_logits = outputs["group_logits"].float().new_full((targets.shape[0], NUM_CLASSES), float("-inf"))
    for group_index, group in enumerate(taxonomy.groups):
        selected = routes == group_index
        fine_logits = outputs["conditional_logits"][group_index].float()
        masked_logits[:, list(group)] = torch.where(selected[:, None], fine_logits, masked_logits[:, list(group)])
    return masked_logits.argmax(dim=1)  # canonical-ID tie break, also used by hard/soft/flat


def flat_oracle(outputs: Mapping[str, Any], targets: torch.Tensor, taxonomy: Taxonomy) -> torch.Tensor:
    """Fair flat diagnostic with the same true-group information as oracle routing."""
    _require_heads(outputs, ("flat",))
    routes = taxonomy.group_targets(targets)
    mapping = torch.tensor(taxonomy.label_to_group, device=targets.device)
    logits = outputs["flat_logits"].masked_fill(mapping[None] != routes[:, None], float("-inf"))
    return logits.argmax(dim=1)


def prediction_probabilities(outputs: Mapping[str, Any], mode: str, taxonomy: Taxonomy,
                             decoding: str = "soft") -> torch.Tensor:
    if mode in ("flat", "capacity_control", "aux_flat"):
        _require_heads(outputs, ("flat",))
        return outputs["flat_logits"].float().softmax(dim=1)
    if mode == "hierarchy" and decoding in ("soft", "hard"):
        return soft_leaf_probabilities(outputs, taxonomy) if decoding == "soft" else hard_leaf_probabilities(outputs, taxonomy)
    raise ValueError("invalid prediction mode or hierarchy decoding")
