"""Read Original features without changing its forward path or classification.

Positions refer to the existing frame-cache index, never verified source PTS.
Temporary forward hooks are removed even when extraction fails.
"""
from pathlib import Path, PosixPath, WindowsPath

import torch
from torch import nn

POSITION_BASIS = "cache_index_fraction_not_verified_pts"
EVIDENCE_KEYS = ("global_tokens", "local_tokens", "global_positions", "local_positions")


def validate_original_checkpoint_protocol(checkpoint):
    """Require saved evidence of the exact full-data training protocol.

    Canonical historical ``train_adafocus`` checkpoints have an integer epoch
    for finetuning and a string such as ``head_warmup_3`` for head warmup, but
    omit ``phase``. The newer aligned trainer additionally saves ``phase``.
    A saved finetuning epoch establishes that epoch's completion, not that an
    entire launcher or experiment suite has completed.
    """
    from experiments.aligned_protocol import validate_manifest_protocol

    if not isinstance(checkpoint, dict):
        raise ValueError("Original checkpoint must contain a saved training protocol")
    audit = checkpoint.get("split_audit")
    if not isinstance(audit, dict):
        raise ValueError(
            "formal source-policy run requires Original best.pt with saved "
            "split_audit counts and manifest_sha256 for train7372/val823; "
            "locate the matching full-protocol Original checkpoint, do not "
            "invent its training provenance"
        )
    counts, hashes = audit.get("counts"), audit.get("manifest_sha256")
    if (not isinstance(counts, dict) or not isinstance(hashes, dict)
            or any(type(value) is not int for value in counts.values())):
        raise ValueError("Original split_audit requires integer counts and manifest_sha256 mappings")
    try:
        validate_manifest_protocol(counts, hashes)
    except RuntimeError as error:
        raise ValueError(
            "Original checkpoint was not trained under the fixed full "
            "train7372/val823 protocol; locate the matching Original best.pt: "
            + str(error)
        ) from error
    if "phase" in checkpoint and checkpoint["phase"] != "finetune":
        raise ValueError("formal source-policy A must be a finetune checkpoint, not head/module warmup")
    epoch = checkpoint.get("epoch")
    if type(epoch) is not int or epoch < 1:
        raise ValueError(
            "formal source-policy A requires a saved positive integer finetune "
            "epoch; warmup-only or provenance-incomplete checkpoints are unsupported"
        )
    return {"counts": dict(counts), "manifest_sha256": dict(hashes),
            "epoch": epoch, "phase": checkpoint.get("phase", "historical_integer_finetune_epoch")}


class FrozenOriginalEvidence(nn.Module):
    def __init__(self, visual: nn.Module):
        super().__init__()
        if getattr(visual, "model_name", None) != "adafocus_original":
            raise ValueError("A must be the unmodified AdaFocusFSN Original")
        core = visual.core
        for name in ("fsn_local_adapter", "fsn_interaction_mode", "local_motion_mode", "context_mode"):
            if getattr(core, name, "none") != "none":
                raise ValueError(f"A contains an experimental module: {name}")
        local = core.local_CNN
        if any(getattr(local, name, None) is not None
               for name in ("local_adapter", "local_motion", "aligned_context")):
            raise ValueError("A contains an experimental local module")
        if getattr(local, "return_feature_grid", False) or getattr(core, "fsn_interaction", None) is not None:
            raise ValueError("A must use the Original forward path")
        self.visual = visual
        for parameter in visual.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        self.visual.eval()  # BN/dropout and sampling remain in inference mode.
        return self

    @torch.no_grad()
    def forward(self, frames):
        if (frames.ndim != 5 or frames.shape[0] < 1 or frames.shape[1] < 1
                or frames.shape[2:] != (3, 224, 224) or not frames.is_floating_point()):
            raise ValueError("frames must be float [B,T,3,224,224] in [0,1]")
        if not bool(torch.isfinite(frames).all()) or bool(((frames < 0) | (frames > 1)).any()):
            raise ValueError("frames must be finite and in [0,1]")
        self.visual.eval()
        global_values, local_values = [], []

        def global_hook(module, inputs, output):
            if not isinstance(output, (tuple, list)) or len(output) < 4:
                raise RuntimeError("unexpected global_CNN feature contract")
            global_values.append(output[3].detach().clone())

        def local_hook(module, inputs, output):
            local_values.append(output.detach().flatten(1).clone())

        handles = [self.visual.core.global_CNN.register_forward_hook(global_hook),
                   self.visual.core.local_CNN.base_model.avgpool.register_forward_hook(local_hook)]
        try:
            result = self.visual(frames)
        finally:
            for handle in handles:
                handle.remove()
        if len(global_values) != 1 or len(local_values) != 1:
            raise RuntimeError("expected one global and one local evidence capture")
        b = frames.shape[0]
        tg, ti, tl = (self.visual.num_glance_segments, self.visual.num_input_focus_segments,
                      self.visual.num_focus_segments)
        if global_values[0].shape != (b * tg, 1280) or local_values[0].shape != (b * tl, 2048):
            raise RuntimeError("unexpected Original feature dimensions")
        indices = result["eval_outputs"][8].reshape(b, tl)
        offsets = torch.arange(b, device=indices.device).unsqueeze(1) * ti
        relative = indices - offsets
        if relative.dtype != torch.long or bool(((relative < 0) | (relative >= ti)).any()):
            raise RuntimeError("invalid Original focus indices")
        _, global_positions = self.visual._take_uniform(frames, tg)
        _, input_positions = self.visual._take_uniform(frames, ti)
        local_positions = input_positions.gather(1, relative)
        if bool((local_positions[:, 1:] < local_positions[:, :-1]).any()):
            raise RuntimeError("Original selected frames are unordered")
        evidence = {"logits": result["logits"].detach(),
                    "global_tokens": global_values[0].reshape(b, tg, 1280),
                    "local_tokens": local_values[0].reshape(b, tl, 2048),
                    "global_positions": global_positions.detach(),
                    "local_positions": local_positions.detach()}
        if evidence["logits"].shape != (b, 7) or not all(
                bool(torch.isfinite(value).all()) for value in evidence.values()):
            raise RuntimeError("invalid Original evidence")
        return evidence


def load_original(checkpoint_path, device="cpu", *, require_full_protocol=False):
    """Strictly load a trusted, finetuned seven-class Original best.pt.

    This is not the official SSv2 initializer: no head or shared weight is reset.
    Formal source-policy calls additionally require exact saved full-data
    manifest provenance; the default retains legacy prototype compatibility.
    """
    from experiments.model_wrappers import AdaFocusFSN

    device = torch.device(device)
    # Canonical trainers save argparse Path values alongside tensors. Allow only
    # those known types instead of falling back to unrestricted pickle loading.
    with torch.serialization.safe_globals([PosixPath, WindowsPath]):
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("expected a finetuned checkpoint containing model state")
    if require_full_protocol:
        validate_original_checkpoint_protocol(checkpoint)
    state = checkpoint["model"]
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict) or saved_args.get("variant", "original") != "original":
        raise ValueError("checkpoint is not from the canonical Original trainer")
    for key, expected in (("num_glance_segments", 8), ("num_input_focus_segments", 36),
                          ("num_focus_segments", 12), ("patch_size", 128), ("mc_sample_times", 128)):
        if key in saved_args and saved_args[key] != expected:
            raise ValueError(f"checkpoint sampling mismatch: {key}")
    model = AdaFocusFSN(num_classes=7, modified=False, device=device,
                       num_glance_segments=8, num_input_focus_segments=36,
                       num_focus_segments=12, patch_size=128, mc_sample_times=128)
    if "class_weights" in state:
        weights = state["class_weights"]
        if weights.shape != (7,) or not bool(torch.isfinite(weights).all()):
            raise ValueError("invalid checkpoint class_weights")
        model.set_class_weights(weights)
    model.load_state_dict(state, strict=True)
    return FrozenOriginalEvidence(model.to(device))
