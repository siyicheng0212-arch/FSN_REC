"""Paired intervention audit for a trained early local-motion checkpoint.

The disabled model is the SAME checkpoint, not a separately trained Original.
Replaying torch RNG isolates the intervention from AdaFocus's Monte-Carlo path.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from experiments.full_data import FullClipDataset
from experiments.metrics import compute_classification_metrics
from experiments.model_wrappers import AdaFocusFSN


MOTION_VARIANTS = {
    "local_motion": ("matching", "none"),
    "local_appearance": ("appearance", "none"),
    "local_motion_context": ("matching", "global"),
}


def json_diagnostics(values: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu()
            result[key] = float(value) if value.numel() == 1 else value.tolist()
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    return result


def _rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    return (
        torch.random.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def _restore_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    torch.random.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def _autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@torch.no_grad()
def paired_forward(
    model: AdaFocusFSN, video: torch.Tensor, device: torch.device,
    intervention: str = "module",
) -> tuple[dict, dict, dict]:
    module = model.local_motion_module
    if module is None:
        raise ValueError("paired audit requires a local-motion module")
    if intervention not in {"module", "context"}:
        raise ValueError("intervention must be module or context")
    if intervention == "context" and module.context_mode != "global":
        raise ValueError("context intervention requires a global-conditioned module")
    before = _rng_state()
    after_on = before
    enabled_before = module.enabled
    context_before = getattr(module, "context_enabled", True)
    try:
        module.enabled = True
        module.context_enabled = True
        with _autocast(device):
            on = model(video)
        diagnostics = json_diagnostics(model.get_local_motion_diagnostics())
        after_on = _rng_state()
        _restore_rng(before)
        if intervention == "module":
            module.enabled = False
        else:
            module.context_enabled = False
        with _autocast(device):
            off = model(video)
        return on, off, diagnostics
    finally:
        module.enabled = enabled_before
        module.context_enabled = context_before
        # Preserve the state advance of one enabled forward for the caller.
        _restore_rng(after_on)


def prediction_effect(on: torch.Tensor, off: torch.Tensor, targets: torch.Tensor) -> dict:
    on = on.detach().float().cpu()
    off = off.detach().float().cpu()
    targets = targets.detach().cpu()
    if on.shape != off.shape or on.ndim != 2 or targets.shape != (on.shape[0],):
        raise ValueError("on/off logits and targets must have matching sample dimensions")
    if not torch.isfinite(on).all() or not torch.isfinite(off).all():
        raise ValueError("non-finite intervention logits")
    pred_on, pred_off = on.argmax(1), off.argmax(1)
    changed = pred_on != pred_off
    corrected = (pred_on == targets) & (pred_off != targets)
    harmed = (pred_on != targets) & (pred_off == targets)
    delta = (on - off).abs()
    return {
        "samples": len(targets),
        "changed": int(changed.sum()),
        "corrected": int(corrected.sum()),
        "harmed": int(harmed.sum()),
        "changed_both_wrong": int((changed & ~corrected & ~harmed).sum()),
        "mean_abs_logit_delta": float(delta.mean()),
        "max_abs_logit_delta": float(delta.max()),
    }


@torch.no_grad()
def audit_loader(
    model: AdaFocusFSN, loader: DataLoader, device: torch.device,
    intervention: str = "module",
) -> tuple[dict, list[dict]]:
    model.eval()
    if len(loader) == 0:
        raise ValueError("audit loader is empty")
    on_all, off_all, targets_all, metadata_all, rows = [], [], [], [], []
    diagnostic_sums: dict[str, float] = {}
    diagnostic_weights: dict[str, int] = {}
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        target = batch["label"].cpu()
        on_output, off_output, diagnostics = paired_forward(model, video, device, intervention)
        on, off = on_output["logits"].float().cpu(), off_output["logits"].float().cpu()
        on_all.append(on); off_all.append(off); targets_all.append(target)
        for key, value in diagnostics.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                diagnostic_sums[key] = diagnostic_sums.get(key, 0.) + float(value) * len(target)
                diagnostic_weights[key] = diagnostic_weights.get(key, 0) + len(target)
        eval_outputs = on_output.get("eval_outputs")
        indices = None
        if eval_outputs is not None:
            indices = eval_outputs[8].reshape(len(target), -1).detach().cpu()
            indices = indices - torch.arange(len(target))[:, None] * model.num_input_focus_segments
        for i in range(len(target)):
            metadata = {
                "clip_id": batch["clip_id"][i], "source": batch["source"][i],
                "duration": float(batch["duration"][i]),
            }
            metadata_all.append(metadata)
            row = {
                **metadata, "target": int(target[i]),
                "prediction_on": int(on[i].argmax()), "prediction_off": int(off[i].argmax()),
                "logits_on": on[i].tolist(), "logits_off": off[i].tolist(),
            }
            if indices is not None:
                ix = indices[i]
                row["focus_cache_indices"] = ix.tolist()
                # full_data uses bin-centre requests. These are requested sample
                # times, not verified decoded PTS or physical object velocities.
                times = (ix.float() + .5) * metadata["duration"] / model.num_input_focus_segments
                row["requested_focus_seconds_within_clip"] = times.tolist()
                row["requested_focus_gap_seconds"] = times.diff().tolist()
            rows.append(row)
    on = torch.cat(on_all); off = torch.cat(off_all); targets = torch.cat(targets_all)
    metrics_on = compute_classification_metrics(on, targets, metadata_all)
    metrics_off = compute_classification_metrics(off, targets, metadata_all)
    result = {
        "intervention": intervention,
        "comparison": "same_checkpoint_enabled_vs_intervention_disabled",
        "rng_replayed_per_batch": True,
        **prediction_effect(on, off, targets),
        "metrics_on": metrics_on, "metrics_off": metrics_off,
        "macro_f1_delta": metrics_on["all"]["macro_f1"] - metrics_off["all"]["macro_f1"],
        "module_diagnostics_mean": {
            key: value / diagnostic_weights[key] for key, value in diagnostic_sums.items()
        },
        "timestamp_note": "uniform-cache requested bin centres; repeated pixels and cuts can invalidate motion",
    }
    return result, rows


def model_from_saved_args(saved: dict[str, Any], device: torch.device) -> AdaFocusFSN:
    variant = saved["variant"]
    if variant not in MOTION_VARIANTS:
        raise ValueError(f"checkpoint variant {variant!r} is not a local-motion experiment")
    mode, context = MOTION_VARIANTS[variant]
    return AdaFocusFSN(
        num_classes=7, modified=False, device=device,
        num_glance_segments=8, num_input_focus_segments=36,
        num_focus_segments=12, patch_size=128, mc_sample_times=128,
        local_motion_mode=mode, local_motion_context=context,
        local_motion_dim=saved.get("local_motion_dim", 64),
        local_motion_window=saved.get("local_motion_window", 3),
        local_motion_temperature=saved.get("local_motion_temperature", .07),
        local_motion_context_grid=saved.get("local_motion_context_grid", 2),
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="trained best.pt, not SSv2 pretraining")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--intervention", choices=("module", "context"), default="module")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu only for diagnostic smoke tests")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_saved_args(checkpoint["args"], device)
    # A None registered buffer is absent from a fresh wrapper state_dict;
    # trained checkpoints contain the chosen class weights. Materialize it
    # before strict loading rather than ignoring a genuine state mismatch.
    class_weights = checkpoint["model"].get("class_weights")
    model.set_class_weights(None if class_weights is None else class_weights.to(device))
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = FullClipDataset(args.manifest_dir / f"{args.split}.jsonl", args.cache_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False, pin_memory=device.type == "cuda")
    torch.manual_seed(int(checkpoint["args"].get("seed", 42)))
    if device.type == "cuda": torch.cuda.manual_seed_all(int(checkpoint["args"].get("seed", 42)))
    result, rows = audit_loader(model, loader, device, args.intervention)
    result.update(checkpoint=str(args.checkpoint.resolve()), split=args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.split}_{args.intervention}_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / f"{args.split}_{args.intervention}_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({key: result[key] for key in (
        "intervention", "samples", "changed", "corrected", "harmed", "macro_f1_delta"
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
