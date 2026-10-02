"""Matched continuation experiments on a frozen-A TCN; never train A or R."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.fsn_tcn.io import fresh_output, load_chain, write_json, write_jsonl
from experiments.relation.data import sha256_file
from .augmentation import apply_view, build_epoch_plan
from .config import VARIANTS, baseline_source_hashes, build_base, code_hashes, load_config
from .data import load_bundle, validate_source_kinds
from .model import ContextUtilityTCN


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(bundle, config, checkpoint, variant, seed, device="cpu"):
    if variant not in VARIANTS:
        raise ValueError("unknown context-utility variant")
    seed_everything(seed)
    width = bundle.index.global_dim + bundle.index.local_dim
    base, audit = build_base(config, width, checkpoint, device, fingerprint=bundle.fingerprint)
    mode = "dynamic" if variant == "dynamic_aug" else "scalar" if variant == "scalar_aug" else "plain"
    model = ContextUtilityTCN(base, width, config["baseline"]["temporal_paths"], mode=mode,
                              dim=config["module"]["dim"], gate_seed=seed).to(device)
    # Factory construction/extra modules must not change the continuation RNG.
    seed_everything(seed)
    return model, audit


def sealed_inputs(bundle, config, checkpoint):
    return {"fingerprint": bundle.fingerprint, "config": config,
            "base_checkpoint_sha256": sha256_file(checkpoint), "source_sha256": code_hashes(),
            "external_baseline_sha256": baseline_source_hashes(config),
            "legacy_audit_sha256": sha256_file(config["baseline"]["legacy_audit"])
                if config["baseline"]["kind"] == "legacy" else None}


def load_trained(path, bundle, config, checkpoint, variant, seed, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    for name, value in sealed_inputs(bundle, config, checkpoint).items():
        if state.get(name) != value:
            raise ValueError(f"trained checkpoint seal mismatch: {name}")
    if state.get("variant") != variant or state.get("seed") != seed:
        raise ValueError("trained checkpoint variant/seed mismatch")
    model, audit = make_model(bundle, config, checkpoint, variant, seed, device)
    model.load_state_dict(state["model"], strict=True)
    return model.eval(), state, audit


def _optimizer(model, settings):
    kwargs = {"lr": settings["lr"], "weight_decay": settings["weight_decay"]}
    if settings["optimizer"] == "AdamW":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if settings["optimizer"] == "SGD":
        return torch.optim.SGD(model.parameters(), momentum=settings["momentum"], **kwargs)
    raise ValueError("unsupported optimizer")


def _read_clip(bundle, clip, device):
    values, logits, _ = load_chain(bundle, {"ordered_clip_ids": [clip]}, device)
    return values[0], logits[0]


def gradient_audit(model):
    groups = {"base": {"tensors": 0, "nonzero": 0, "norm_sq": 0.},
              "gate": {"tensors": 0, "nonzero": 0, "norm_sq": 0.}}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError("nonfinite gradient; preserve output and stop")
        group = groups["gate" if not name.startswith("base.") else "base"]
        group["tensors"] += 1
        norm = float(parameter.grad.detach().float().norm())
        group["nonzero"] += norm > 0
        group["norm_sq"] += norm * norm
    for group in groups.values():
        group["norm"] = group.pop("norm_sq") ** .5
    return groups


def train_variant(bundle, config, checkpoint, seed, variant, output, device="cpu"):
    """CPU is permitted only as a helper for synthetic integration tests."""
    if seed not in config["seeds"]:
        raise ValueError("seed was not predeclared")
    validate_source_kinds(bundle)
    settings, aug = config["optimization"], config["augmentation"]
    plans = [build_epoch_plan(bundle, seed=seed, epoch=epoch,
              anchors_per_segment=aug["anchors_per_segment"], max_neighbors=aug["max_neighbors"])
             for epoch in range(1, settings["epochs"] + 1)]
    if any(not audit["has_constructed_replacements"] for _, audit in plans):
        raise ValueError("no train-only neighbor replacement can be constructed; do not silently change protocol")
    model, base_audit = make_model(bundle, config, checkpoint, variant, seed, device)
    # A genuinely unmodified initial scorer, not gate-off on trained weights.
    initial_base, _ = build_base(config, bundle.index.global_dim + bundle.index.local_dim,
                                 checkpoint, device, fingerprint=bundle.fingerprint)
    initial_base.eval()
    initial_base.requires_grad_(False)
    seed_everything(seed)
    optimizer = _optimizer(model, settings)
    out = fresh_output(output)
    seals = sealed_inputs(bundle, config, checkpoint)
    protocol = {"schema": "fsn-context-utility-training-v1", **seals,
                "seed": seed, "variant": variant, "base_audit": base_audit,
                "A_trainable_parameters": 0, "R_training": False,
                "loss": "mean original-segment CE + fixed weight * mean current-anchor-view CE",
                "donor_positions_supervised": False, "anchor_loss_weight": aug["anchor_loss_weight"],
                "clean_second_view": variant == "continue", "fresh_optimizer": True,
                "selection": "original val823 Macro-F1; ties lower CE",
                "evaluation_role": "val823_development_not_independent_test",
                "strict_boundary_isolation_claimed": False,
                "plan_sha256_per_epoch": [audit["plan_sha256"] for _, audit in plans],
                "gate_coefficients_are_probabilities": False}
    write_json(out / "protocol.json", protocol, exclusive=True)
    plan_dir = out / "private_plans"
    plan_dir.mkdir()
    for epoch, (plan, audit) in enumerate(plans, 1):
        write_json(plan_dir / f"epoch_{epoch:03d}.json", {"plan": plan, "audit": audit}, exclusive=True)
    from .evaluate import evaluate_chains
    history, best_key, stale = [], None, 0
    start = time.monotonic()
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch, (plan, plan_audit) in enumerate(plans, 1):
        model.train()
        total_loss, original_clips, anchor_count, steps, frozen_batches = 0., 0, 0, 0, 0
        gradient_maxima = {"base": 0., "gate": 0.}
        last_gradient = None
        for offset in range(0, len(plan), settings["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            originals, anchors = [], []
            batch_clips, batch_anchors = 0, 0
            for entry in plan[offset:offset + settings["batch_size"]]:
                features, a_logits, labels = load_chain(bundle, entry, device)
                values = model(features, a_logits)
                originals.append(F.cross_entropy(values.float(), labels, reduction="sum"))
                if variant == "continue":
                    view_features, view_logits = features, a_logits
                    positions = entry["anchor_positions"]
                else:
                    view_features, view_logits, positions = apply_view(features, a_logits, entry,
                        lambda clip: _read_clip(bundle, clip, device))
                view_output = model(view_features, view_logits)
                positions = torch.tensor(positions, dtype=torch.long, device=device)
                anchors.append(F.cross_entropy(view_output[positions].float(), labels[positions], reduction="sum"))
                batch_clips += len(labels)
                batch_anchors += len(positions)
            loss = sum(originals) / batch_clips + aug["anchor_loss_weight"] * sum(anchors) / batch_anchors
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite classification loss; preserve output and stop")
            total_loss += float(loss.detach())
            original_clips += batch_clips
            anchor_count += batch_anchors
            if loss.requires_grad:
                loss.backward()
                last_gradient = gradient_audit(model)
                for name in gradient_maxima:
                    gradient_maxima[name] = max(gradient_maxima[name], last_gradient[name]["norm"])
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_grad"], error_if_nonfinite=True)
                optimizer.step()
                steps += 1
            else:
                frozen_batches += 1
        if not steps:
            raise RuntimeError("no trainable optimizer step in epoch; do not report successful training")
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError("nonfinite updated parameter; preserve output and stop")
        report, _ = evaluate_chains(bundle, bundle.chains["val"], model,
                                    baseline=initial_base, device=device)
        key = (report["metrics"]["macro_f1"], -report["cross_entropy"])
        history.append({"epoch": epoch, "mean_batch_combined_loss": total_loss / (steps + frozen_batches),
                        "original_clip_visits": original_clips, "anchor_view_visits": anchor_count,
                        "optimizer_steps": steps, "all_frozen_batches": frozen_batches,
                        "gradient_max_norm": gradient_maxima, "last_gradient": last_gradient,
                        "augmentation": plan_audit, "val": report})
        state = {"model": model.state_dict(), **seals, "variant": variant, "seed": seed,
                 "epoch": epoch, "protocol": protocol}
        torch.save(state, out / "last.pt")
        if best_key is None or key > best_key:
            best_key, stale = key, 0
            torch.save(state, out / "best.pt")
        else:
            stale += 1
        write_json(out / "history.json", history)
        print(json.dumps({"stage": "context_utility", "variant": variant, "seed": seed,
                          "epoch": epoch, "val_macro_f1": key[0], "optimizer_steps": steps}), flush=True)
        if stale >= settings["patience"]:
            break
    saved = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model"], strict=True)
    result, rows = evaluate_chains(bundle, bundle.chains["val"], model, baseline=initial_base, device=device)
    result.update(variant=variant, seed=seed, best_epoch=saved["epoch"], epochs_completed=len(history),
                  elapsed_seconds=time.monotonic() - start,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  added_parameters=model.added_parameter_count(), A_trainable_parameters=0,
                  best_sha256=sha256_file(out / "best.pt"),
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None)
    write_json(out / "result.json", result, exclusive=True)
    write_jsonl(out / "val_predictions.jsonl", rows)
    return result


def validate_smoke(report, bundle, config, config_path, checkpoint):
    expected = sealed_inputs(bundle, config, checkpoint)
    if not report.get("all_passed") or not report.get("real_cuda_context_utility_smoke_completed"):
        raise RuntimeError("formal training requires a successful real CUDA frozen-evidence smoke")
    if report.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("smoke config bytes differ")
    for key in ("fingerprint", "base_checkpoint_sha256", "source_sha256", "external_baseline_sha256", "legacy_audit_sha256"):
        if report.get(key) != expected[key]:
            raise RuntimeError(f"smoke seal differs: {key}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-index", "source-protocol-dir", "config", "base-checkpoint", "output", "smoke-report"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    args = parser.parse_args(argv)
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal training requires CUDA; CPU helpers are only for synthetic tests")
    config = load_config(args.config)
    bundle = load_bundle(args.feature_index, args.source_protocol_dir)
    validate_smoke(json.loads(Path(args.smoke_report).read_text()), bundle, config, args.config, args.base_checkpoint)
    train_variant(bundle, config, args.base_checkpoint, args.seed, args.variant, args.output, args.device)


if __name__ == "__main__":
    main()
