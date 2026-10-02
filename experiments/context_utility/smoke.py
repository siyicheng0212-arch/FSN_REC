"""Three real frozen-evidence CUDA updates before any formal continuation."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.fsn_tcn.io import load_chain, write_json
from experiments.relation.data import sha256_file
from .augmentation import apply_view, build_epoch_plan
from .config import VARIANTS, build_base, load_config
from .data import load_bundle
from .train import gradient_audit, make_model, sealed_inputs


def run_checks(model, baseline, features, a_logits, labels, *, anchor_view=None):
    """Internal CPU fixture helper; it cannot certify a real CUDA preflight."""
    model.eval()
    baseline.eval()
    with torch.no_grad():
        expected = baseline(features, a_logits)
        observed = model(features, a_logits)
        unit = model(features, a_logits, gate_mode="unit")
        singleton = model(features[:1], a_logits[:1])
        singleton_baseline = baseline(features[:1], a_logits[:1])
    # float32 mandatory smoke; exactness under other dtypes is separately tested.
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(unit, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(singleton, singleton_baseline, rtol=1e-5, atol=1e-6)
    before = {key: value.detach().clone() for key, value in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
    gradients, losses, named_gradients = [], [], []
    for step in range(3):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(features, a_logits)
        loss = F.cross_entropy(output.float(), labels)
        if anchor_view is not None:
            view_features, view_logits, anchors = anchor_view
            anchor_index = torch.tensor(anchors, dtype=torch.long, device=features.device)
            torch.testing.assert_close(view_features[anchor_index], features[anchor_index], rtol=0, atol=0)
            torch.testing.assert_close(view_logits[anchor_index], a_logits[anchor_index], rtol=0, atol=0)
            view_output = model(view_features, view_logits)
            loss = loss + F.cross_entropy(view_output[anchor_index].float(), labels[anchor_index])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite smoke loss")
        loss.backward()
        gradients.append(gradient_audit(model))
        named_gradients.append({name: float(p.grad.detach().float().norm())
                                for name, p in model.named_parameters() if p.grad is not None})
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError("nonfinite updated smoke parameter")
        losses.append(float(loss.detach()))
    changed = [key for key, parameter in model.named_parameters()
               if not torch.equal(parameter.detach(), before[key])]
    if not changed or not any(item["base"]["nonzero"] for item in gradients):
        raise RuntimeError("smoke did not update the TCN")
    if model.added_parameter_count() and not any(item["gate"]["nonzero"] for item in gradients):
        raise RuntimeError("smoke did not train the context coefficients")
    gate_updated = sum(not key.startswith("base.") for key in changed)
    if model.added_parameter_count() and not gate_updated:
        raise RuntimeError("smoke gradients did not produce a gate parameter update")
    projection_trainable = None
    early_network_trainable = None
    if model.mode == "dynamic":
        projection_trainable = any(any(name.startswith("gate.projection.") and value > 0
                                   for name, value in row.items()) for row in named_gradients[1:])
        early_network_trainable = any(any(name.startswith("gate.network.0.") and value > 0
                                      for name, value in row.items()) for row in named_gradients[1:])
        if not projection_trainable or not early_network_trainable:
            raise RuntimeError("dynamic projection/early gate layer did not receive a later-step gradient")
    return {"all_passed": True, "dtype": str(features.dtype), "steps": 3,
            "initial_max_logit_difference": float((observed - expected).abs().max()),
            "singleton_matches_own_baseline": True, "gate_one_matches_initial_baseline": True,
            "losses": losses, "gradients": gradients,
            "updated_parameter_tensors": len(changed),
            "updated_gate_parameter_tensors": gate_updated,
            "dynamic_projection_later_step_gradient": projection_trainable,
            "dynamic_early_gate_later_step_gradient": early_network_trainable,
            "paired_anchor_view_exercised": anchor_view is not None,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "added_parameters": model.added_parameter_count(),
            "strict_boundary_isolation_claimed": False}


def run_smoke(bundle, config, checkpoint, device):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real CUDA is required; a CPU fixture cannot certify this smoke")
    aug = config["augmentation"]
    plan, audit = build_epoch_plan(bundle, seed=config["seeds"][0], epoch=1,
                                   anchors_per_segment=aug["anchors_per_segment"],
                                   max_neighbors=aug["max_neighbors"])
    candidates = [entry for entry in plan if entry["replacements"] and len(entry["ordered_clip_ids"]) > 1]
    if not candidates:
        raise ValueError("no real training pair with a train-only clinical donor")
    entry = max(candidates, key=lambda row: min(len(row["ordered_clip_ids"]), 8))
    # Limit only this diagnostic input; training keeps complete original segments.
    length = len(entry["ordered_clip_ids"])
    start = max(0, min(entry["anchor_positions"][0] - 3, length - 8))
    stop = min(length, start + 8)
    diagnostic = {"ordered_clip_ids": entry["ordered_clip_ids"][start:stop],
                  "anchor_positions": [p - start for p in entry["anchor_positions"] if start <= p < stop],
                  "replacements": [{"position": row["position"] - start, "donor_clip_id": row["donor_clip_id"]}
                                   for row in entry["replacements"] if start <= row["position"] < stop]}
    if not diagnostic["replacements"]:
        raise ValueError("smoke diagnostic window has no actual paired replacement")
    features, logits, labels = load_chain(bundle, diagnostic, device)
    def read_donor(clip):
        donor_features, donor_logits, _ = load_chain(bundle, {"ordered_clip_ids": [clip]}, device)
        return donor_features[0], donor_logits[0]
    view_features, view_logits, anchors = apply_view(features, logits, diagnostic, read_donor)
    for replacement in diagnostic["replacements"]:
        donor_features, donor_logits = read_donor(replacement["donor_clip_id"])
        position = replacement["position"]
        torch.testing.assert_close(view_features[position], donor_features, rtol=0, atol=0)
        torch.testing.assert_close(view_logits[position], donor_logits, rtol=0, atol=0)
    reports = {}
    for variant in VARIANTS:
        model, _ = make_model(bundle, config, checkpoint, variant, config["seeds"][0], device)
        baseline, _ = build_base(config, features.shape[1], checkpoint, device,
                                 fingerprint=bundle.fingerprint)
        torch.cuda.reset_peak_memory_stats(device)
        view = (features, logits, anchors) if variant == "continue" else (view_features, view_logits, anchors)
        reports[variant] = run_checks(model, baseline, features, logits, labels, anchor_view=view)
        reports[variant]["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        del model, baseline
        torch.cuda.empty_cache()
    return {"all_passed": True, "real_cuda_context_utility_smoke_completed": True,
            "groups": reports, "augmentation_audit": audit,
            "A_frozen": True, "raw_video_backbone_retrained": False,
            "evidence_scope": "sealed real frozen-A features, not new video decoding or verified PTS"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-index", "source-protocol-dir", "config", "base-checkpoint", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    path = Path(args.output).resolve()
    root = Path(__file__).resolve().parents[2]
    if path.exists() or path.is_relative_to(root):
        raise ValueError("use a new smoke report outside the code worktree")
    report = {"all_passed": False, "real_cuda_context_utility_smoke_completed": False,
              "formal_training_started": False}
    try:
        if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("real CUDA required for formal preflight")
        config = load_config(args.config)
        bundle = load_bundle(args.feature_index, args.source_protocol_dir)
        report.update(sealed_inputs(bundle, config, args.base_checkpoint))
        report["config_sha256"] = sha256_file(args.config)
        report.update(run_smoke(bundle, config, args.base_checkpoint, args.device))
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report, exclusive=True)
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, report, exclusive=True)


if __name__ == "__main__":
    main()
