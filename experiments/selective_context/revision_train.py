"""Train only G over the verified historical old A+TCN on frozen evidence.

No relation model, edge cutting, TCN continuation, or A finetune is performed.
Because the old TCN saw train7372 during its original fit, stacking G on its
in-sample train outputs is an explicit distribution-shift risk. This first
development experiment must not be presented as out-of-fold gate training.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from experiments.context_utility.config import (
    baseline_source_hashes, build_base, load_config as load_legacy_config,
)
from experiments.context_utility.data import load_bundle
from experiments.fsn_tcn.io import fresh_output, load_chain, write_json, write_jsonl
from experiments.legacy_tcn_audit.audit import load_historical_predictions
from experiments.legacy_tcn_audit.metrics import assert_known_history, summary as historical_summary
from experiments.relation.data import sha256_file
from .revision_config import VARIANTS, load_config
from .revision_evaluate import _historical_segments, evaluate_revision
from .revision_model import RevisionGateTCN


ROOT = Path(__file__).resolve().parents[2]


def revision_source_hashes():
    # The launcher includes every dependency it audits. Import here to avoid
    # an import cycle during the launcher's read-only preparation.
    from .revision_run_suite import source_hashes
    return source_hashes()


@torch.inference_mode()
def _historical_parity(bundle, legacy_base, historical, chain_layout, device="cpu"):
    """Verify actual old logits/classes under precisely its old chain layout."""
    seen = set()
    legacy_base.eval()
    for chain in bundle.chains["val"]:
        ids = chain["ordered_clip_ids"]
        features, a_logits, _ = load_chain(bundle, chain, device)
        for start, stop in _historical_segments(chain, chain_layout):
            output = legacy_base(features[start:stop], a_logits[start:stop])
            if output.shape != (stop - start, 7) or not bool(torch.isfinite(output).all()):
                raise ValueError("audited historical TCN returned invalid scores")
            for clip, scores in zip(ids[start:stop], output):
                if clip in seen:
                    raise ValueError("historical parity includes duplicate val clip")
                seen.add(clip)
                observed = historical[clip]
                if int(scores.argmax()) != observed["prediction"]:
                    raise ValueError("legacy checkpoint does not reproduce historic val823 classes")
                if "tcn_logits" in observed:
                    target = torch.tensor(observed["tcn_logits"], device=scores.device, dtype=scores.dtype)
                    if not torch.allclose(scores, target, atol=1e-5, rtol=0):
                        raise ValueError("legacy checkpoint does not reproduce historic val823 logits")
    if seen != set(historical) or len(seen) != 823:
        raise ValueError("historical checkpoint parity must cover exactly all val823")
    return {"class_parity_clips": 823, "logit_parity_checked": all("tcn_logits" in row for row in historical.values()),
            "chain_layout": chain_layout}


def prepare_revision_inputs(args, device="cpu"):
    """Read-only strict legacy load, exact historical parity and data isolation."""
    config = load_config(args.config)
    legacy_config = load_legacy_config(args.legacy_config)
    if legacy_config["baseline"]["kind"] != "legacy":
        raise ValueError("revision needs an actually verified old A+TCN, never the new reference TCN")
    bundle = load_bundle(args.feature_index, args.source_protocol_dir)
    feature_dim = bundle.index.global_dim + bundle.index.local_dim
    base, base_audit = build_base(legacy_config, feature_dim, args.legacy_checkpoint,
                                  device=device, fingerprint=bundle.fingerprint)
    historical = load_historical_predictions(bundle, args.historical_predictions,
        column_map=getattr(args, "historical_column_map", None),
        logits_key=getattr(args, "historical_logits_key", None))
    # Preserve the known 50 corrections, 15 spoils, 5 wrong-to-wrong and
    # direction-specific historical mistakes. A different saved prediction
    # cohort is not an interchangeable old A+TCN baseline.
    assert_known_history(historical_summary([historical[clip] for clip in sorted(historical)]))
    parity = _historical_parity(bundle, base, historical, args.chain_layout, device)
    return {"bundle": bundle, "base": base.eval().requires_grad_(False),
            "config": config, "legacy_config": legacy_config,
            "base_audit": base_audit, "parity": parity, "historical": historical}


def revision_smoke_seal(args, prepared):
    """Prevent smoke success from being reused after any input/code changes."""
    return {"fingerprint": prepared["bundle"].fingerprint,
            "config_sha256": sha256_file(args.config),
            "legacy_config_sha256": sha256_file(args.legacy_config),
            "legacy_checkpoint_sha256": sha256_file(args.legacy_checkpoint),
            "historical_predictions_sha256": sha256_file(args.historical_predictions),
            "historical_column_map_sha256": sha256_file(args.historical_column_map)
                if getattr(args, "historical_column_map", None) else None,
            "historical_logits_key": getattr(args, "historical_logits_key", None),
            "chain_layout": args.chain_layout,
            "source_sha256": revision_source_hashes(),
            "external_baseline_sha256": baseline_source_hashes(prepared["legacy_config"]),
            "legacy_audit_sha256": sha256_file(prepared["legacy_config"]["baseline"]["legacy_audit"]),
            "parity": prepared["parity"]}


def verify_smoke(args, prepared):
    report = json.loads(Path(args.smoke_report).read_text(encoding="utf-8"))
    if report.get("all_passed") is not True or report.get("real_cuda_tcn_revision_smoke_completed") is not True:
        raise ValueError("formal revision training requires successful real CUDA smoke")
    for key, expected in revision_smoke_seal(args, prepared).items():
        if report.get(key) != expected:
            raise ValueError(f"smoke input/source mismatch: {key}")


def make_revision_model(prepared, variant, seed, device="cpu"):
    if variant not in VARIANTS or seed not in prepared["config"]["seeds"]:
        raise ValueError("revision variant/seed not predeclared")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    bundle = prepared["bundle"]
    feature_dim = bundle.index.global_dim + bundle.index.local_dim
    return RevisionGateTCN(deepcopy(prepared["base"]), feature_dim,
                           gate_dim=prepared["config"]["module"]["gate_dim"],
                           variant=variant, freeze_base=True).to(device)


def _train_chain(bundle, chain, model, device, chain_layout):
    features, a_logits, labels = load_chain(bundle, chain, device)
    total = None
    for start, stop in _historical_segments(chain, chain_layout):
        logits = model(features[start:stop], a_logits[start:stop])
        term = F.cross_entropy(logits.float(), labels[start:stop], reduction="sum")
        total = term if total is None else total + term
    return total, len(labels)


def _gate_gradient(model):
    norm_sq, active = 0.0, 0
    for name, parameter in model.named_parameters():
        if name.startswith("base_tcn."):
            if parameter.requires_grad or parameter.grad is not None:
                raise RuntimeError("old TCN unexpectedly received gradients")
            continue
        if not parameter.requires_grad:
            continue
        grad = parameter.grad
        if grad is not None:
            if not bool(torch.isfinite(grad).all()):
                raise RuntimeError("nonfinite G gradient")
            norm = float(grad.detach().float().norm())
            norm_sq += norm ** 2
            active += int(norm > 0)
    return {"nonzero_gate_tensors": active, "norm": norm_sq ** .5}


def train_variant(args, prepared=None):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal gate training requires a real CUDA GPU")
    out = Path(args.output).resolve()
    if out.exists() or out.is_relative_to(ROOT):
        raise ValueError("output must be fresh and private outside the code worktree")
    prepared = prepare_revision_inputs(args, device) if prepared is None else prepared
    verify_smoke(args, prepared)
    config, bundle = prepared["config"], prepared["bundle"]
    if args.seed not in config["seeds"] or args.variant not in VARIANTS:
        raise ValueError("variant/seed not fixed in experiment config")
    model = make_revision_model(prepared, args.variant, args.seed, device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters or any(parameter.requires_grad for parameter in model.base_tcn.parameters()):
        raise RuntimeError("trainable parameter set must contain only G")
    settings = config["optimization"]
    optimizer = torch.optim.AdamW(parameters, lr=settings["lr"], weight_decay=settings["weight_decay"])
    seal = revision_smoke_seal(args, prepared)
    protocol = {"schema": "fsn-tcn-revision-training-v1", **seal,
                "variant": args.variant, "seed": args.seed,
                "A_trainable_parameters": 0, "old_TCN_trainable_parameters": 0,
                "G_trainable_parameters": model.added_parameter_count(),
                "optimization": settings,
                "training": "gate-only CE on train7372 using original historical TCN chain layout",
                "selection": "val823 Macro-F1 with lower CE as tie break",
                "evaluation_role": "val823_development_not_independent_test",
                "in_sample_stacking_risk": "old TCN was trained on train7372; gate training sees its in-sample predictions; group-OOF old-TCN train logits needed for defensible generalization claims",
                "old_TCN_scores_unchanged": True, "no_R_no_cut": True,
                "test_split_read": False}
    fresh_output(out)
    write_json(out / "protocol.json", protocol, exclusive=True)
    history, best_key, stale, start_time = [], None, 0, time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order = list(bundle.chains["train"])
        random.Random(args.seed + epoch * 65537).shuffle(order)
        total_CE, visits, steps, gradient_max = 0.0, 0, 0, 0.0
        gradient_batches = 0
        for offset in range(0, len(order), settings["chains_per_step"]):
            optimizer.zero_grad(set_to_none=True)
            cumulative, n = None, 0
            for chain in order[offset:offset + settings["chains_per_step"]]:
                loss, count = _train_chain(bundle, chain, model, device, args.chain_layout)
                cumulative = loss if cumulative is None else cumulative + loss
                n += count
            if n == 0:
                raise RuntimeError("train batch was empty")
            loss = cumulative / n
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite gate training loss")
            total_CE += float(cumulative.detach())
            visits += n
            loss.backward()
            gradient = _gate_gradient(model)
            gradient_max = max(gradient_max, gradient["norm"])
            gradient_batches += int(gradient["nonzero_gate_tensors"] > 0)
            torch.nn.utils.clip_grad_norm_(parameters, settings["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            steps += 1
        if steps == 0 or gradient_batches == 0:
            raise RuntimeError("G never received a nonzero gradient; preserve partial output")
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError("updated gate has nonfinite parameters")
        val, _ = evaluate_revision(bundle, bundle.chains["val"], model,
                                   legacy_base=prepared["base"], chain_layout=args.chain_layout,
                                   variant=args.variant, device=device)
        key = (val["metrics"]["new"]["macro_f1"], -val["cross_entropy"])
        history.append({"epoch": epoch, "train_CE": total_CE / visits, "optimizer_steps": steps,
                        "gradient_batches": gradient_batches, "gradient_max_norm": gradient_max,
                        "val": val})
        checkpoint = {"schema": "fsn-tcn-revision-checkpoint-v1", "model": model.state_dict(),
                      "input_seal": seal, "variant": args.variant, "seed": args.seed,
                      "epoch": epoch, "protocol": protocol}
        torch.save(checkpoint, out / "last.pt")
        if best_key is None or key > best_key:
            best_key, stale = key, 0
            torch.save(checkpoint, out / "best.pt")
        else:
            stale += 1
        write_json(out / "history.json", history)
        print(json.dumps({"stage": "revision_gate", "variant": args.variant, "seed": args.seed,
                          "epoch": epoch, "val_macro_f1": key[0], "optimizer_steps": steps}), flush=True)
        if stale >= settings["patience"]:
            break
    best = torch.load(out / "best.pt", map_location=device, weights_only=True)
    if (best.get("input_seal") != seal or best.get("variant") != args.variant
            or best.get("seed") != args.seed):
        raise ValueError("gate checkpoint lineage changed after fitting")
    model.load_state_dict(best["model"], strict=True)
    model.eval()
    result, rows = evaluate_revision(bundle, bundle.chains["val"], model,
                                     legacy_base=prepared["base"], chain_layout=args.chain_layout,
                                     variant=args.variant, device=device)
    # Compute distribution diagnostics on train; train metrics are never used
    # for checkpoint/threshold selection and individual train rows stay private.
    train_report, _ = evaluate_revision(bundle, bundle.chains["train"], model,
                                        legacy_base=prepared["base"], chain_layout=args.chain_layout,
                                        variant=args.variant, device=device, split="train")
    result.update({"best_epoch": best["epoch"], "epochs_completed": len(history),
                   "elapsed_seconds": time.monotonic() - start_time,
                   "gate_trainable_parameters": model.added_parameter_count(),
                   "old_TCN_trainable_parameters": 0, "A_trainable_parameters": 0,
                   "best_sha256": sha256_file(out / "best.pt"),
                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                   "train_vs_val_confidence_distribution": {
                       "train_in_sample_old_TCN": train_report["confidence_distributions"],
                       "val": result["confidence_distributions"]},
                   "in_sample_stacking_risk": protocol["in_sample_stacking_risk"]})
    write_json(out / "result.json", result, exclusive=True)
    write_jsonl(out / "val_predictions.jsonl", rows)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-index", "source-protocol-dir", "legacy-config", "legacy-checkpoint",
                 "historical-predictions", "chain-layout", "config", "variant", "seed", "output", "smoke-report"):
        kwargs = {}
        if name == "chain-layout":
            kwargs["choices"] = ("full_chain", "eligible_segments")
        if name == "variant":
            kwargs["choices"] = VARIANTS
        if name == "seed":
            kwargs["type"] = int
        p.add_argument("--" + name, required=True, **kwargs)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--historical-column-map")
    p.add_argument("--historical-logits-key")
    return p


def main(argv=None):
    train_variant(parser().parse_args(argv))


if __name__ == "__main__":
    main()
