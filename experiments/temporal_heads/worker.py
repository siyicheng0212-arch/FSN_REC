"""Immutable audit, CUDA smoke, serial head training and sealed evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from experiments.direction_gate.worker import (base_model, native_outputs, replacement_plan,
    replacement_segment, simple_metrics, retention, gradient_norms)
from experiments.relation.data import read_jsonl, sha256_file
from experiments.relation.evaluate_suite import _paired
from experiments.relation.tcn_baseline import collate_segments
from experiments.videomae_tcn.evidence import prepare
from experiments.videomae_tcn.io import atomic_json, write_json
from experiments.videomae_tcn.metrics import summarize
from experiments.videomae_tcn.worker import _torch_save
from .model import CONFIG, VARIANTS, TemporalHead, objective
from .run import SCHEMA, unchanged


def seed_all():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def make_model(variant, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = TemporalHead(variant, 1543)
    return model.to(device)


def audit(p, out, segments, metadata, info, device):
    old = base_model(p, info, device)
    data = {s: native_outputs(old, segments[s], device) for s in ("train", "val")}
    historical = read_jsonl(Path(p["base_run"]) / "evaluation/private_val_predictions.jsonl")
    lookup = {r["clip_id"]: r for r in historical}
    if len(lookup) != len(historical) or set(lookup) != set(data["val"]["clip_ids"]):
        raise RuntimeError("old 823 prediction coverage changed")
    for i, clip in enumerate(data["val"]["clip_ids"]):
        r = lookup[clip]
        if (r["label_id"], r["a_prediction"], r["tcn_prediction"]) != (int(data["val"]["labels"][i]), int(data["val"]["a"][i].argmax()), int(data["val"]["t"][i].argmax())):
            raise RuntimeError("old TCN per-clip replay changed")
    generator = torch.Generator().manual_seed(42)
    data["orders"] = torch.stack([torch.randperm(len(segments["train"]), generator=generator) for _ in range(CONFIG["epochs"])])
    plans = replacement_plan(segments["val"], metadata, "val", seed=42)
    _torch_save(out / "private_evidence.pt", data)
    write_json(out / "private_challenge_plan.json", plans)
    write_json(out / "evidence_seal.json", {"cache_sha256": sha256_file(out / "private_evidence.pt"),
               "plans_sha256": sha256_file(out / "private_challenge_plan.json"),
               "orders_sha256": hashlib.sha256(data["orders"].numpy().tobytes()).hexdigest()})
    write_json(out / "diagnostic_summary.json", {"historical_val_replay_exact": True, "test_read": False,
        "counts": info["counts"], "train_A": simple_metrics(data["train"], data["train"]["a"]),
        "train_historical_TCN": simple_metrics(data["train"], data["train"]["t"]),
        "all_new_heads_fresh": True, "synthetic_val_anchors": len(plans["replacements"]),
        "native_train_outputs_not_honest_out_of_fold": True})
    print(json.dumps({"stage": "audit", "history_replay_823": True, "counts": info["counts"], "train_segments": len(segments["train"])}), flush=True)


def cache(out):
    seal = json.loads((out / "evidence_seal.json").read_text())
    if sha256_file(out / "private_evidence.pt") != seal["cache_sha256"] or sha256_file(out / "private_challenge_plan.json") != seal["plans_sha256"]:
        raise RuntimeError("private evidence changed")
    data = torch.load(out / "private_evidence.pt", map_location="cpu", weights_only=True)
    if hashlib.sha256(data["orders"].numpy().tobytes()).hexdigest() != seal["orders_sha256"]:
        raise RuntimeError("common training order changed")
    return data, json.loads((out / "private_challenge_plan.json").read_text())


def forward_loss(model, batch, device):
    f, a, y, valid = (v.to(device) for v in batch[:4])
    return objective(model(f, a, valid, all_stages=True), y, valid)


def smoke(p, out, segments, device):
    # Include actual singleton and different lengths, not only convenient long chains.
    items = [next(s for s in segments["train"] if len(s.clip_ids) == k) for k in (1, 2, 3, 4)]
    batch = collate_segments(items)
    report = {}
    for variant in VARIANTS:
        seed_all(); model = make_model(variant, device)
        torch.cuda.reset_peak_memory_stats(device)
        model.eval()
        with torch.no_grad():
            together = model(batch[0].to(device), batch[1].to(device), batch[3].to(device))
            for i, item in enumerate(items):
                solo = collate_segments([item])
                separate = model(solo[0].to(device), solo[1].to(device), solo[3].to(device))
                torch.testing.assert_close(together[i, :len(item.clip_ids)], separate[0], atol=2e-5, rtol=2e-5)
        opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        model.train(); rows = []
        for step in range(3):
            before = {n: w.detach().clone() for n, w in model.named_parameters()}
            opt.zero_grad(set_to_none=True)
            loss, ce, smooth = forward_loss(model, batch, device)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("CUDA smoke loss nonfinite")
            loss.backward(); grads = gradient_norms(model)
            if not grads or max(grads.values()) <= 0:
                raise RuntimeError("CUDA smoke has no finite gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_grad_norm"]); opt.step()
            change = max(float((w.detach()-before[n]).abs().max()) for n, w in model.named_parameters())
            if change <= 0:
                raise RuntimeError("CUDA smoke no true parameter update")
            rows.append({"step": step+1, "loss": float(loss.detach()), "stage_CE": float(ce.detach()),
                         "smoothing": float(smooth.detach()), "max_gradient": max(grads.values()),
                         "gradient_parameter_count": len(grads), "max_parameter_update": change})
        report[variant] = {"steps": rows, "parameters": sum(w.numel() for w in model.parameters()),
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                           "independent_segment_eval_equivalence": True}
    write_json(out / "cuda_smoke.json", {"all_passed": True, "steps_per_model": 3,
        "A_loaded_or_trained": False, "precision": "float32", "variants": report,
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)}})
    print(json.dumps({"stage": "smoke", "all_passed": True, "parameter_counts": {v: r["parameters"] for v, r in report.items()}}), flush=True)


@torch.no_grad()
def predict(model, segments, device):
    model.eval(); private, logits = [], []
    for start in range(0, len(segments), CONFIG["batch_size_segments"]):
        items = segments[start:start+CONFIG["batch_size_segments"]]
        batch = collate_segments(items)
        pred = model(batch[0].to(device), batch[1].to(device), batch[3].to(device)).cpu()
        for i, s in enumerate(items):
            for j, clip in enumerate(s.clip_ids):
                private.append({"clip_id": clip, "label_id": int(s.labels[j]),
                                "a_prediction": int(s.a_logits[j].argmax()), "tcn_prediction": int(pred[i, j].argmax())})
                logits.append(pred[i, j])
    return private, torch.stack(logits)


def train(p, out, segments, data, variant, device):
    directory = out / variant; directory.mkdir(mode=0o700, exist_ok=False)
    seed_all(); model = make_model(variant, device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    torch.cuda.reset_peak_memory_stats(device)
    history, best, steps = [], -math.inf, 0
    began = time.monotonic()
    for epoch, order in enumerate(data["orders"]):
        model.train(); weighted, ce_sum, sm_sum, visits, gmax = 0., 0., 0., 0, 0.
        start = time.monotonic()
        for b, indices in enumerate(order.split(CONFIG["batch_size_segments"])):
            torch.manual_seed(42 + epoch * math.ceil(len(order)/16) + b)
            torch.cuda.manual_seed_all(42 + epoch * math.ceil(len(order)/16) + b)
            batch = collate_segments([segments["train"][int(i)] for i in indices])
            opt.zero_grad(set_to_none=True)
            loss, ce, smooth = forward_loss(model, batch, device)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("formal head loss nonfinite")
            loss.backward(); grads = gradient_norms(model)
            if not grads:
                raise RuntimeError("formal head gradients missing")
            gmax = max(gmax, max(grads.values()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_grad_norm"]); opt.step(); steps += 1
            count = int(batch[3].sum()); visits += count
            weighted += float(loss.detach()) * count; ce_sum += float(ce.detach()) * count; sm_sum += float(smooth.detach()) * count
        if visits != p["evidence"]["counts"]["train"] or gmax <= 0:
            raise RuntimeError("incomplete train clip coverage or zero gradients")
        _, logits = predict(model, segments["val"], device)
        metrics = simple_metrics(data["val"], logits)
        row = {"phase": "mature_temporal_head", "epoch": epoch+1, "train_loss": weighted/visits,
               "stage_CE": ce_sum/visits, "smoothing": sm_sum/visits, "train_clip_visits": visits,
               "cumulative_steps": steps, "max_parameter_gradient": gmax, "seconds": time.monotonic()-start,
               "val_macro_f1": metrics["macro_f1"], "val_accuracy": metrics["accuracy"]}
        history.append(row)
        if metrics["macro_f1"] > best + 1e-12:
            best = metrics["macro_f1"]
            _torch_save(directory / "best.pt", {"schema": SCHEMA, "variant": variant, "model": model.state_dict(),
                "epoch": epoch+1, "macro_f1": best, "source_commit": p["source_commit"],
                "input_sha256": p["evidence"]["input_sha256"], "training": CONFIG})
        atomic_json(directory / "history.json", history)
        print(json.dumps({"variant": variant, **row, "best": best}), flush=True)
    ck = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    write_json(directory / "result.json", {"training_completed": True, "test_read": False, "variant": variant,
        "epochs_completed": len(history), "optimizer_steps": steps, "best_epoch": ck["epoch"], "best_macro_f1": best,
        "best_sha256": sha256_file(directory / "best.pt"), "parameters": sum(w.numel() for w in model.parameters()),
        "trainable_parameters": sum(w.numel() for w in model.parameters()), "A_parameters_trained": 0,
        "fresh_head_not_old_checkpoint_continuation": True,
        "resources": {"elapsed_seconds": time.monotonic()-began,
                      "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                      "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}})


@torch.no_grad()
def challenge(model, segments, plan, native, device):
    logits = native.clone()
    lookup = {c: i for i, c in enumerate(c for s in segments for c in s.clip_ids)}
    for row in plan["replacements"]:
        batch = collate_segments([replacement_segment(segments, row)])
        value = model(batch[0].to(device), batch[1].to(device), batch[3].to(device))[0, row["anchor_position"]].cpu()
        logits[lookup[row["anchor_clip_id"]]] = value
    return logits


def evaluate(p, out, segments, metadata, data, plans, device):
    directory = out / "evaluation"; directory.mkdir(mode=0o700, exist_ok=False)
    prior = json.loads((Path(p["prior_context_run"]) / "evaluation/public_safe_summary.json").read_text())
    report = {"schema": SCHEMA, "source_commit": p["source_commit"], "training_completed": True,
        "validation_role": "val823_development_not_independent_test", "test_read": False, "A_frozen": True,
        "counts": p["evidence"]["counts"], "training": CONFIG, "upstream": p["upstream"],
        "architecture_and_loss_differ_not_architecture_causality": True, "variants": {},
        "A": simple_metrics(data["val"], data["val"]["a"]),
        "historical_TCN": simple_metrics(data["val"], data["val"]["t"]),
        "prior_context_results": {v: {"metrics": r["metrics"], "training_result": r["training_result"]} for v, r in prior["variants"].items()},
        "synthetic_challenge_not_real_edit_boundaries": True,
        "source_and_identity_are_not_model_inputs": True,
        "mature_heads_singletons_use_classifier_not_A_fallback": True}
    preds = {}
    for v in VARIANTS:
        result = json.loads((out / v / "result.json").read_text())
        ck = torch.load(out / v / "best.pt", map_location="cpu", weights_only=True)
        if not result["training_completed"] or sha256_file(out / v / "best.pt") != result["best_sha256"] or \
                ck["source_commit"] != p["source_commit"] or ck["input_sha256"] != p["evidence"]["input_sha256"] or ck["training"] != CONFIG or ck["variant"] != v:
            raise RuntimeError("best checkpoint seal mismatch")
        model = make_model(v, device); model.load_state_dict(ck["model"], strict=True)
        private, logits = predict(model, segments["val"], device)
        metrics = simple_metrics(data["val"], logits)
        if not math.isclose(metrics["macro_f1"], result["best_macro_f1"], rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("best checkpoint replay mismatch")
        swapped = challenge(model, segments["val"], plans, logits, device)
        report["variants"][v] = {"training_result": result, "metrics": metrics,
            "aggregate": summarize(private, metadata, segments["val"]),
            "paired_vs_historical_TCN": _paired(data["val"]["labels"].numpy(), logits.argmax(-1).numpy(), data["val"]["t"].argmax(-1).numpy()),
            "retention": retention(data["val"]["labels"], data["val"]["a"], data["val"]["t"], logits),
            "synthetic_cross_record": simple_metrics(data["val"], swapped),
            "synthetic_changed_predictions": int((swapped.argmax(-1) != logits.argmax(-1)).sum()),
            "synthetic_logit_mean_absolute_difference": float((swapped-logits).abs().mean())}
        preds[v] = logits
        with (directory / (v + "_private_val_predictions.jsonl")).open("x") as stream:
            for r in private:
                stream.write(json.dumps(r)+"\n")
    for v in VARIANTS:
        report["variants"][v]["paired_vs_same_budget_plain"] = _paired(data["val"]["labels"].numpy(), preds[v].argmax(-1).numpy(), preds["plain_tcn_fresh"].argmax(-1).numpy())
    serialized = json.dumps(report)
    if any(k in serialized for k in ('"clip_id"', '"recording_key"', '/root/', '/Users/', '"feature_path"')):
        raise RuntimeError("privacy leak in aggregate")
    write_json(directory / "public_safe_summary.json", report)
    print(json.dumps({v: report["variants"][v]["metrics"]["macro_f1"] for v in VARIANTS}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True); parser.add_argument("--run-protocol", required=True)
    args = parser.parse_args(); p = json.loads(Path(args.run_protocol).read_text())
    unchanged(p); torch.set_num_threads(2)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker requires one UUID-mapped CUDA device, never DDP")
    device, out = torch.device("cuda:0"), Path(p["output"])
    segments, metadata, info = prepare(p["index"], p["source_protocol_dir"])
    if info != p["evidence"]:
        raise RuntimeError("frozen evidence seal changed")
    if args.stage == "audit":
        audit(p, out, segments, metadata, info, device); return
    data, plans = cache(out)
    if args.stage == "smoke":
        smoke(p, out, segments, device)
    elif args.stage.startswith("train_") and args.stage[6:] in VARIANTS:
        train(p, out, segments, data, args.stage[6:], device)
    elif args.stage == "evaluate":
        evaluate(p, out, segments, metadata, data, plans, device)
    else:
        raise ValueError("unknown stage")


if __name__ == "__main__":
    main()
