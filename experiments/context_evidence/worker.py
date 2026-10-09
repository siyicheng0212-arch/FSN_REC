"""One immutable stage; private evidence stays on the server."""
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
from torch.nn import functional as F

from experiments.direction_gate.model import train_statistics
from experiments.direction_gate.worker import (base_model, native_outputs, replacement_plan,
    replacement_segment, challenge_outputs, simple_metrics, retention, gradient_norms)
from experiments.relation.data import read_jsonl, sha256_file
from experiments.relation.evaluate_suite import _paired
from experiments.relation.tcn_baseline import TCNConfig, Segment, collate_segments
from experiments.videomae_tcn.evidence import prepare
from experiments.videomae_tcn.io import atomic_json, write_json
from experiments.videomae_tcn.metrics import summarize
from experiments.videomae_tcn.worker import _torch_save
from .model import ContextEvidenceTCN, VARIANTS, AUGMENTED, CONTEXT_VARIANTS
from .run import SCHEMA, TRAINING, unchanged


def seed_all():
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)


def make_model(p, statistics, variant, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        m = ContextEvidenceTCN(TCNConfig(**p["evidence"]["model_config"]), variant, statistics)
    if variant != "clip_only_matched":
        ck = torch.load(Path(p["base_run"]) / "tcn/best.pt", map_location="cpu", weights_only=True)
        m.load_old(ck["model"])
    return m.to(device)


def audit(p, out, device):
    segments, metadata, info = prepare(p["index"], p["source_protocol_dir"])
    old = base_model(p, info, device)
    evidence, plans, report = {}, {}, {}
    for split in ("train", "val"):
        data = native_outputs(old, segments[split], device)
        flat = [Segment([clip], s.features[j:j+1], s.a_logits[j:j+1], s.labels[j:j+1], [s.source_kinds[j]])
                for s in segments[split] for j, clip in enumerate(s.clip_ids)]
        center = native_outputs(old, flat, device)
        if center["clip_ids"] != data["clip_ids"]:
            raise RuntimeError("counterfactual clip order changed")
        if split == "val":
            historical = read_jsonl(Path(p["base_run"]) / "evaluation/private_val_predictions.jsonl")
            lookup = {r["clip_id"]: r for r in historical}
            if len(lookup) != len(historical) or set(lookup) != set(data["clip_ids"]):
                raise RuntimeError("historical 823 prediction coverage changed")
            for i, clip in enumerate(data["clip_ids"]):
                row = lookup[clip]
                if (row["label_id"], row["a_prediction"], row["tcn_prediction"]) != (
                        int(data["labels"][i]), int(data["a"][i].argmax()), int(data["t"][i].argmax())):
                    raise RuntimeError("historical TCN per-clip replay failed")
        data["center"] = center["t"]
        data["context_candidate"] = data["a"] + data["t"] - data["center"]
        plans[split] = replacement_plan(segments[split], metadata, split, seed=42)
        data["challenge_t"] = challenge_outputs(old, segments[split], data, plans[split], device)
        data["challenge_context"] = data["a"] + data["challenge_t"] - data["center"]
        lengths = [len(s.clip_ids) for s in segments[split]]
        report[split] = {"count": len(data["labels"]), "A": simple_metrics(data, data["a"]),
            "historical_TCN": simple_metrics(data, data["t"]), "center_only_same_checkpoint": simple_metrics(data, data["center"]),
            "context_delta_no_training": simple_metrics(data, data["context_candidate"]),
            "synthetic_context_delta": simple_metrics(data, data["challenge_context"]),
            "synthetic_anchor_count": len(plans[split]["replacements"]),
            "segments": {"count": len(lengths), "median_clips": float(np.median(lengths)),
                         "maximum_clips": max(lengths), "singleton_count": lengths.count(1)}}
        evidence[split] = data
    if not plans["train"]["replacements"]:
        raise RuntimeError("no eligible train-only unchanged-anchor augmentation evidence")
    generator = torch.Generator().manual_seed(42)
    evidence["segment_orders"] = torch.stack([torch.randperm(len(segments["train"]), generator=generator)
                                              for _ in range(TRAINING["epochs"])])
    evidence["anchor_orders"] = torch.stack([torch.randperm(len(plans["train"]["replacements"]), generator=generator)
                                             for _ in range(TRAINING["epochs"])])
    evidence["statistics"] = train_statistics(evidence["train"]["a"], evidence["train"]["context_candidate"])
    _torch_save(out / "private_evidence.pt", evidence)
    write_json(out / "private_anchor_plans.json", plans)
    write_json(out / "evidence_seal.json", {"cache_sha256": sha256_file(out / "private_evidence.pt"),
               "plans_sha256": sha256_file(out / "private_anchor_plans.json"),
               "segment_orders_sha256": hashlib.sha256(evidence["segment_orders"].numpy().tobytes()).hexdigest(),
               "anchor_orders_sha256": hashlib.sha256(evidence["anchor_orders"].numpy().tobytes()).hexdigest()})
    write_json(out / "diagnostic_summary.json", {"historical_val_replay_exact": True, "test_read": False,
               "design_is_exploratory_after_prior_validation_audit": True, "splits": report,
               "native_train_A_perfect_is_not_honest_out_of_fold_evidence": True})
    print(json.dumps({"stage": "audit", "native_train_context_pair": report["train"]["context_delta_no_training"]["paired_vs_A"],
                      "train_synthetic_pair": report["train"]["synthetic_context_delta"]["paired_vs_A"]}), flush=True)


def cache(out):
    seal = json.loads((out / "evidence_seal.json").read_text())
    if sha256_file(out / "private_evidence.pt") != seal["cache_sha256"] or sha256_file(out / "private_anchor_plans.json") != seal["plans_sha256"]:
        raise RuntimeError("private audit inputs changed")
    data = torch.load(out / "private_evidence.pt", map_location="cpu", weights_only=True)
    for key in ("segment_orders", "anchor_orders"):
        if hashlib.sha256(data[key].numpy().tobytes()).hexdigest() != seal[key + "_sha256"]:
            raise RuntimeError("shared sampling orders changed")
    return data, json.loads((out / "private_anchor_plans.json").read_text())


def batch_loss(model, batch, device, anchor_positions=None):
    f, a, y, valid, _ = batch
    f, a, y, valid = (v.to(device) for v in (f, a, y, valid))
    prediction = model(f, a, valid)
    if not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("nonfinite context-evidence logits")
    if anchor_positions is None:
        return F.cross_entropy(prediction[valid], y[valid])
    positions = torch.as_tensor(anchor_positions, device=device)
    rows = torch.arange(len(positions), device=device)
    # Donor/replaced positions are deliberately NEVER supervised.
    return F.cross_entropy(prediction[rows, positions], y[rows, positions])


def anchor_batch(segments, plan, indices, augmented):
    rows = [plan["replacements"][int(i)] for i in indices]
    views = []
    for r in rows:
        original = segments[r["segment"]]
        view = replacement_segment(segments, r) if augmented else original
        j = r["anchor_position"]
        if not np.array_equal(view.features[j], original.features[j]) or not np.array_equal(view.a_logits[j], original.a_logits[j]) \
                or view.labels[j] != original.labels[j]:
            raise RuntimeError("unchanged anchor features, A logits or label violated")
        views.append(view)
    return collate_segments(views), [r["anchor_position"] for r in rows]


def smoke(p, out, segments, data, plans, device):
    items = [s for s in segments["train"] if len(s.clip_ids) > 1][:16]
    batch = collate_segments(items)
    historical = base_model(p, p["evidence"], device)
    with torch.no_grad():
        historical_batch = historical(batch[0].to(device), batch[1].to(device), batch[3].to(device))
    report = {}
    for v in VARIANTS:
        seed_all(); m = make_model(p, data["statistics"], v, device).eval()
        with torch.no_grad():
            initial, details = m(batch[0].to(device), batch[1].to(device), batch[3].to(device), return_details=True)
            if v != "clip_only_matched":
                torch.testing.assert_close(details["full"], historical_batch, atol=2e-5, rtol=2e-5)
            elif not torch.equal(initial, batch[1].to(device)):
                raise RuntimeError("fresh clip-only zero correction did not reproduce A")
        m.train()
        opt = torch.optim.AdamW((w for w in m.parameters() if w.requires_grad), lr=.001, weight_decay=.01)
        anchor, positions = anchor_batch(segments["train"], plans["train"],
            list(range(min(16, len(plans["train"]["replacements"])))), v in AUGMENTED)
        rows = []
        for step in range(3):
            before = {n: w.detach().clone() for n, w in m.named_parameters()}
            opt.zero_grad(set_to_none=True)
            loss = .5 * (batch_loss(m, batch, device) + batch_loss(m, anchor, device, positions))
            loss.backward(); grads = gradient_norms(m)
            if not grads or max(grads.values()) <= 0:
                raise RuntimeError("missing real numeric-feature gradients")
            if v == "context_delta_gate_aug" and max(grads.get("gate_bias", 0), grads.get("gate_weight", 0)) <= 0:
                raise RuntimeError("context coefficient received no finite real-feature gradient")
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step()
            change = max(float((w.detach()-before[n]).abs().max()) for n, w in m.named_parameters())
            if change <= 0:
                raise RuntimeError("real CUDA smoke did not update parameters")
            rows.append({"step": step+1, "loss": float(loss.detach()), "gradients": grads, "maximum_update": change})
        m.eval()
        single = collate_segments([Segment([s.clip_ids[0]], s.features[:1], s.a_logits[:1], s.labels[:1], s.source_kinds[:1]) for s in items])
        with torch.no_grad():
            if v in CONTEXT_VARIANTS and not torch.equal(m(single[0].to(device), single[1].to(device), single[3].to(device)), single[1].to(device)):
                raise RuntimeError("context singleton did not exactly preserve A")
        report[v] = rows
    ab, ap = anchor_batch(segments["train"], plans["train"], list(range(min(16, len(plans["train"]["replacements"])))), True)
    if not ap:
        raise RuntimeError("smoke did not verify paired donor augmentation")
    write_json(out / "cuda_smoke.json", {"all_passed": True, "steps_per_model": 3,
        "A_loaded_or_trained": False, "anchor_donor_pair_checks": True, "precision": "float32",
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)},
        "variants": report})


@torch.no_grad()
def predictions(model, segments, device, mode="learned"):
    private, coefficients, deltas, logits = [], [], [], []
    model.eval()
    for start in range(0, len(segments), 16):
        batch = collate_segments(segments[start:start+16])
        out, details = model(batch[0].to(device), batch[1].to(device), batch[3].to(device), mode=mode, return_details=True)
        if not bool(torch.isfinite(out).all()):
            raise RuntimeError("nonfinite validation output")
        out = out.cpu()
        for b, s in enumerate(batch[4]):
            n = len(s.clip_ids)
            private.extend({"clip_id": clip, "label_id": int(s.labels[j]), "a_prediction": int(s.a_logits[j].argmax()),
                            "tcn_prediction": int(out[b, j].argmax())} for j, clip in enumerate(s.clip_ids))
            coefficients.append(details["coefficient"][b, :n].cpu())
            deltas.append(details["context_delta"][b, :n].cpu()); logits.append(out[b, :n])
    return private, torch.cat(logits), torch.cat(coefficients), torch.cat(deltas)


def train(p, out, segments, data, plans, variant, device):
    directory = out / variant; directory.mkdir(mode=0o700, exist_ok=False)
    seed_all(); model = make_model(p, data["statistics"], variant, device)
    optimizer = torch.optim.AdamW((w for w in model.parameters() if w.requires_grad), lr=.001, weight_decay=.01)
    torch.cuda.reset_peak_memory_stats(device)
    began, history, best, steps = time.monotonic(), [], -math.inf, 0
    for epoch in range(30):
        model.train()
        # A fixed per-batch seed prevents gate construction or smoke RNG from
        # changing common-width controls' dropout schedule.
        order = data["segment_orders"][epoch].tolist()
        batches = [order[k:k+16] for k in range(0, len(order), 16)]
        anchors = np.array_split(data["anchor_orders"][epoch].numpy(), len(batches))
        normal_sum, anchor_sum, normal_visits, anchor_visits, grad_max = 0., 0., 0, 0, {}
        start = time.monotonic()
        for b, indices in enumerate(batches):
            torch.manual_seed(42 + epoch * len(batches) + b)
            torch.cuda.manual_seed_all(42 + epoch * len(batches) + b)
            batch = collate_segments([segments["train"][i] for i in indices])
            ab, positions = anchor_batch(segments["train"], plans["train"], anchors[b], variant in AUGMENTED)
            optimizer.zero_grad(set_to_none=True)
            normal = batch_loss(model, batch, device)
            extra = batch_loss(model, ab, device, positions)
            loss = .5 * (normal + extra)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite formal loss")
            loss.backward()
            for n, g in gradient_norms(model).items():
                grad_max[n] = max(grad_max.get(n, 0.), g)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step(); steps += 1
            nv, av = int(batch[3].sum()), len(positions)
            normal_sum += float(normal.detach()) * nv; anchor_sum += float(extra.detach()) * av
            normal_visits += nv; anchor_visits += av
        if normal_visits != p["evidence"]["counts"]["train"] or anchor_visits != len(plans["train"]["replacements"]) \
                or not grad_max or max(grad_max.values()) <= 0:
            raise RuntimeError("incomplete common training coverage or missing gradients")
        private, logits, coeff, delta = predictions(model, segments["val"], device)
        metrics = simple_metrics(data["val"], logits)
        row = {"phase": "context_evidence_train", "epoch": epoch+1,
            "train_loss": .5*(normal_sum/normal_visits + anchor_sum/anchor_visits),
            "normal_CE": normal_sum/normal_visits, "anchor_CE": anchor_sum/anchor_visits,
            "normal_clip_visits": normal_visits, "anchor_visits": anchor_visits, "cumulative_steps": steps,
            "val_macro_f1": metrics["macro_f1"], "val_accuracy": metrics["accuracy"],
            "gradient_max": grad_max, "seconds": time.monotonic()-start}
        history.append(row)
        if metrics["macro_f1"] > best + 1e-12:
            best = metrics["macro_f1"]
            _torch_save(directory / "best.pt", {"schema": SCHEMA, "variant": variant, "model": model.state_dict(),
                "statistics": data["statistics"], "epoch": epoch+1, "macro_f1": best,
                "source_commit": p["source_commit"], "input_sha256": p["evidence"]["input_sha256"],
                "base_sha256": p["base_sha256"], "training": TRAINING})
        atomic_json(directory / "history.json", history)
        print(json.dumps({"variant": variant, "epoch": epoch+1, "train_loss": row["train_loss"],
                          "val_macro_f1": metrics["macro_f1"], "val_accuracy": metrics["accuracy"], "best": best}), flush=True)
    ck = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    write_json(directory / "result.json", {"training_completed": True, "test_read": False, "variant": variant,
        "epochs_completed": len(history), "optimizer_steps": steps, "best_epoch": ck["epoch"], "best_macro_f1": best,
        "best_sha256": sha256_file(directory / "best.pt"), "parameters": sum(w.numel() for w in model.parameters()),
        "trainable_parameters": sum(w.numel() for w in model.parameters() if w.requires_grad),
        "classifier_bias_is_cancelled_and_frozen": variant in CONTEXT_VARIANTS,
        "new_gate_parameters": 7 if variant == "context_delta_gate_aug" else 0,
        "A_parameters_trained": 0, "initialized_from_old_TCN": variant != "clip_only_matched",
        "receptive_field_clips": model.receptive_field_clips,
        "resources": {"elapsed_seconds": time.monotonic()-began,
                      "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                      "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}})


@torch.no_grad()
def challenge(model, segments, plans, native, device):
    logits = native.clone()
    lookup = {clip: i for i, clip in enumerate(c for s in segments for c in s.clip_ids)}
    for row in plans["replacements"]:
        batch = collate_segments([replacement_segment(segments, row)])
        output = model(batch[0].to(device), batch[1].to(device), batch[3].to(device))
        value = output[0, row["anchor_position"]].cpu()
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("nonfinite unchanged-anchor challenge")
        logits[lookup[row["anchor_clip_id"]]] = value
    return logits


def evaluate(p, out, segments, metadata, data, plans, device):
    directory = out / "evaluation"; directory.mkdir(mode=0o700, exist_ok=False)
    report = {"schema": SCHEMA, "source_commit": p["source_commit"], "training_completed": True,
        "validation_role": "val823_development_not_independent_test", "test_read": False, "A_frozen": True,
        "definition_selected_after_prior_validation_audit": True, "counts": p["evidence"]["counts"],
        "training": TRAINING, "source_and_identity_are_not_model_inputs": True,
        "A": simple_metrics(data["val"], data["val"]["a"]),
        "historical_TCN": simple_metrics(data["val"], data["val"]["t"]), "variants": {},
        "synthetic_challenge_not_real_edit_boundaries": True}
    old = data["val"]["t"]
    all_logits = {}
    for v in VARIANTS:
        result = json.loads((out / v / "result.json").read_text())
        ck = torch.load(out / v / "best.pt", map_location="cpu", weights_only=True)
        if not result["training_completed"] or sha256_file(out / v / "best.pt") != result["best_sha256"] \
                or ck["source_commit"] != p["source_commit"] or ck["input_sha256"] != p["evidence"]["input_sha256"] \
                or ck["base_sha256"] != p["base_sha256"] or ck["training"] != TRAINING or ck["variant"] != v:
            raise RuntimeError("best checkpoint or training seal changed")
        m = make_model(p, ck["statistics"], v, device); m.load_state_dict(ck["model"], strict=True)
        private, logits, coeff, delta = predictions(m, segments["val"], device)
        metrics = simple_metrics(data["val"], logits)
        if not math.isclose(metrics["macro_f1"], result["best_macro_f1"], rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("best checkpoint replay differs from training selection")
        agg = summarize(private, metadata, segments["val"])
        challenged = challenge(m, segments["val"], plans["val"], logits, device)
        report["variants"][v] = {"training_result": result, "metrics": metrics, "aggregate": agg,
            "paired_vs_historical_TCN": _paired(data["val"]["labels"].numpy(), logits.argmax(-1).numpy(), old.argmax(-1).numpy()),
            "retention": retention(data["val"]["labels"], data["val"]["a"], old, logits),
            "synthetic_cross_record": simple_metrics(data["val"], challenged),
            "coefficient": {"mean": float(coeff.mean()), "std": float(coeff.std(unbiased=False)),
                            "quantiles": torch.quantile(coeff, torch.tensor([0., .1, .5, .9, 1.])).tolist()}}
        if v == "context_delta_gate_aug":
            _, unit, _, _ = predictions(m, segments["val"], device, mode="unit")
            report["variants"][v]["same_checkpoint_g1"] = simple_metrics(data["val"], unit)
            by_id = {r["clip_id"]: r for r in metadata}
            groups = {}
            for i, row in enumerate(private):
                key = (by_id[row["clip_id"]]["source_kind"], int(data["val"]["a"][i].argmax()))
                groups.setdefault(key, []).append(i)
            generator = np.random.default_rng(42)
            perm = torch.arange(len(coeff))
            for positions in groups.values():
                if len(positions) > 1:
                    shift = int(generator.integers(1, len(positions)))
                    perm[positions] = torch.tensor(np.roll(positions, shift).copy())
            moved = int((perm != torch.arange(len(coeff))).sum())
            changed = int((coeff[perm] != coeff).sum())
            shuffled = data["val"]["a"] + coeff[perm, None] * delta
            report["variants"][v]["coefficient_permutation"] = {"seed": 42,
                "grouping": "source_kind_for_audit_only_and_predicted_A_class",
                "moved_positions": moved, "changed_coefficient_positions": changed,
                "effective_changed_context_positions": int(((coeff[perm] != coeff) & (delta.abs().sum(-1) > 0)).sum()),
                "metrics": simple_metrics(data["val"], shuffled)}
        all_logits[v] = logits
        with (directory / (v + "_private_val_predictions.jsonl")).open("x") as stream:
            for row in private:
                stream.write(json.dumps(row)+"\n")
    base = all_logits["legacy_aug"]
    for v in VARIANTS:
        report["variants"][v]["paired_vs_same_budget_legacy_aug"] = _paired(data["val"]["labels"].numpy(), all_logits[v].argmax(-1).numpy(), base.argmax(-1).numpy())
    # Allow only an aggregate safe summary to leave the private output.
    serialized = json.dumps(report)
    if any(k in serialized for k in ('"clip_id"', '"recording_key"', '/root/', '/Users/', '"feature_path"')):
        raise RuntimeError("private key or path leaked into safe aggregate")
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
    if args.stage == "audit":
        audit(p, out, device); return
    data, plans = cache(out)
    segments, metadata, info = prepare(p["index"], p["source_protocol_dir"])
    if info != p["evidence"]:
        raise RuntimeError("frozen evidence seal changed")
    if args.stage == "smoke":
        smoke(p, out, segments, data, plans, device)
    elif args.stage.startswith("train_") and args.stage[6:] in VARIANTS:
        train(p, out, segments, data, plans, args.stage[6:], device)
    elif args.stage == "evaluate":
        evaluate(p, out, segments, metadata, data, plans, device)
    else:
        raise ValueError("unknown stage")


if __name__ == "__main__":
    main()
