"""Verified historical replay, all-cell CUDA checks, training and safe aggregation."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.anchor_reference import worker as common
from experiments.anchor_reference.data import anchor_batch
from experiments.input_evidence.worker import eligible_audit, factorial_contrasts, public_safe, signal
from experiments.direction_gate.worker import gradient_norms, retention, simple_metrics
from experiments.paper_validation import perturb
from experiments.relation.data import sha256_file
from experiments.relation.evaluate_suite import _paired
from experiments.relation.tcn_baseline import Segment, TCNConfig, collate_segments
from experiments.videomae_tcn.io import atomic_json, write_json
from experiments.videomae_tcn.metrics import summarize
from experiments.videomae_tcn.worker import _torch_save
from .model import CONFIG, TASKS, BenchmarkHead, objective
from .metrics import recording_interval
from .run_suite import unchanged


def model(p, backbone, task, device):
    return BenchmarkHead(TCNConfig(**p["subplans"][backbone]["evidence"]["model_config"]), task).to(device)


def loss_on_batch(m, batch, device, positions=None):
    f, a, y, valid = [v.to(device) for v in batch[:4]]
    return objective(m(f, a, valid, all_stages=True), y, valid, m.spec, positions)


@torch.no_grad()
def anchor_signal(m, segments, plans, augmented, device):
    m.eval()
    z, y, a = [], [], []
    for start in range(0, len(plans["rows"]), 16):
        batch, positions = anchor_batch(segments, plans, range(start, min(start + 16, len(plans["rows"]))), augmented)
        output = m(batch[0].to(device), batch[1].to(device), batch[3].to(device)).cpu()
        indices = torch.arange(len(positions)), torch.tensor(positions)
        z.append(output[indices]); y.append(batch[2][indices]); a.append(batch[1][indices])
    return signal({"labels": torch.cat(y), "a": torch.cat(a)}, torch.cat(z))


def smoke(p, backbone, task, segments, fixed, device):
    if device.type != "cuda":
        raise RuntimeError("real smoke is CUDA-only; CPU fixtures are not server checks")
    # A real singleton, one short chain, one longer chain and a duplicate length.
    singles = [s for s in segments["train"] if len(s.clip_ids) == 1]
    multis = sorted([s for s in segments["train"] if len(s.clip_ids) > 1], key=lambda s: len(s.clip_ids))
    if not singles or len(multis) < 3:
        raise RuntimeError("real batch4 smoke needs singleton and multiple candidate chains")
    batch = collate_segments([singles[0], multis[0], multis[1], multis[-1]])
    m = model(p, backbone, task, device).eval()
    originals = batch[0].clone(), batch[1].clone()
    with torch.no_grad():
        z = m(batch[0].to(device), batch[1].to(device), batch[3].to(device))
        for i, s in enumerate(batch[4]):
            solo = collate_segments([s])
            torch.testing.assert_close(z[i, :len(s.clip_ids)], m(solo[0].to(device), solo[1].to(device), solo[3].to(device))[0], atol=2e-5, rtol=2e-5)
        changed = batch[0].clone(); changed[..., -7:] += 3
        alter = m(changed.to(device), changed[..., -7:].to(device), batch[3].to(device))
        if m.spec["scores"] == "zero":
            torch.testing.assert_close(z, alter, atol=0, rtol=0)
        elif float((z - alter).abs().max()) == 0:
            raise RuntimeError("raw score control has no sensitivity to the declared inputs")
    opt = torch.optim.AdamW(m.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    ab, positions = anchor_batch(segments["train"], fixed["train_replacements"], range(4), m.spec["augmented"])
    torch.cuda.reset_peak_memory_stats(device)
    updates = []
    m.train()
    for step in range(3):
        torch.manual_seed(42 + step); torch.cuda.manual_seed_all(42 + step)
        before = {n: w.detach().clone() for n, w in m.named_parameters()}
        opt.zero_grad(set_to_none=True)
        normal, _, smooth = loss_on_batch(m, batch, device)
        anchor, _, anchor_smooth = loss_on_batch(m, ab, device, positions)
        loss = .5 * (normal + anchor)
        if not bool(torch.isfinite(loss)) or float(anchor_smooth.detach()) != 0:
            raise RuntimeError("finite loss and no synthetic-view smoothing required")
        loss.backward()
        gradients = gradient_norms(m)
        if not gradients or max(gradients.values()) <= 0:
            raise RuntimeError("smoke requires finite nonzero gradients")
        if m.spec["head"] in ("native_tcn", "center_only"):
            grad = m.core.base.stem[0].weight.grad
            if grad is None or float(grad.abs().sum()) == 0 or \
                    (m.spec["scores"] == "zero" and bool((grad[:, -7:] != 0).any())):
                raise RuntimeError("native stem gradient/input mask failed")
            if float(m.core.base.classifier.weight.grad.abs().sum()) == 0:
                raise RuntimeError("native classification output did not receive gradients")
            for block in m.core.base.blocks:
                if block.conv.weight.grad is None or float(block.conv.weight.grad.abs().sum()) == 0:
                    raise RuntimeError("native temporal/center convolution received no finite gradient")
                if m.spec["head"] == "center_only" and bool((block.conv.weight.grad[:, :, [0, 2]] != 0).any()):
                    raise RuntimeError("center-only control leaked a neighboring convolution term")
        else:
            projection = m.core.PG.conv_1x1_in if m.spec["head"] == "mstcnpp" else m.core.encoder.conv_1x1
            if projection.weight.grad is None or float(projection.weight.grad.abs().sum()) == 0 or \
                    (m.spec["scores"] == "zero" and bool((projection.weight.grad[:, -7:] != 0).any())):
                raise RuntimeError("mature input projection gradient/score mask failed")
        torch.nn.utils.clip_grad_norm_(m.parameters(), CONFIG["clip_gradient_norm"])
        opt.step()
        change = max(float((w.detach() - before[n]).abs().max()) for n, w in m.named_parameters())
        if change <= 0 or any(not bool(torch.isfinite(w).all()) for w in m.parameters()):
            raise RuntimeError("smoke requires real finite parameter updates")
        updates.append({"step": step + 1, "loss": float(loss.detach()), "normal_smoothing": float(smooth.detach()),
                        "gradients": gradients, "maximum_parameter_update": change})
    if not torch.equal(batch[0], originals[0]) or not torch.equal(batch[1], originals[1]):
        raise RuntimeError("shared raw frozen evidence was mutated")
    out = Path(p["subplans"][backbone]["output"])
    (out / "smoke").mkdir(mode=0o700, exist_ok=True)
    write_json(out / "smoke" / (task + ".json"), {"all_passed": True, "task_specification": m.spec,
               "steps": updates, "real_batch_size": 4, "precision": "float32_numeric_feature_head",
               "singleton_padding_independence": True, "A_loaded_or_trained": False,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "gpu_name": torch.cuda.get_device_name(device),
               "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"], "torch": str(torch.__version__), "CUDA": torch.version.cuda})


def checkpoint_identity(p, backbone, task, seal):
    return {"schema": CONFIG["schema"], "source_commit": p["source_commit"], "backbone": backbone, "variant": task,
            "task_specification": TASKS[task], "input_sha256": p["subplans"][backbone]["evidence"]["input_sha256"],
            "training": CONFIG, "sampling_hash": seal["sampling_hash"], "train_replacement_hash": seal["train_replacement_hash"]}


def train(p, backbone, task, segments, fixed, data, seal, device):
    if task in p["reuse"][backbone]:
        raise RuntimeError("explicitly reused completed cells must never be trained again")
    out = Path(p["subplans"][backbone]["output"]) / task
    out.mkdir(mode=0o700, exist_ok=False)
    m = model(p, backbone, task, device)
    opt = torch.optim.AdamW(m.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CONFIG["epochs"], eta_min=CONFIG["minimum_learning_rate"])
    history, best, steps = [], -math.inf, 0
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    initial = perturb.infer(m, segments["train"], device)
    write_json(out / "initial_train_signal.json", signal(data["train"], initial))
    for epoch in range(CONFIG["epochs"]):
        unchanged(p)
        m.train(); beginning = time.monotonic()
        order = fixed["sampling"]["normal_orders"][epoch]
        batches = [order[i:i + 16] for i in range(0, len(order), 16)]
        anchors = np.array_split(fixed["sampling"]["anchor_orders"][epoch], len(batches))
        if any(len(a) == 0 for a in anchors):
            raise RuntimeError("common anchor-view sampling cannot supply every optimizer step")
        total, nvisit, avisit, gmax = 0., 0, 0, {}
        ce_total = sm_total = 0.
        for i, indices in enumerate(batches):
            batch = collate_segments([segments["train"][j] for j in indices])
            ab, positions = anchor_batch(segments["train"], fixed["train_replacements"], anchors[i], m.spec["augmented"])
            seed = 42 + epoch * len(batches) + i
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            opt.zero_grad(set_to_none=True)
            normal, nce, sm = loss_on_batch(m, batch, device)
            anchor, ace, asm = loss_on_batch(m, ab, device, positions)
            loss = .5 * (normal + anchor)
            if not bool(torch.isfinite(loss)) or float(asm.detach()) != 0:
                raise RuntimeError("training loss invalid or synthetic smoothing enabled")
            loss.backward()
            for name, value in gradient_norms(m).items():
                gmax[name] = max(gmax.get(name, 0), value)
            torch.nn.utils.clip_grad_norm_(m.parameters(), CONFIG["clip_gradient_norm"])
            opt.step()
            if any(not bool(torch.isfinite(w).all()) for w in m.parameters()):
                raise RuntimeError("nonfinite formal parameters")
            steps += 1; nvisit += int(batch[3].sum()); avisit += len(positions)
            total += float(loss.detach()); ce_total += float((.5 * (nce + ace)).detach()); sm_total += float(sm.detach())
        if nvisit != 7372 or avisit != len(fixed["train_replacements"]["rows"]) or not gmax or max(gmax.values()) <= 0:
            raise RuntimeError("common train coverage/gradients incomplete")
        z = perturb.infer(m, segments["val"], device)
        metrics = simple_metrics(data["val"], z)
        row = {"epoch": epoch + 1, "phase": "frozen_input_benchmark", "cumulative_steps": steps,
               "train_loss": total / len(batches), "weighted_stage_CE": ce_total / len(batches),
               "normal_smoothing": sm_total / len(batches), "normal_clip_visits": nvisit, "anchor_visits": avisit,
               "val_macro_f1": metrics["macro_f1"], "val_accuracy": metrics["accuracy"],
               "learning_rate": opt.param_groups[0]["lr"], "gradients_max": gmax, "seconds": time.monotonic() - beginning}
        if epoch + 1 in CONFIG["diagnostic_epochs"]:
            row["native_train_signal"] = signal(data["train"], perturb.infer(m, segments["train"], device))
            row["preserved_anchor_train_signal"] = anchor_signal(m, segments["train"], fixed["train_replacements"], m.spec["augmented"], device)
        history.append(row)
        if metrics["macro_f1"] > best + 1e-12:
            best = metrics["macro_f1"]
            _torch_save(out / "best.pt", {**checkpoint_identity(p, backbone, task, seal), "model": m.core.state_dict(),
                       "epoch": epoch + 1, "macro_f1": best, "accuracy": metrics["accuracy"]})
        atomic_json(out / "history.json", history)
        scheduler.step()
        print(json.dumps({"backbone": backbone, "variant": task, "epoch": epoch + 1,
                          "train_loss": row["train_loss"], "val_macro_f1": metrics["macro_f1"], "best": best}), flush=True)
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    write_json(out / "result.json", {**checkpoint_identity(p, backbone, task, seal), **m.parameter_counts(),
        "training_completed": True, "new_training": True, "test_read": False, "epochs_completed": len(history),
        "optimizer_steps": steps, "best_epoch": ck["epoch"], "best_macro_f1": ck["macro_f1"],
        "best_accuracy_same_F1_checkpoint": ck["accuracy"], "best_sha256": sha256_file(out / "best.pt"),
        "resources": {"elapsed_seconds": time.monotonic() - started,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}})


def validate_completed(p, backbone, task, device):
    out = Path(p["subplans"][backbone]["output"])
    _, _, seal = common.cache(out)
    ref = p["reuse"][backbone].get(task)
    if ref:
        r = json.loads(Path(ref["result"]).read_text())
        ck = torch.load(ref["checkpoint"], map_location="cpu", weights_only=True)
        old_seal = json.loads(Path(ref["audit_seal"]).read_text())
        if any(old_seal[k] != seal[k] for k in ("sampling_hash", "train_replacement_hash", "validation_challenge_hash")) or \
                ck.get("sampling_hash") != seal["sampling_hash"] or ck.get("train_replacement_hash") != seal["train_replacement_hash"]:
            raise RuntimeError("reused complete cell's actual private plans are not identical")
        result = {k: r[k] for k in ("epochs_completed", "optimizer_steps", "best_epoch", "best_macro_f1",
                  "best_accuracy_same_F1_checkpoint", "resources", "best_sha256")}
        result.update({"new_training": False, "training_completed": True, "original_source_commit": ref["source_commit"],
                       "reuse_replayed_and_read_only": True})
    else:
        directory = out / task
        r = json.loads((directory / "result.json").read_text())
        ck = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        history = json.loads((directory / "history.json").read_text())
        identity = checkpoint_identity(p, backbone, task, seal)
        expected_steps = CONFIG["epochs"] * math.ceil(p["subplans"][backbone]["evidence"]["segments"]["train"]["count"] / 16)
        if any(r.get(k) != v or ck.get(k) != v for k, v in identity.items()) or r.get("training_completed") is not True or r.get("test_read") is not False or \
                r.get("epochs_completed") != 60 or r.get("optimizer_steps") != expected_steps or \
                [row.get("epoch") for row in history] != list(range(1, 61)) or history[-1].get("cumulative_steps") != expected_steps or \
                ck.get("epoch") != r.get("best_epoch") or ck.get("macro_f1") != r.get("best_macro_f1") or \
                ck.get("accuracy") != r.get("best_accuracy_same_F1_checkpoint") or sha256_file(directory / "best.pt") != r.get("best_sha256"):
            raise RuntimeError("complete fixed-budget checkpoint/history/result seals are required")
        result = {k: r[k] for k in ("epochs_completed", "optimizer_steps", "best_epoch", "best_macro_f1",
                  "best_accuracy_same_F1_checkpoint", "resources", "best_sha256", "new_training", "training_completed")}
    m = model(p, backbone, task, device)
    m.core.load_state_dict(ck["model"], strict=True)
    return m, result


def evaluate(p, backbone, segments, metadata, fixed, data, seal, device):
    out = Path(p["subplans"][backbone]["output"])
    folder = out / "evaluation"; folder.mkdir(mode=0o700, exist_ok=False)
    y, a = data["val"]["labels"].numpy(), data["val"]["a"].argmax(-1).numpy()
    report = {"schema": CONFIG["schema"], "source_commit": p["source_commit"], "backbone": backbone,
        "training_completed": True, "counts": p["subplans"][backbone]["evidence"]["counts"], "test_read": False,
        "validation_role": "val823_repeated_development_not_independent_test", "A_frozen": True,
        "A": simple_metrics(data["val"], data["val"]["a"]), "historical_TCN": simple_metrics(data["val"], data["val"]["t"]),
        "training": CONFIG, "registry": TASKS, "models": {}, "patient_independence_verified": False,
        "cross_head_architecture_loss_capacity_and_compute_differ": True,
        "frozen_train_features_not_out_of_fold_even_with_scores_masked": True,
        "candidate_adjacency_not_manually_verified_continuity": True,
        "synthetic_interventions_not_natural_edit_boundaries": True}
    native = {}
    for task in TASKS:
        m, result = validate_completed(p, backbone, task, device)
        z = perturb.infer(m, segments["val"], device)
        metrics = simple_metrics(data["val"], z)
        if not math.isclose(metrics["macro_f1"], result["best_macro_f1"], abs_tol=1e-12, rel_tol=0) or \
                not math.isclose(metrics["accuracy"], result["best_accuracy_same_F1_checkpoint"], abs_tol=1e-12, rel_tol=0):
            raise RuntimeError("selected best-F1 checkpoint must replay both metrics, including reused results")
        native[task] = z
        ref = p["reuse"][backbone].get(task)
        directory = Path(ref["run"]) / ref["variant"] if ref else out / task
        history = json.loads((directory / "history.json").read_text())
        train_signals = [{"epoch": row["epoch"], **{k: row[k] for k in
            ("native_train_signal", "preserved_anchor_train_signal") if k in row}}
            for row in history if row["epoch"] in CONFIG["diagnostic_epochs"]]
        rows = [{"clip_id": clip, "label_id": int(y[i]), "a_prediction": int(a[i]), "tcn_prediction": int(z[i].argmax())}
                for i, clip in enumerate(data["val"]["clip_ids"])]
        with (folder / (task + "_private_predictions.jsonl")).open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        aggregate = summarize(rows, metadata, segments["val"])
        aggregate = json.loads(json.dumps(aggregate).replace('"VideoMAE_only"', '"A_only"').replace('"VideoMAE_TCN"', '"temporal_model"'))
        flat = [Segment([c], s.features[j:j + 1], s.a_logits[j:j + 1], s.labels[j:j + 1], [s.source_kinds[j]])
                for s in segments["val"] for j, c in enumerate(s.clip_ids)]
        centered = perturb.infer(m, flat, device)
        interventions = {"no_neighbor": {"overall_metrics": simple_metrics(data["val"], centered),
                         **eligible_audit(data["val"], z, centered, list(range(len(z))))}}
        for name in ("shuffle", "cross_clinical"):
            changed, eligible = perturb.intervene(m, segments["val"], z, fixed["validation"][name], device)
            interventions[name] = {"overall_metrics": simple_metrics(data["val"], changed),
                **eligible_audit(data["val"], z, changed, eligible), "maximum_logit_difference": float((changed - z).abs().max()),
                "changed_prediction_count": int((changed.argmax(-1) != z.argmax(-1)).sum())}
        report["models"][task] = {"training_result": result, "parameters": m.parameter_counts(), "metrics": metrics,
            "train_signal_at_registered_epochs": train_signals,
            "aggregate": aggregate, "interventions": interventions,
            "singleton_A_protection_diagnostic_only": simple_metrics(data["val"], perturb.singleton_fallback(z, data["val"]["a"], segments["val"])),
            "paired_vs_historical_TCN": _paired(y, z.argmax(-1).numpy(), data["val"]["t"].argmax(-1).numpy()),
            "historical_correction_retention": retention(data["val"]["labels"], data["val"]["a"], data["val"]["t"], z)}
    for task, entry in report["models"].items():
        control = "visual_aug" if TASKS[task]["scores"] == "zero" and TASKS[task]["augmented"] else \
                  "logits_aug" if TASKS[task]["augmented"] else "visual" if TASKS[task]["scores"] == "zero" else "logits"
        entry["paired_vs_matched_input_native_TCN"] = _paired(y, native[task].argmax(-1).numpy(), native[control].argmax(-1).numpy())
        entry["per_class_vs_matched_input_native_TCN"] = [
            {"label_id": c["label_id"], "support": c["support"], "f1_change": c["f1"] - b["f1"],
             "macro_f1_contribution": (c["f1"] - b["f1"]) / 7}
            for c, b in zip(entry["metrics"]["per_class"], report["models"][control]["metrics"]["per_class"])]
        entry["recording_cluster_interval_vs_native_TCN"] = recording_interval(y, native[task].argmax(-1).numpy(),
            native[control].argmax(-1).numpy(), data["val"]["clip_ids"], metadata)
    report["native_input_augmentation_contrasts"] = factorial_contrasts({v: report["models"][v]["metrics"]["macro_f1"] for v in ("logits", "logits_aug", "visual", "visual_aug")})
    report["combined_multistage_supervision_smoothing_ablation_F1"] = {
        head: report["models"][head + "_logits"]["metrics"]["macro_f1"] - report["models"][head + "_logits_final_ce"]["metrics"]["macro_f1"]
        for head in ("mstcnpp", "asformer")}
    write_json(folder / "public_safe_summary.json", public_safe(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-protocol", required=True)
    parser.add_argument("--stage", choices=("audit", "smoke", "train", "evaluate", "aggregate"), required=True)
    parser.add_argument("--backbone", required=True); parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    p = json.loads(Path(args.run_protocol).read_text()); unchanged(p)
    torch.set_num_threads(2)
    if args.stage == "aggregate":
        from .report import aggregate
        aggregate(p); return
    child = p["subplans"][args.backbone]; out = Path(child["output"])
    if args.stage == "audit":
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"):
            raise RuntimeError("each independent worker requires exactly one UUID-mapped CUDA device")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    segments, metadata, info = common.evidence(child["backbone"], child["index"], child["source_protocol_dir"], child["historical_run"])
    if info != child["evidence"]:
        raise RuntimeError("sealed full7372/823 evidence changed")
    if args.stage == "audit":
        common.audit(child, out, segments, metadata)
        # Validate actual generated plans before ANY later CUDA/training job.
        _, audited, _ = common.cache(out)
        replays = {}
        for task in p["reuse"][args.backbone]:
            m, r = validate_completed(p, args.backbone, task, device)
            metrics = simple_metrics(audited["val"], perturb.infer(m, segments["val"], device))
            if not math.isclose(metrics["macro_f1"], r["best_macro_f1"], abs_tol=1e-12, rel_tol=0) or \
                    not math.isclose(metrics["accuracy"], r["best_accuracy_same_F1_checkpoint"], abs_tol=1e-12, rel_tol=0):
                raise RuntimeError("reused native checkpoint does not replay its historical best metrics")
            replays[task] = {"selected_checkpoint_replay": True, "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"]}
        write_json(out / "reuse_replay.json", replays)
        return
    fixed, data, seal = common.cache(out)
    if args.stage == "smoke":
        smoke(p, args.backbone, args.variant, segments, fixed, device)
    elif args.stage == "train":
        train(p, args.backbone, args.variant, segments, fixed, data, seal, device)
    else:
        evaluate(p, args.backbone, segments, metadata, fixed, data, seal, device)


if __name__ == "__main__":
    main()
