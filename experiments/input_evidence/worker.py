"""Sealed input diagnostic stages. All identities, evidence and weights stay private."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.anchor_reference import worker as shared_worker
from experiments.anchor_reference.data import anchor_batch
from experiments.direction_gate.worker import gradient_norms, retention, simple_metrics
from experiments.paper_validation import perturb
from experiments.relation.data import sha256_file
from experiments.relation.evaluate_suite import _paired
from experiments.relation.tcn_baseline import Segment, TCNConfig, collate_segments
from experiments.videomae_tcn.io import atomic_json, write_json
from experiments.videomae_tcn.metrics import summarize
from experiments.videomae_tcn.worker import _torch_save
from .model import AUGMENTED, MASKED, InputEvidenceTCN, input_policy
from .run_suite import CONFIG, unchanged

infer = shared_worker.infer
loss_on_batch = shared_worker.loss_on_batch
check_signal = shared_worker.check_signal
cache = shared_worker.cache


def model(p, variant, device):
    return InputEvidenceTCN(TCNConfig(**p["evidence"]["model_config"]), variant,
                            seed=CONFIG["seed"]).to(device)


def public_safe(value):
    shared_worker.public_safe(value)
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if any(key in raw for key in ('"clip_ids"', '"patient_id"', '"video_path"', '"source_collection"')):
        raise RuntimeError("private evidence escaped the aggregate-only report")
    return value


def score_statistics(scores):
    if scores.ndim != 2 or scores.shape[1] != 7 or not len(scores) or not bool(torch.isfinite(scores).all()):
        raise ValueError("require nonempty finite seven-class scores")
    logp = scores.log_softmax(-1)
    probabilities = logp.exp()
    top = probabilities.topk(2, dim=-1).values
    return {"mean_max_probability": float(top[:, 0].mean()),
            "mean_probability_margin": float((top[:, 0] - top[:, 1]).mean()),
            "mean_entropy": float(-(probabilities * logp).sum(-1).mean())}


def signal(data, scores):
    return {**check_signal(data, scores), "model_scores": score_statistics(scores),
            "A_scores": score_statistics(data["a"])}


def native_oracle(m, batch, device):
    features, scores, _, valid, _ = batch
    features, scores = m.effective_inputs(features.to(device), scores.to(device))
    valid = valid.to(device)
    hidden = m.base.stem(features).transpose(1, 2) * valid[:, None]
    for block in m.base.blocks:
        hidden = block(hidden, valid)
    return m.base.classifier(hidden).transpose(1, 2) * valid[..., None]


@torch.no_grad()
def anchor_signal(m, segments, replacements, augmented, device):
    """Only evaluate preserved anchors; donor labels are never scored."""
    m.eval()
    predictions, labels, scores = [], [], []
    for start in range(0, len(replacements["rows"]), 16):
        indices = range(start, min(start + 16, len(replacements["rows"])))
        batch, positions = anchor_batch(segments, replacements, indices, augmented)
        features, a, y, valid, _ = batch
        z = m(features.to(device), a.to(device), valid.to(device)).cpu()
        rows = torch.arange(len(positions))
        positions = torch.tensor(positions)
        predictions.append(z[rows, positions])
        labels.append(y[rows, positions])
        scores.append(a[rows, positions])
    return signal({"labels": torch.cat(labels), "a": torch.cat(scores)}, torch.cat(predictions))


def audit(p, out, segments, metadata):
    # The shared auditor strictly replays the genuine historical checkpoint,
    # seals 7372/823 evidence and creates one train-only replacement schedule.
    shared_worker.audit(p, out, segments, metadata)
    _, data, _ = cache(out)
    write_json(out / "input_diagnostic_summary.json", {
        "schema": CONFIG["schema"], "historical_823_prediction_replay_exact": True,
        "test_read": False, "train_A_scores": score_statistics(data["train"]["a"]),
        "val_A_scores": score_statistics(data["val"]["a"]),
        "raw_evidence_not_modified": True, "masking_is_inside_every_model_forward": True,
        "not_out_of_fold_even_when_A_logits_are_masked": True})


def smoke(p, out, segments, fixed, variant, device):
    if device.type != "cuda":
        raise RuntimeError("real smoke requires CUDA, never label a CPU check as CUDA")
    folder = out / "smoke"
    folder.mkdir(mode=0o700, exist_ok=True)
    real_segments = [s for s in segments["train"] if len(s.clip_ids) > 1][:4]
    if len(real_segments) != 4:
        raise RuntimeError("need four real multi-clip train segments")
    batch = collate_segments(real_segments)
    m = model(p, variant, device).eval()
    with torch.no_grad():
        before = m(batch[0].to(device), batch[1].to(device), batch[3].to(device))
        oracle = native_oracle(m, batch, device)
        torch.testing.assert_close(before, oracle, atol=2e-5, rtol=2e-5)
        altered = batch[0].clone()
        altered[..., -7:] += 3.0
        response = m(altered.to(device), altered[..., -7:].to(device), batch[3].to(device))
        if variant in MASKED:
            torch.testing.assert_close(before, response, atol=0, rtol=0)
        elif float((response - before).abs().max()) == 0:
            raise RuntimeError("raw-logit control is unexpectedly insensitive to its seven inputs")
    original_features, original_scores = batch[0].clone(), batch[1].clone()
    augmented, positions = anchor_batch(segments["train"], fixed["train_replacements"], range(4), variant in AUGMENTED)
    optimizer = torch.optim.AdamW(m.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    records = []
    torch.cuda.reset_peak_memory_stats(device)
    m.train()
    for step in range(3):
        torch.manual_seed(42 + step)
        torch.cuda.manual_seed_all(42 + step)
        weights = {n: w.detach().clone() for n, w in m.named_parameters()}
        optimizer.zero_grad(set_to_none=True)
        loss = .5 * (loss_on_batch(m, batch, device) + loss_on_batch(m, augmented, device, positions))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite real-feature smoke loss")
        loss.backward()
        gradients = gradient_norms(m)
        for name in ("base.stem.0.weight", "base.classifier.weight",
                     *(f"base.blocks.{i}.conv.weight" for i in range(3))):
            if gradients.get(name, 0) <= 0:
                raise RuntimeError("TCN stem, temporal convolutions and classifier require finite nonzero gradients")
        if variant in MASKED and bool((m.base.stem[0].weight.grad[:, -7:] != 0).any()):
            raise RuntimeError("masked A inputs leaked into the stem gradient")
        torch.nn.utils.clip_grad_norm_(m.parameters(), CONFIG["clip_gradient_norm"])
        optimizer.step()
        updates = {n: float((w.detach() - weights[n]).abs().max()) for n, w in m.named_parameters()}
        if updates["base.classifier.weight"] <= 0 or updates["base.stem.0.weight"] <= 0 or \
                any(not bool(torch.isfinite(w).all()) for w in m.parameters()):
            raise RuntimeError("finite real parameter updates are required")
        records.append({"step": step + 1, "loss": float(loss.detach()), "gradients": gradients, "updates": updates})
    if not torch.equal(batch[0], original_features) or not torch.equal(batch[1], original_scores):
        raise RuntimeError("input masking modified shared frozen evidence in place")
    m.eval()
    singles = [Segment([s.clip_ids[0]], s.features[:1], s.a_logits[:1], s.labels[:1], s.source_kinds[:1])
               for s in real_segments]
    single_batch = collate_segments(singles)
    with torch.no_grad():
        torch.testing.assert_close(m(single_batch[0].to(device), single_batch[1].to(device), single_batch[3].to(device)),
                                   native_oracle(m, single_batch, device), atol=2e-5, rtol=2e-5)
    write_json(folder / (variant + ".json"), {
        "all_passed": True, "variant": variant, "input_policy": input_policy(variant), "steps": records,
        "raw_evidence_unchanged": True, "A_parameters_trained": 0,
        "mask_checked_in_forward_and_gradient": variant in MASKED,
        "native_plain_same_weight_replay": True, "singleton_not_forced_A": True,
        "torch": str(torch.__version__), "CUDA": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device), "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})


def train(p, out, segments, fixed, data, seal, variant, device):
    folder = out / variant
    folder.mkdir(mode=0o700, exist_ok=False)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    m = model(p, variant, device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"], eta_min=CONFIG["minimum_learning_rate"])
    history, best, steps = [], -math.inf, 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    initial, _, _ = infer(m, segments["train"], device)
    write_json(folder / "initial_train_signal.json", signal(data["train"], initial))
    for epoch in range(CONFIG["epochs"]):
        unchanged(p)
        m.train()
        epoch_start = time.monotonic()
        order = fixed["sampling"]["normal_orders"][epoch]
        batches = [order[k:k + 16] for k in range(0, len(order), 16)]
        anchors = np.array_split(fixed["sampling"]["anchor_orders"][epoch], len(batches))
        if any(len(a) == 0 for a in anchors):
            raise RuntimeError("insufficient preserved-anchor views for the common sampling schedule")
        normal_sum = anchor_sum = 0.0
        normal_count = anchor_count = 0
        grad_max = {}
        for b, indices in enumerate(batches):
            batch = collate_segments([segments["train"][i] for i in indices])
            ab, positions = anchor_batch(segments["train"], fixed["train_replacements"], anchors[b], variant in AUGMENTED)
            step_seed = 42 + epoch * len(batches) + b
            torch.manual_seed(step_seed)
            torch.cuda.manual_seed_all(step_seed)
            optimizer.zero_grad(set_to_none=True)
            normal = loss_on_batch(m, batch, device)
            anchor = loss_on_batch(m, ab, device, positions)
            loss = .5 * (normal + anchor)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite formal classification loss")
            loss.backward()
            for name, grad in gradient_norms(m).items():
                grad_max[name] = max(grad_max.get(name, 0), grad)
            if variant in MASKED and bool((m.base.stem[0].weight.grad[:, -7:] != 0).any()):
                raise RuntimeError("masked input leaked in formal training")
            torch.nn.utils.clip_grad_norm_(m.parameters(), CONFIG["clip_gradient_norm"])
            optimizer.step()
            if any(not bool(torch.isfinite(w).all()) for w in m.parameters()):
                raise RuntimeError("nonfinite trained weights")
            steps += 1
            nv, av = int(batch[3].sum()), len(positions)
            normal_sum += float(normal.detach()) * nv
            anchor_sum += float(anchor.detach()) * av
            normal_count += nv
            anchor_count += av
        if normal_count != 7372 or anchor_count != len(fixed["train_replacements"]["rows"]) or \
                not grad_max or max(grad_max.values()) <= 0:
            raise RuntimeError("training coverage/finite gradients are incomplete")
        z, _, _ = infer(m, segments["val"], device)
        metrics = simple_metrics(data["val"], z)
        row = {"epoch": epoch + 1, "phase": "native_input_diagnostic", "cumulative_steps": steps,
               "input_policy": input_policy(variant), "normal_CE": normal_sum / normal_count,
               "anchor_CE": anchor_sum / anchor_count,
               "train_loss": .5 * (normal_sum / normal_count + anchor_sum / anchor_count),
               "normal_clip_visits": normal_count, "anchor_visits": anchor_count,
               "val_macro_f1": metrics["macro_f1"], "val_accuracy": metrics["accuracy"],
               "learning_rate": optimizer.param_groups[0]["lr"], "gradients_max": grad_max,
               "seconds": time.monotonic() - epoch_start}
        if epoch + 1 in CONFIG["diagnostic_epochs"]:
            t, _, _ = infer(m, segments["train"], device)
            row["native_train_signal"] = signal(data["train"], t)
            row["preserved_anchor_train_signal"] = anchor_signal(
                m, segments["train"], fixed["train_replacements"], variant in AUGMENTED, device)
        history.append(row)
        if metrics["macro_f1"] > best + 1e-12:
            best = metrics["macro_f1"]
            _torch_save(folder / "best.pt", {
                "schema": CONFIG["schema"], "source_commit": p["source_commit"], "variant": variant,
                "backbone": p["backbone"], "input_policy": input_policy(variant), "model": m.state_dict(),
                "epoch": epoch + 1, "macro_f1": best, "accuracy": metrics["accuracy"],
                "input_sha256": p["evidence"]["input_sha256"], "training": p["training"],
                "sampling_hash": seal["sampling_hash"], "train_replacement_hash": seal["train_replacement_hash"]})
        atomic_json(folder / "history.json", history)
        scheduler.step()
        print(json.dumps({k: row[k] for k in ("epoch", "train_loss", "val_macro_f1", "val_accuracy")}
                         | {"variant": variant, "best": best}), flush=True)
    ck = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    write_json(folder / "result.json", {
        "training_completed": True, "test_read": False, "variant": variant,
        "epochs_completed": len(history), "optimizer_steps": steps, "best_epoch": ck["epoch"],
        "best_macro_f1": ck["macro_f1"], "best_accuracy_same_F1_checkpoint": ck["accuracy"],
        "best_sha256": sha256_file(folder / "best.pt"), "fresh_native_classifier": True,
        "historical_checkpoint_used_as_initialization": False, "seed": 42, **m.parameter_counts(),
        "sampling_hash": seal["sampling_hash"], "train_replacement_hash": seal["train_replacement_hash"],
        "resources": {"elapsed_seconds": time.monotonic() - started,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)},
        "receptive_field_clips": 15})


def load_completed(p, out, segments, seal, variant, device):
    r = json.loads((out / variant / "result.json").read_text())
    history = json.loads((out / variant / "history.json").read_text())
    ck = torch.load(out / variant / "best.pt", map_location="cpu", weights_only=True)
    expected_steps = CONFIG["epochs"] * math.ceil(len(segments["train"]) / CONFIG["batch_size_segments"])
    expected = {"schema": CONFIG["schema"], "source_commit": p["source_commit"], "variant": variant,
                "backbone": p["backbone"], "input_policy": input_policy(variant),
                "input_sha256": p["evidence"]["input_sha256"], "training": p["training"],
                "sampling_hash": seal["sampling_hash"], "train_replacement_hash": seal["train_replacement_hash"]}
    if any(ck.get(k) != value for k, value in expected.items()) or r.get("training_completed") is not True or \
            r.get("variant") != variant or r.get("input_policy") != input_policy(variant) or \
            r.get("epochs_completed") != CONFIG["epochs"] or r.get("optimizer_steps") != expected_steps or \
            len(history) != CONFIG["epochs"] or [row.get("epoch") for row in history] != list(range(1, CONFIG["epochs"] + 1)) or \
            history[-1].get("cumulative_steps") != expected_steps or \
            sha256_file(out / variant / "best.pt") != r.get("best_sha256") or \
            ck.get("epoch") != r.get("best_epoch") or ck.get("macro_f1") != r.get("best_macro_f1") or \
            ck.get("accuracy") != r.get("best_accuracy_same_F1_checkpoint"):
        raise RuntimeError("complete checkpoint and input-policy seals do not match")
    m = model(p, variant, device)
    m.load_state_dict(ck["model"], strict=True)
    return m, ck, r


def factorial_contrasts(f1):
    if set(f1) != set(CONFIG["variants"]) or any(not math.isfinite(x) for x in f1.values()):
        raise ValueError("factorial contrast requires all four finite cell results")
    return {"mask_without_augmentation": f1["visual"] - f1["logits"],
            "mask_with_augmentation": f1["visual_aug"] - f1["logits_aug"],
            "augmentation_with_logits": f1["logits_aug"] - f1["logits"],
            "augmentation_without_logits": f1["visual_aug"] - f1["visual"],
            "mask_by_augmentation_interaction": (f1["visual_aug"] - f1["visual"]) - (f1["logits_aug"] - f1["logits"]),
            "descriptive_single_seed_not_statistical_significance": True}


def eligible_audit(data, native, changed, eligible):
    if len(set(eligible)) != len(eligible) or any(i < 0 or i >= len(native) for i in eligible):
        raise ValueError("intervention cohort must contain unique valid anchors")
    if not eligible:
        return {"eligible_count": 0, "native_eligible_metrics": None, "eligible_metrics": None,
                "paired_eligible_vs_native": None, "eligible_macro_f1_change": None}
    subset = {"labels": data["labels"][eligible], "a": data["a"][eligible]}
    before = simple_metrics(subset, native[eligible])
    after = simple_metrics(subset, changed[eligible])
    paired = _paired(subset["labels"].numpy(), changed[eligible].argmax(-1).numpy(), native[eligible].argmax(-1).numpy())
    return {"eligible_count": len(eligible), "native_eligible_metrics": before, "eligible_metrics": after,
            "paired_eligible_vs_native": paired, "eligible_macro_f1_change": after["macro_f1"] - before["macro_f1"]}


def evaluate(p, out, segments, metadata, fixed, data, seal, device):
    folder = out / "evaluation"
    folder.mkdir(mode=0o700, exist_ok=False)
    report = {"schema": CONFIG["schema"], "source_commit": p["source_commit"], "backbone": p["backbone"],
              "training_completed": True, "counts": p["evidence"]["counts"], "training": p["training"],
              "validation_role": p["validation_role"], "test_read": False, "A_frozen": True,
              "A": simple_metrics(data["val"], data["val"]["a"]),
              "historical_TCN": simple_metrics(data["val"], data["val"]["t"]), "models": {},
              "historical_results_reference_only_not_matched_budget": True,
              "masked_visual_features_still_from_in_sample_A": True, "exploratory_not_confirmatory": True,
              "sampling_hash": seal["sampling_hash"], "train_replacement_hash": seal["train_replacement_hash"],
              "synthetic_interventions_not_verified_edit_boundaries": True}
    y = data["val"]["labels"].numpy()
    a = data["val"]["a"].argmax(-1).numpy()
    native = {}
    for variant in p["variants"]:
        m, ck, result = load_completed(p, out, segments, seal, variant, device)
        z, _, _ = infer(m, segments["val"], device)
        metrics = simple_metrics(data["val"], z)
        if not math.isclose(metrics["macro_f1"], ck["macro_f1"], abs_tol=1e-12) or \
                not math.isclose(metrics["accuracy"], ck["accuracy"], abs_tol=1e-12):
            raise RuntimeError("best-F1 checkpoint does not replay its selected metrics")
        native[variant] = z
        rows = [{"clip_id": clip, "label_id": int(y[i]), "a_prediction": int(a[i]),
                 "tcn_prediction": int(z[i].argmax())} for i, clip in enumerate(data["val"]["clip_ids"])]
        with (folder / (variant + "_private_predictions.jsonl")).open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        aggregate = summarize(rows, metadata, segments["val"])
        aggregate = json.loads(json.dumps(aggregate).replace('"VideoMAE_only"', '"A_only"')
                               .replace('"VideoMAE_TCN"', '"temporal_model"'))
        centered, _, _ = infer(m, segments["val"], device, mode="zero")
        flat = [Segment([c], s.features[j:j + 1], s.a_logits[j:j + 1], s.labels[j:j + 1], [s.source_kinds[j]])
                for s in segments["val"] for j, c in enumerate(s.clip_ids)]
        torch.testing.assert_close(centered, perturb.infer(m, flat, device), atol=2e-5, rtol=2e-5)
        interventions = {"no_neighbor": {"overall_metrics": simple_metrics(data["val"], centered),
                         **eligible_audit(data["val"], z, centered, list(range(len(z)))),
                         "maximum_logit_difference": float((centered - z).abs().max())}}
        for name in ("shuffle", "cross_clinical"):
            changed, eligible = perturb.intervene(m, segments["val"], z, fixed["validation"][name], device)
            interventions[name] = {"overall_metrics": simple_metrics(data["val"], changed),
                                  **eligible_audit(data["val"], z, changed, eligible),
                                  "maximum_logit_difference": float((changed - z).abs().max()),
                                  "changed_prediction_count": int((changed.argmax(-1) != z.argmax(-1)).sum())}
        history = json.loads((out / variant / "history.json").read_text())
        report["models"][variant] = {
            "training_result": result, "metrics": metrics, "aggregate": aggregate, "interventions": interventions,
            "singleton_A_protection_diagnostic_only": simple_metrics(
                data["val"], perturb.singleton_fallback(z, data["val"]["a"], segments["val"])),
            "paired_vs_historical_TCN": _paired(y, z.argmax(-1).numpy(), data["val"]["t"].argmax(-1).numpy()),
            "historical_correction_retention": retention(data["val"]["labels"], data["val"]["a"], data["val"]["t"], z),
            "training_signal_at_registered_epochs": [{k: row[k] for k in
                ("epoch", "normal_CE", "anchor_CE", "native_train_signal", "preserved_anchor_train_signal")}
                for row in history if "native_train_signal" in row]}
    for variant, entry in report["models"].items():
        entry["paired_vs_matched_cells"] = {control: _paired(y, native[variant].argmax(-1).numpy(),
                                                            native[control].argmax(-1).numpy())
                                            for control in p["variants"] if control != variant}
        entry["per_class_vs_matched_cells"] = {
            control: [{"label_id": current["label_id"], "support": current["support"],
                       "f1_change": current["f1"] - baseline["f1"],
                       "macro_f1_contribution": (current["f1"] - baseline["f1"]) / 7}
                      for current, baseline in zip(entry["metrics"]["per_class"], report["models"][control]["metrics"]["per_class"])]
            for control in p["variants"] if control != variant}
    f1 = {v: entry["metrics"]["macro_f1"] for v, entry in report["models"].items()}
    report["factorial_F1_contrasts"] = factorial_contrasts(f1)
    write_json(folder / "public_safe_summary.json", public_safe(report))
    print(json.dumps(f1), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "smoke", "train", "evaluate"), required=True)
    parser.add_argument("--run-protocol", required=True)
    parser.add_argument("--variant")
    args = parser.parse_args()
    p = json.loads(Path(args.run_protocol).read_text())
    unchanged(p)
    torch.set_num_threads(2)
    if args.stage == "audit":
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or \
                not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"):
            raise RuntimeError("each worker requires exactly one CUDA GPU mapped by UUID; never DDP")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    out = Path(p["output"])
    segments, metadata, info = shared_worker.evidence(
        p["backbone"], p["index"], p["source_protocol_dir"], p["historical_run"])
    if info != p["evidence"]:
        raise RuntimeError("sealed full-split frozen evidence changed")
    if args.stage == "audit":
        audit(p, out, segments, metadata)
        return
    fixed, data, seal = cache(out)
    if args.stage in ("smoke", "train") and args.variant not in p["variants"]:
        raise ValueError("explicit registered input cell is required")
    if args.stage == "smoke":
        smoke(p, out, segments, fixed, args.variant, device)
    elif args.stage == "train":
        train(p, out, segments, fixed, data, seal, args.variant, device)
    else:
        evaluate(p, out, segments, metadata, fixed, data, seal, device)


if __name__ == "__main__":
    main()
