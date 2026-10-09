"""Strict base replay, private evidence diagnostics, disposable smoke and gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.relation.data import read_jsonl, sha256_file
from experiments.relation.tcn_baseline import FrozenEvidenceTCN, TCNConfig, Segment, collate_segments
from experiments.relation.evaluate_suite import classification_metrics, _paired, _directions
from experiments.videomae_tcn.evidence import prepare
from experiments.videomae_tcn.io import atomic_json, write_json
from experiments.videomae_tcn.metrics import summarize
from experiments.videomae_tcn.worker import _torch_save
from .model import AcceptanceGate, VARIANTS, Q_NAMES, train_statistics
from .run import SCHEMA, TRAINING, unchanged


def base_model(p, info, device):
    checkpoint = torch.load(Path(p["base_run"]) / "tcn/best.pt", map_location="cpu", weights_only=True)
    expected = {"schema": "fsn-videomae-plain-tcn-v1", "config": info["model_config"],
                "input_sha256": info["input_sha256"], "A_checkpoint_sha256": info["A_checkpoint_sha256"],
                "epoch": p["base_result"]["best_epoch"],
                "val_macro_f1": p["base_result"]["best_val_macro_f1"]}
    if any(checkpoint.get(k) != v for k, v in expected.items()):
        raise ValueError("ordinary TCN checkpoint disagrees with sealed evidence")
    m = FrozenEvidenceTCN(TCNConfig(**info["model_config"])).to(device)
    m.load_state_dict(checkpoint["model"], strict=True)
    m.requires_grad_(False).eval()
    return m


@torch.no_grad()
def native_outputs(model, segments, device):
    a, t, y, ids = [], [], [], []
    for start in range(0, len(segments), 16):
        batch = collate_segments(segments[start:start + 16])
        out = model(batch[0].to(device), batch[1].to(device), batch[3].to(device)).cpu()
        if not bool(torch.isfinite(out).all()):
            raise RuntimeError("nonfinite frozen TCN output")
        for row, seg in enumerate(batch[4]):
            n = len(seg.clip_ids)
            a.append(batch[1][row, :n]); t.append(out[row, :n]); y.append(batch[2][row, :n])
            ids.extend(seg.clip_ids)
    return {"a": torch.cat(a), "t": torch.cat(t), "labels": torch.cat(y), "clip_ids": ids}


def replacement_plan(segments, metadata, split, seed=42):
    """No labels in sampling. A donor is from this split, never the anchor record."""
    by_id = {r["clip_id"]: r for r in metadata}
    pool = [(si, j, clip) for si, seg in enumerate(segments) for j, clip in enumerate(seg.clip_ids)
            if by_id[clip]["split"] == split and by_id[clip]["source_kind"] == "clinical"]
    rng, replacements, excluded = np.random.default_rng(seed), [], {}
    for si, seg in enumerate(segments):
        for j, clip in enumerate(seg.clip_ids):
            anchor = by_id[clip]
            if anchor["source_kind"] != "clinical":
                continue
            if len(seg.clip_ids) == 1:
                excluded["clinical_singleton"] = excluded.get("clinical_singleton", 0) + 1
                continue
            eligible = [v for v in pool if by_id[v[2]]["recording_key"] != anchor["recording_key"]]
            same = [v for v in eligible if by_id[v[2]].get("source_collection") == anchor.get("source_collection")]
            choices = same or eligible
            if not choices:
                excluded["no_different_recording_donor"] = excluded.get("no_different_recording_donor", 0) + 1
                continue
            ds, dj, donor = choices[int(rng.integers(len(choices)))]
            neighbors = [k for k in (j - 1, j + 1) if 0 <= k < len(seg.clip_ids)]
            replaced = neighbors[int(rng.integers(len(neighbors)))]
            replacements.append({"segment": si, "anchor_position": j, "anchor_clip_id": clip,
                                 "neighbor_position": replaced, "donor_segment": ds,
                                 "donor_position": dj, "donor_clip_id": donor,
                                 "same_collection": bool(same)})
    return {"split": split, "seed": seed, "replacements": replacements, "excluded": excluded}


def replacement_segment(segments, row):
    original, donor = segments[row["segment"]], segments[row["donor_segment"]]
    pos, dp = row["neighbor_position"], row["donor_position"]
    if pos == row["anchor_position"] or original.clip_ids[row["anchor_position"]] != row["anchor_clip_id"] \
            or donor.clip_ids[dp] != row["donor_clip_id"]:
        raise ValueError("replacement plan anchor/donor mismatch")
    features, a = original.features.copy(), original.a_logits.copy()
    features[pos], a[pos] = donor.features[dp], donor.a_logits[dp]
    # labels are never used for donor/replaced positions; only unchanged anchor is assessed.
    return Segment(list(original.clip_ids), features, a, original.labels.copy(), list(original.source_kinds))


@torch.no_grad()
def challenge_outputs(model, segments, native, replacement, device):
    result = native["t"].clone()
    lookup = {clip: i for i, clip in enumerate(native["clip_ids"])}
    for row in replacement["replacements"]:
        seg = replacement_segment(segments, row)
        batch = collate_segments([seg])
        out = model(batch[0].to(device), batch[1].to(device), batch[3].to(device)).cpu()
        value = out[0, row["anchor_position"]]
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("nonfinite synthetic challenge output")
        result[lookup[row["anchor_clip_id"]]] = value
    return result


def simple_metrics(data, logits):
    y, pred, a = data["labels"].numpy(), logits.argmax(-1).cpu().numpy(), data["a"].argmax(-1).numpy()
    return {**classification_metrics(y, pred), "paired_vs_A": _paired(y, pred, a),
            "sweep_reperfusion": _directions(y, pred)}


def diagnose(p, out, device):
    segments, metadata, info = prepare(p["index"], p["source_protocol_dir"])
    if info != p["evidence"]:
        raise RuntimeError("sealed evidence changed")
    model = base_model(p, info, device)
    evidence, challenge_plans, diagnostic = {}, {}, {}
    for split in ("train", "val"):
        data = native_outputs(model, segments[split], device)
        if split == "val":
            old = read_jsonl(Path(p["base_run"]) / "evaluation/private_val_predictions.jsonl")
            lookup = {r["clip_id"]: r for r in old}
            if len(lookup) != len(old) or set(lookup) != set(data["clip_ids"]):
                raise RuntimeError("historical val predictions do not cover identical clips")
            for i, clip in enumerate(data["clip_ids"]):
                expected = lookup[clip]
                actual = {"label_id": int(data["labels"][i]), "a_prediction": int(data["a"][i].argmax()),
                          "tcn_prediction": int(data["t"][i].argmax())}
                if any(expected[k] != v for k, v in actual.items()):
                    raise RuntimeError("ordinary TCN replay differs from historical per-clip predictions")
            mf1 = simple_metrics(data, data["t"])["macro_f1"]
            if not math.isclose(mf1, p["base_result"]["best_val_macro_f1"], rel_tol=0, abs_tol=1e-12):
                raise RuntimeError("ordinary TCN replay differs from historical best-F1")
        challenge = replacement_plan(segments[split], metadata, split)
        data["challenge_t"] = challenge_outputs(model, segments[split], data, challenge, device)
        challenge_plans[split] = challenge
        statistics = train_statistics(data["a"], data["t"])
        diagnostic[split] = {"count": len(data["labels"]), "A": simple_metrics(data, data["a"]),
                             "TCN": simple_metrics(data, data["t"]),
                             "synthetic_cross_record_TCN": simple_metrics(data, data["challenge_t"]),
                             "prediction_pair_counts": statistics["pair_counts"],
                             "perturbed_anchor_count": len(challenge["replacements"]),
                             "replacement_excluded": challenge["excluded"],
                             "same_collection_donor_count": sum(r["same_collection"] for r in challenge["replacements"])}
        evidence[split] = data
    statistics = train_statistics(evidence["train"]["a"], evidence["train"]["t"])
    generator = torch.Generator().manual_seed(TRAINING["seed"])
    evidence["orders"] = torch.stack([torch.randperm(info["counts"]["train"], generator=generator)
                                      for _ in range(TRAINING["epochs"])])
    evidence["statistics"] = statistics
    _torch_save(out / "private_gate_evidence.pt", evidence)
    write_json(out / "private_replacement_plans.json", challenge_plans)
    write_json(out / "evidence_cache_seal.json", {
        "cache_sha256": sha256_file(out / "private_gate_evidence.pt"),
        "replacement_plan_sha256": sha256_file(out / "private_replacement_plans.json"),
        "sampling_order_sha256": hashlib.sha256(evidence["orders"].numpy().tobytes()).hexdigest()})
    write_json(out / "diagnostic_summary.json", {"historical_val_per_clip_replay_passed": True,
               "A_and_TCN_frozen": True, "test_read": False, "splits": diagnostic,
               "train_only_statistics": statistics, "challenge_is_synthetic": True,
               "train_gate_sees_native_evidence_only": True})
    for split in diagnostic:
        print(json.dumps({"stage": "diagnose", "split": split, "count": diagnostic[split]["count"],
                          "paired_vs_A": diagnostic[split]["TCN"]["paired_vs_A"],
                          "perturbed_anchors": diagnostic[split]["perturbed_anchor_count"]}), flush=True)


def cache(out):
    seal = json.loads((out / "evidence_cache_seal.json").read_text())
    if sha256_file(out / "private_gate_evidence.pt") != seal["cache_sha256"] or \
            sha256_file(out / "private_replacement_plans.json") != seal["replacement_plan_sha256"]:
        raise RuntimeError("private evidence or challenge plan changed")
    data = torch.load(out / "private_gate_evidence.pt", map_location="cpu", weights_only=True)
    if hashlib.sha256(data["orders"].numpy().tobytes()).hexdigest() != seal["sampling_order_sha256"]:
        raise RuntimeError("shared training sampling changed")
    return data


def gradient_norms(model):
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError("nonfinite gate gradient")
            result[name] = float(parameter.grad.norm())
    return result


def smoke(data, device):
    train = data["train"]
    # Diagnose on the least-easy TRAIN clips, never val. No smoke weights are inherited.
    losses = F.cross_entropy(train["t"], train["labels"], reduction="none")
    chosen = losses.argsort(descending=True)[:min(256, len(losses))]
    a, t, y = (train[k][chosen].to(device) for k in ("a", "t", "labels"))
    before_a, before_t = a.clone(), t.clone()
    reports = {}
    for variant in VARIANTS:
        m = AcceptanceGate(variant, data["statistics"]).to(device)
        if not torch.equal(m(a, t, "unit")[0], t) or not torch.equal(m(a, t, "zero")[0], a):
            raise RuntimeError("forced gate does not exactly reproduce frozen bases")
        opt = torch.optim.AdamW(m.parameters(), lr=TRAINING["lr"], weight_decay=TRAINING["weight_decay"])
        steps = []
        for i in range(3):
            old = {n: p.detach().clone() for n, p in m.named_parameters()}
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(m(a, t)[0], y)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite real gate smoke loss")
            loss.backward()
            grads = gradient_norms(m)
            if not grads or max(grads.values()) <= 0:
                raise RuntimeError("gate gets no train-feature gradient")
            opt.step()
            updates = {n: float((p.detach() - old[n]).abs().max()) for n, p in m.named_parameters()}
            if max(updates.values()) <= 0:
                raise RuntimeError("gate did not update")
            steps.append({"step": i + 1, "loss": float(loss.detach()), "gradients": grads, "updates": updates})
        reports[variant] = steps
    if not torch.equal(a, before_a) or not torch.equal(t, before_t):
        raise RuntimeError("frozen evidence changed")
    return {"all_passed": True, "device": str(device), "torch": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "training_weights_not_inherited": True, "variants": reports}


def train_gate(p, out, data, variant, device):
    directory = out / variant
    directory.mkdir(mode=0o700, exist_ok=False)
    model = AcceptanceGate(variant, data["statistics"]).to(device)
    train, val = data["train"], data["val"]
    a, t, y = (train[k].to(device) for k in ("a", "t", "labels"))
    av, tv = (val[k].to(device) for k in ("a", "t"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING["lr"], weight_decay=TRAINING["weight_decay"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    began, best, steps, history = time.monotonic(), -math.inf, 0, []
    size = TRAINING["batch_size_clips"]
    for epoch, order in enumerate(data["orders"], 1):
        loss_sum, visits, grads = 0.0, 0, {}
        start = time.monotonic()
        for start_idx in range(0, len(order), size):
            indices = order[start_idx:start_idx + size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(a[indices], t[indices])[0], y[indices])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite formal gate loss")
            loss.backward()
            norms = gradient_norms(model)
            for n, norm in norms.items():
                grads[n] = max(grads.get(n, 0), norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["clip_grad_norm"])
            optimizer.step()
            steps += 1
            visits += len(indices)
            loss_sum += float(loss.detach()) * len(indices)
        if visits != p["evidence"]["counts"]["train"] or not grads or max(grads.values()) <= 0:
            raise RuntimeError("incomplete training coverage or missing finite gate gradients")
        with torch.no_grad():
            logits = model(av, tv)[0].cpu()
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("nonfinite gate validation logits")
            metrics = simple_metrics(val, logits)
        row = {"phase": "gate_train", "epoch": epoch, "train_loss": loss_sum / visits,
               "train_clip_visits": visits, "cumulative_steps": steps, "gradients_max": grads,
               "val_macro_f1": metrics["macro_f1"], "val_accuracy": metrics["accuracy"],
               "epoch_seconds": time.monotonic() - start}
        history.append(row)
        if metrics["macro_f1"] > best + 1e-12:
            best = metrics["macro_f1"]
            _torch_save(directory / "best.pt", {"schema": SCHEMA, "variant": variant,
                        "state_dict": model.state_dict(), "statistics": data["statistics"],
                        "source_commit": p["source_commit"], "epoch": epoch, "macro_f1": best,
                        "base_sha256": p["base_sha256"], "training": TRAINING})
        atomic_json(directory / "history.json", history)
        print(json.dumps({"variant": variant, **row, "best_macro_f1": best}), flush=True)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    ck = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    write_json(directory / "result.json", {"training_completed": True, "variant": variant,
               "best_epoch": ck["epoch"], "best_macro_f1": best, "epochs_completed": len(history),
               "optimizer_steps": steps, "parameters": sum(p.numel() for p in model.parameters()),
               "base_parameters_trained": 0, "test_read": False,
               "best_sha256": sha256_file(directory / "best.pt"),
               "resources": {"elapsed_seconds": time.monotonic() - began,
                             "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                             "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}})


def alpha_summary(alpha, a, t):
    x = alpha.cpu().numpy()
    result = {"count": len(x), "mean": float(x.mean()), "std": float(x.std()),
              "quantiles": np.quantile(x, [0, .1, .5, .9, 1]).tolist()}
    result["by_prediction_direction"] = []
    ca, ct = a.argmax(-1).numpy(), t.argmax(-1).numpy()
    for i in range(7):
        for j in range(7):
            values = x[(ca == i) & (ct == j)]
            if len(values):
                result["by_prediction_direction"].append({"A_class": i, "TCN_class": j,
                           "count": len(values), "mean": float(values.mean()), "std": float(values.std())})
    return result


def retention(labels, a, old, new):
    y, ap, tp, npred = (x.numpy() for x in (labels, a.argmax(-1), old.argmax(-1), new.argmax(-1)))
    fixes, harm = (ap != y) & (tp == y), (ap == y) & (tp != y)
    return {"old_TCN_fixes": int(fixes.sum()), "retained_fixes": int((fixes & (npred == y)).sum()),
            "lost_fixes": int((fixes & (npred != y)).sum()), "old_TCN_harm": int(harm.sum()),
            "prevented_old_harm": int((harm & (npred == y)).sum()),
            "new_harm_where_old_TCN_was_correct": int(((ap == y) & (tp == y) & (npred != y)).sum())}


def evaluate_all(p, out, data, device):
    segments, metadata, info = prepare(p["index"], p["source_protocol_dir"])
    val = data["val"]
    directory = out / "evaluation"
    directory.mkdir(mode=0o700, exist_ok=False)
    report = {"schema": SCHEMA, "source_commit": p["source_commit"], "training_completed": True,
              "test_read": False, "validation_role": "val823_development_not_independent_test",
              "A_and_TCN_frozen": True, "source_or_identity_is_not_gate_input": True,
              "training": TRAINING, "counts": info["counts"], "gates": {},
              "A": simple_metrics(val, val["a"]), "TCN": simple_metrics(val, val["t"]),
              "synthetic_cross_record_TCN": simple_metrics(val, val["challenge_t"]),
              "six_sample_class_caution": True, "synthetic_challenge_not_natural_edit_detection": True}
    for variant in VARIANTS:
        result = json.loads((out / variant / "result.json").read_text())
        ck = torch.load(out / variant / "best.pt", map_location="cpu", weights_only=True)
        if result.get("training_completed") is not True or sha256_file(out / variant / "best.pt") != result["best_sha256"] \
                or any(ck.get(k) != v for k, v in {"schema": SCHEMA, "variant": variant,
                       "source_commit": p["source_commit"], "base_sha256": p["base_sha256"],
                       "statistics": data["statistics"], "training": TRAINING}.items()):
            raise ValueError("gate checkpoint does not match fixed configuration/seals")
        model = AcceptanceGate(variant, data["statistics"]).to(device)
        model.load_state_dict(ck["state_dict"], strict=True)
        model.eval()
        with torch.no_grad():
            native, alpha = model(val["a"].to(device), val["t"].to(device))
            challenged, alpha_challenge = model(val["a"].to(device), val["challenge_t"].to(device))
            native, challenged = native.cpu(), challenged.cpu()
        rows = [{"clip_id": clip, "label_id": int(val["labels"][i]),
                 "a_prediction": int(val["a"][i].argmax()), "tcn_prediction": int(native[i].argmax())}
                for i, clip in enumerate(val["clip_ids"])]
        aggregate = summarize(rows, metadata, segments["val"])
        metrics = simple_metrics(val, native)
        if not math.isclose(metrics["macro_f1"], ck["macro_f1"], rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("gate best-F1 checkpoint failed replay")
        pair = _paired(val["labels"].numpy(), native.argmax(-1).numpy(), val["t"].argmax(-1).numpy())
        gate = {"metrics": metrics, "paired_vs_old_TCN": pair,
                "retention": retention(val["labels"], val["a"], val["t"], native),
                "alpha": alpha_summary(alpha, val["a"], val["t"]),
                "slices": aggregate["slices"], "recording_aggregate": aggregate["recording_aggregate"],
                "per_class_changes_vs_A": aggregate["per_class_changes"], "training_result": result,
                "synthetic_cross_record": {"metrics": simple_metrics(val, challenged),
                     "paired_vs_challenged_TCN": _paired(val["labels"].numpy(), challenged.argmax(-1).numpy(),
                                                          val["challenge_t"].argmax(-1).numpy()),
                     "alpha": alpha_summary(alpha_challenge, val["a"], val["challenge_t"])}}
        report["gates"][variant] = gate
        # ID/logit/alpha outputs remain private, never export these.
        with (directory / ("private_" + variant + "_val_predictions.jsonl")).open("x") as stream:
            os.chmod(stream.name, 0o600)
            for i, row in enumerate(rows):
                stream.write(json.dumps({**row, "gate_logits": native[i].tolist(),
                             "alpha": float(alpha[i]), "synthetic_logits": challenged[i].tolist()}) + "\n")
        print(json.dumps({"variant": variant, "macro_f1": metrics["macro_f1"],
                          "accuracy": metrics["accuracy"], "retention": gate["retention"]}), flush=True)
    write_json(directory / "public_safe_summary.json", report)
    write_json(directory / "result.json", {"status": "complete", "val_count": len(val["labels"]), "test_read": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-protocol", required=True)
    args = parser.parse_args()
    p = json.loads(Path(args.run_protocol).read_text())
    out = Path(p["output"])
    if Path(args.run_protocol).resolve() != out / "run_protocol.json" or \
            p.get("schema") != SCHEMA or p.get("training") != TRAINING or not (out / ".launch_once").is_file():
        raise RuntimeError("worker requires fixed launched protocol")
    unchanged(p)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != p["gpu_uuid"] or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker must see exactly selected physical GPU UUID")
    torch.set_num_threads(2)
    torch.manual_seed(TRAINING["seed"])
    torch.cuda.manual_seed_all(TRAINING["seed"])
    device = torch.device("cuda:0")
    if args.stage == "diagnose":
        diagnose(p, out, device)
    else:
        data = cache(out)
        if args.stage == "smoke":
            write_json(out / "cuda_smoke.json", smoke(data, device))
        elif args.stage.startswith("train_") and args.stage[6:] in VARIANTS:
            if json.loads((out / "cuda_smoke.json").read_text())["all_passed"] is not True:
                raise RuntimeError("CUDA smoke required")
            train_gate(p, out, data, args.stage[6:], device)
        elif args.stage == "evaluate":
            evaluate_all(p, out, data, device)
        else:
            raise ValueError("unknown stage")
    unchanged(p)


if __name__ == "__main__":
    main()
