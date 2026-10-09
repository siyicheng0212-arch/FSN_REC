"""Verified historical AdaFocus evidence adapter; never substitute a reference TCN."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import math
import numpy as np
import torch
from experiments.aligned_protocol import DATA_PROTOCOL, EXPECTED_COUNTS, EXPECTED_MANIFEST_SHA
from experiments.relation.data import load_feature_index, load_edges, read_jsonl, sha256_file
from experiments.relation.protocol import validate_source_artifacts, SOURCE_FILES
from experiments.relation.tcn_baseline import FrozenEvidenceTCN, TCNConfig, Segment, build_segments, collate_segments
from experiments.videomae_tcn.evidence import _disjoint_recordings, _candidate_time_checks
from experiments.videomae_tcn.io import ROOT, write_json
from experiments.videomae_tcn.worker import _torch_save
from experiments.direction_gate.model import train_statistics
from experiments.direction_gate.worker import native_outputs, replacement_plan, simple_metrics, gradient_norms
from experiments.context_evidence import worker as context

SOURCE_SHA = "fbf22b530f9046085e12a7e3889a1664602a590da9dfcee207d6810ce455d7d6"
INDEX_SHA = "306e9985f408ffe8254ea989ed2c1629da2f9f0755cef9d064583b3e46df7bfe"
A_SHA = "0709a0ab102fea1a225a98a842fd87c3719d0e3022ac054593577c30a9b0f3f7"
TCN_SHA = "d0d2fb28d397f7caab2d1b56807350a4faa4e9132039dd90626f92e8edd8b329"
CONFIG = {"global_dim": 1280, "local_dim": 2048, "hidden_dim": 64, "blocks": 3, "dropout": .1}


def prepare(index_path, source_dir, base_run):
    index_path, source_dir, base = map(Path, (index_path, source_dir, base_run))
    if sha256_file(ROOT / "experiments/relation/tcn_baseline.py") != SOURCE_SHA or \
            sha256_file(index_path) != INDEX_SHA or sha256_file(base / "tcn/best.pt") != TCN_SHA:
        raise ValueError("not the verified historical source/index/TCN checkpoint")
    ck = torch.load(base / "tcn/best.pt", map_location="cpu", weights_only=True)
    original = json.loads((base / "tcn/private_protocol.json").read_text())
    if ck["config"] != CONFIG or original["config"] != CONFIG or ck["protocol"] != original["seals"] or \
            ck["epoch"] != 7 or ck["val_macro_f1"] != .7959464061963992 or \
            original["representation"] != "mean(global_tokens), mean(local_tokens), A logits" or \
            original["test_evaluated"] is not False or original["A_frozen"] is not True or \
            original["optimizer"] != {"name": "AdamW", "lr": .001, "weight_decay": .01}:
        raise ValueError("historical checkpoint/config/optimizer not verified")
    index = load_feature_index(index_path)
    source = validate_source_artifacts(source_dir, index, source_dir / "edges.jsonl")
    seal = json.loads((index_path.parent / "protocol.json").read_text())
    expected = {"status": "complete", "A_frozen": True, "data_protocol": DATA_PROTOCOL,
                "checkpoint_sha256": A_SHA, "index_sha256": INDEX_SHA,
                "source_audit_sha256": sha256_file(source_dir / "audit.json")}
    expected.update({s + "_manifest_sha256": h for s, h in EXPECTED_MANIFEST_SHA.items()})
    if any(seal.get(k) != v for k, v in expected.items()) or index.checkpoint_sha256 != A_SHA or \
            index.global_dim != 1280 or index.local_dim != 2048 or \
            any(not r.get("feature_sha256") for r in index.clips.values()):
        raise ValueError("historical AdaFocus export seal mismatch")
    counts = {s: sum(r["split"] == s for r in index.clips.values()) for s in ("train", "val")}
    if counts != EXPECTED_COUNTS:
        raise ValueError("expected measured complete 7372/823, no internal split")
    metadata = read_jsonl(source_dir / "manifest_metadata.jsonl")
    _disjoint_recordings(metadata)
    edges = load_edges(source_dir / "edges.jsonl", index)
    _candidate_time_checks(metadata, edges, source)
    segments = {s: build_segments(index, metadata, read_jsonl(source_dir / f"chains_{s}.jsonl"), edges, s)
                for s in ("train", "val")}
    paths = {"index": index_path, "export_protocol": index_path.parent / "protocol.json",
             "source_audit": source_dir / "audit.json", **{n: source_dir / n for n in SOURCE_FILES}}
    input_sha = {n: sha256_file(f) for n, f in paths.items()}
    s = ck["protocol"]
    if s["index_sha256"] != input_sha["index"] or s["source_audit_sha256"] != input_sha["source_audit"] or \
            s["checkpoint_sha256"] != A_SHA or s["script_sha256"] != SOURCE_SHA or \
            {x: s[x+"_manifest_sha256"] for x in ("train", "val")} != EXPECTED_MANIFEST_SHA:
        raise ValueError("historical TCN input seals differ")
    info = {"counts": counts, "manifest_sha256": EXPECTED_MANIFEST_SHA,
        "input_sha256": input_sha, "A_checkpoint_sha256": A_SHA, "model_config": CONFIG,
        "segments": {s: {"count": len(items), "singleton_clips": sum(len(t.clip_ids)==1 for t in items),
                         "maximum_length": max(len(t.clip_ids) for t in items)} for s, items in segments.items()}}
    return segments, metadata, info


def historical_model(base_run, device):
    ck = torch.load(Path(base_run) / "tcn/best.pt", map_location="cpu", weights_only=True)
    model = FrozenEvidenceTCN(TCNConfig(**CONFIG)).to(device)
    model.load_state_dict(ck["model"], strict=True)
    return model.requires_grad_(False).eval()


def exact_replay(data, rows):
    by_id = {r["clip_id"]: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != set(data["clip_ids"]):
        raise ValueError("historical prediction coverage mismatch")
    for i, clip in enumerate(data["clip_ids"]):
        r = by_id[clip]
        if (r["label_id"], r["a_prediction"], r["tcn_prediction"]) != (
            int(data["labels"][i]), int(data["a"][i].argmax()), int(data["t"][i].argmax())):
            raise ValueError("historical prediction not reproduced clip by clip")


def audit(p, out, segments, metadata, device):
    old = historical_model(p["base_run"], device)
    evidence, plans = {}, {}
    for split in ("train", "val"):
        data = native_outputs(old, segments[split], device)
        if split == "val":
            exact_replay(data, read_jsonl(Path(p["base_run"]) / "tcn/private_val_predictions.jsonl"))
            if not math.isclose(simple_metrics(data, data["t"])["macro_f1"], .7959464061963992, abs_tol=1e-12):
                raise RuntimeError("historical old TCN F1 mismatch")
        flat = [Segment([c], s.features[j:j+1], s.a_logits[j:j+1], s.labels[j:j+1], [s.source_kinds[j]])
                for s in segments[split] for j, c in enumerate(s.clip_ids)]
        center = native_outputs(old, flat, device)
        if center["clip_ids"] != data["clip_ids"]:
            raise RuntimeError("center order mismatch")
        data["center"] = center["t"]
        data["context_candidate"] = data["a"] + data["t"] - data["center"]
        evidence[split] = data
        plans[split] = replacement_plan(segments[split], metadata, split, seed=42)
    evidence["segment_orders"], evidence["anchor_orders"] = context.sampling_orders(
        42, len(segments["train"]), len(plans["train"]["replacements"]))
    evidence["statistics"] = train_statistics(evidence["train"]["a"], evidence["train"]["context_candidate"])
    _torch_save(out / "private_evidence.pt", evidence)
    write_json(out / "private_anchor_plans.json", plans)
    write_json(out / "evidence_seal.json", {"cache_sha256": sha256_file(out / "private_evidence.pt"),
        "plans_sha256": sha256_file(out / "private_anchor_plans.json"),
        **{k+"_sha256": hashlib.sha256(evidence[k].numpy().tobytes()).hexdigest() for k in ("segment_orders", "anchor_orders")}})
    write_json(out / "diagnostic_summary.json", {"historical_val_replay_exact": True,
        "test_read": False, "A": simple_metrics(evidence["val"], evidence["val"]["a"]),
        "historical_TCN": simple_metrics(evidence["val"], evidence["val"]["t"]),
        "train_A": simple_metrics(evidence["train"], evidence["train"]["a"]),
        "train_TCN": simple_metrics(evidence["train"], evidence["train"]["t"]),
        "train_anchor_count": len(plans["train"]["replacements"]),
        "normal_and_unchanged_anchor_views_only": True, "augmentation_applied": False})


def smoke(p, out, segments, data, plans, device):
    batch = collate_segments([s for s in segments["train"] if len(s.clip_ids)>1][:16])
    old = historical_model(p["base_run"], device)
    with torch.no_grad():
        expected = old(batch[0].to(device), batch[1].to(device), batch[3].to(device))
    report = {}
    for variant in p["variants"]:
        context.seed_all(42)
        model = context.make_model(p, data["statistics"], variant, device).eval()
        with torch.no_grad():
            _, details = model(batch[0].to(device), batch[1].to(device), batch[3].to(device), return_details=True)
            torch.testing.assert_close(details["full"], expected, atol=2e-5, rtol=2e-5)
        model.train(); opt = torch.optim.AdamW((w for w in model.parameters() if w.requires_grad), lr=.001, weight_decay=.01)
        rows = []
        for step in range(3):
            before = {n: w.detach().clone() for n, w in model.named_parameters()}
            opt.zero_grad(set_to_none=True)
            loss = context.batch_loss(model, batch, device); loss.backward()
            grads = gradient_norms(model)
            if not grads or max(grads.values()) <= 0 or not bool(torch.isfinite(loss)):
                raise RuntimeError("missing or nonfinite CUDA gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
            delta = max(float((w.detach()-before[n]).abs().max()) for n,w in model.named_parameters())
            if delta <= 0: raise RuntimeError("no CUDA update")
            rows.append({"step": step+1, "loss": float(loss.detach()), "gradient_max": max(grads.values()), "update_max": delta})
        singleton = collate_segments([Segment([s.clip_ids[0]],s.features[:1],s.a_logits[:1],s.labels[:1],s.source_kinds[:1]) for s in segments["train"][:4]])
        model.eval()
        with torch.no_grad():
            z = model(singleton[0].to(device), singleton[1].to(device), singleton[3].to(device))
            if variant == "context_delta_continue" and not torch.equal(z, singleton[1].to(device)):
                raise RuntimeError("context singleton must be exact A")
        report[variant] = rows
    write_json(out / "cuda_smoke.json", {"all_passed": True, "historical_full_stream_exact_tolerance": 2e-5,
        "steps": 3, "variants": report, "A_trained": False, "precision": "float32"})
