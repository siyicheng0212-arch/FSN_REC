"""Train R first, freeze its decisions, then train one of four identical TCNs."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from experiments.relation.data import EVIDENCE_KEYS, RelationDataset, sha256_file
from experiments.relation.network import RelationConfig, RelationNet
from experiments.relation.train import evaluate as evaluate_relation
from .config import STRATEGIES, baseline_source_hashes, build_base, load_config
from .gates import select_gates
from .hard_negatives import build_hard_negatives
from .io import code_hashes, fresh_output, load_bundle, load_chain, write_json, write_jsonl
from .model import SegmentedTCN


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gradients_finite(model):
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise RuntimeError("nonfinite gradient; preserve outputs and stop")


def evidence_batch(index, pairs, device):
    sides = []
    for position in (0, 1):
        arrays = [index.read(pair[position]) for pair in pairs]
        sides.append({key: torch.from_numpy(np.stack([array[key] for array in arrays])).to(device)
                      for key in EVIDENCE_KEYS})
    return sides


@torch.no_grad()
def score_pairs(model, index, pairs, device, batch_size=16):
    model.eval()
    model.requires_grad_(False)
    pairs = sorted(set(pairs))
    result = {}
    for offset in range(0, len(pairs), batch_size):
        batch = pairs[offset:offset + batch_size]
        values = model(*evidence_batch(index, batch, device)).sigmoid()
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError("nonfinite R score")
        result.update({pair: float(value) for pair, value in zip(batch, values.cpu())})
    return result


def read_scores(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    scores = {}
    for row in rows:
        pair = (row["left_clip_id"], row["right_clip_id"])
        value = row["score"]
        if pair in scores or isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid or duplicate relation score")
        scores[pair] = value
    return scores


def load_relation(path, bundle, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved["fingerprint"] != bundle.fingerprint:
        raise ValueError("R checkpoint and current frozen evidence/protocol disagree")
    if saved["protocol"].get("code_sha256") != code_hashes():
        raise ValueError("R checkpoint source code differs from the current sealed implementation")
    model = RelationNet(RelationConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["model"], strict=True)
    return model.eval().requires_grad_(False), saved


def train_relation(bundle, config, seed, output, device):
    settings = config["relation"]
    hard, audit = build_hard_negatives(bundle.metadata, bundle.edges, seed=seed,
                                       ratio=config["hard_negative_ratio"])
    combined = bundle.edges + hard
    train_rows = [edge for edge in combined if edge["split"] == "train"]
    positive = sum(edge["status"] == "C" for edge in train_rows)
    negative = sum(edge["status"] == "D" for edge in train_rows)
    if not positive or not negative:
        raise ValueError("R requires positive and negative train pairs")
    seed_everything(seed)
    relation_config = RelationConfig(global_dim=bundle.index.global_dim, local_dim=bundle.index.local_dim)
    model = RelationNet(relation_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    loaders = {split: DataLoader(RelationDataset(bundle.index, combined, split),
                                batch_size=settings["batch_size"], shuffle=split == "train",
                                generator=torch.Generator().manual_seed(seed)) for split in ("train", "val")}
    if not len(loaders["val"].dataset):
        raise ValueError("no source-policy validation pairs")
    out = fresh_output(output)
    protocol = {"schema": "fsn-tcn-relation-v1", "fingerprint": bundle.fingerprint,
                "code_sha256": code_hashes(), "seed": seed, "config": config,
                "hard_negative_audit": audit, "train_positive": positive, "train_negative": negative,
                "positive_weight": negative / positive,
                "selection": "unweighted original val source-policy BCE; no synthetic val used for selection",
                "label_policy": "source weak labels plus known synthetic train cross-record negatives",
                "threshold_calibrated": False, "A_frozen": True}
    write_json(out / "protocol.json", protocol, exclusive=True)
    write_jsonl(out / "hard_train_edges.jsonl", hard)
    weight = torch.tensor(negative / positive, device=device)
    history, best, stale, start = [], float("inf"), 0, time.monotonic()
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total, count = 0.0, 0
        for batch in loaders["train"]:
            left, right = ({key: value.to(device) for key, value in batch[side].items()} for side in ("left", "right"))
            targets = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(left, right), targets, pos_weight=weight)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite relation training loss")
            loss.backward()
            gradients_finite(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_grad"])
            optimizer.step()
            total += float(loss.detach()) * len(targets)
            count += len(targets)
        val = evaluate_relation(model, loaders["val"], device, config["threshold"])
        history.append({"epoch": epoch, "train_bce": total / count, "val_source_policy": val})
        state = {"model": model.state_dict(), "config": asdict(relation_config), "fingerprint": bundle.fingerprint,
                 "seed": seed, "epoch": epoch, "protocol": protocol}
        torch.save(state, out / "last.pt")
        if val["bce"] < best:
            best, stale = val["bce"], 0
            torch.save(state, out / "best.pt")
        else:
            stale += 1
        write_json(out / "history.json", history)
        print(json.dumps({"stage": "relation", "seed": seed, "epoch": epoch,
                          "train_bce": total / count, "val_bce": val["bce"]}), flush=True)
        if stale >= settings["patience"]:
            break
    model, saved = load_relation(out / "best.pt", bundle, device)
    pairs = [(edge["left_clip_id"], edge["right_clip_id"]) for edge in bundle.edges]
    scores = score_pairs(model, bundle.index, pairs, device, settings["batch_size"])
    write_jsonl(out / "scores.jsonl", [{"left_clip_id": pair[0], "right_clip_id": pair[1], "score": value}
                                      for pair, value in sorted(scores.items())])
    metrics = {"best_epoch": saved["epoch"], "epochs_completed": len(history),
               "elapsed_seconds": time.monotonic() - start,
               "R_parameters": sum(p.numel() for p in model.parameters()),
               "scores_sha256": sha256_file(out / "scores.jsonl"),
               "best_sha256": sha256_file(out / "best.pt"),
               "hard_negative_audit": audit,
               "source_policy_val": evaluate_relation(model, loaders["val"], device, config["threshold"]),
               "true_edit_boundary_accuracy_claimed": False}
    write_json(out / "result.json", metrics, exclusive=True)
    return metrics


def gates_for(bundle, chain, scores, strategy, config, seed):
    ids = chain["ordered_clip_ids"]
    pairs = list(zip(ids, ids[1:]))
    if any(valid and pair not in scores for pair, valid in zip(pairs, chain["eligible"])):
        raise ValueError("missing frozen relation decision on an eligible candidate")
    return select_gates(strategy, chain["eligible"], [bundle.metadata[left]["source_kind"] for left, _ in pairs],
                        [scores.get(pair, 0.0) for pair in pairs], threshold=config["threshold"],
                        seed=seed, chain_id=chain["chain_id"])


def train_tcn(bundle, config, seed, strategy, relation_dir, output, device):
    if strategy not in STRATEGIES:
        raise ValueError("unknown strategy")
    relation_dir = Path(relation_dir)
    relation, saved = load_relation(relation_dir / "best.pt", bundle, device)
    if saved["seed"] != seed or saved["protocol"]["config"] != config:
        raise ValueError("R must use the same predeclared seed and suite config")
    result = json.loads((relation_dir / "result.json").read_text())
    if (result["scores_sha256"] != sha256_file(relation_dir / "scores.jsonl")
            or result["best_sha256"] != sha256_file(relation_dir / "best.pt")):
        raise ValueError("R output seal changed")
    scores = read_scores(relation_dir / "scores.jsonl")
    del relation
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    input_dim = bundle.index.global_dim + bundle.index.local_dim
    seed_everything(seed)
    base = build_base(config, input_dim, device)
    model = SegmentedTCN(base)
    settings = config["optimization"]
    optimizer = torch.optim.AdamW(base.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    out = fresh_output(output)
    protocol = {"schema": "fsn-tcn-training-v1", "fingerprint": bundle.fingerprint,
                "code_sha256": code_hashes(), "seed": seed, "strategy": strategy, "config": config,
                "R_best_sha256": sha256_file(relation_dir / "best.pt"),
                "scores_sha256": result["scores_sha256"], "input_dim": input_dim,
                "baseline_historical_reproduction": False,
                "baseline_source_sha256": baseline_source_hashes(config),
                "selection": "original val823 Macro-F1, tie lower CE",
                "singleton": "exact frozen A logits; zero trainable contribution",
                "loss_denominator": "all clips in chain batch including frozen singletons",
                "all_four_groups_identical_initialization_for_same_seed": True}
    write_json(out / "protocol.json", protocol, exclusive=True)
    gates = {chain["chain_id"]: gates_for(bundle, chain, scores, strategy, config, seed)
             for chain in bundle.chains["train"]}
    if not any(any(mask) for mask in gates.values()):
        raise RuntimeError("no trainable connected segments under this fixed strategy; preserve outputs, do not lower the threshold silently")
    rng = random.Random(seed)
    history, best, stale, start = [], None, 0, time.monotonic()
    from .evaluate import evaluate_chains
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        ordered = list(bundle.chains["train"])
        rng.shuffle(ordered)
        total, clips, steps, frozen_batches, trainable_clips = 0.0, 0, 0, 0, 0
        for offset in range(0, len(ordered), settings["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            losses, batch_clips = [], 0
            for chain in ordered[offset:offset + settings["batch_size"]]:
                features, a_logits, labels = load_chain(bundle, chain, device)
                open_edges = gates[chain["chain_id"]]
                logits = model(features, a_logits, open_edges)
                loss = F.cross_entropy(logits, labels, reduction="sum")
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite TCN loss")
                losses.append(loss)
                batch_clips += len(labels)
                trainable_clips += sum((i > 0 and open_edges[i - 1]) or (i < len(open_edges) and open_edges[i])
                                       for i in range(len(labels)))
            loss = sum(losses) / batch_clips
            total += float(loss.detach()) * batch_clips
            clips += batch_clips
            if loss.requires_grad:
                loss.backward()
                gradients_finite(base)
                torch.nn.utils.clip_grad_norm_(base.parameters(), settings["clip_grad"])
                optimizer.step()
                steps += 1
            else:
                frozen_batches += 1
        metrics, _ = evaluate_chains(bundle, bundle.chains["val"], base, scores, strategy=strategy,
                                     threshold=config["threshold"], seed=seed, device=device)
        key = (metrics["metrics"]["macro_f1"], -metrics["cross_entropy"])
        history.append({"epoch": epoch, "train_ce": total / clips,
                        "train_clips": clips, "trainable_clip_visits": trainable_clips,
                        "optimizer_steps": steps, "all_frozen_batches": frozen_batches, "val": metrics})
        state = {"model": base.state_dict(), "fingerprint": bundle.fingerprint, "seed": seed,
                 "strategy": strategy, "config": config, "epoch": epoch, "protocol": protocol}
        torch.save(state, out / "last.pt")
        if best is None or key > best:
            best, stale = key, 0
            torch.save(state, out / "best.pt")
        else:
            stale += 1
        write_json(out / "history.json", history)
        print(json.dumps({"stage": "tcn", "strategy": strategy, "seed": seed, "epoch": epoch,
                          "train_ce": total / clips, "val_macro_f1": key[0], "optimizer_steps": steps}), flush=True)
        if stale >= settings["patience"]:
            break
    saved = torch.load(out / "best.pt", map_location=device, weights_only=True)
    base.load_state_dict(saved["model"], strict=True)
    metrics, private = evaluate_chains(bundle, bundle.chains["val"], base, scores, strategy=strategy,
                                       threshold=config["threshold"], seed=seed, device=device)
    metrics.update(best_epoch=saved["epoch"], epochs_completed=len(history), elapsed_seconds=time.monotonic() - start,
                   trainable_TCN_parameters=sum(p.numel() for p in base.parameters()),
                   A_trainable_parameters=0, R_trainable_parameters_during_TCN=0,
                   best_sha256=sha256_file(out / "best.pt"), strategy=strategy, seed=seed,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None)
    write_json(out / "result.json", metrics, exclusive=True)
    write_jsonl(out / "val_predictions.jsonl", private)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("relation", "tcn"))
    for name in ("feature-index", "source-protocol-dir", "config", "output", "smoke-report"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--strategy", choices=STRATEGIES)
    parser.add_argument("--relation-dir")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.seed not in config["seeds"]:
        parser.error("seed is not in the predeclared config")
    bundle = load_bundle(args.feature_index, args.source_protocol_dir)
    smoke = json.loads(Path(args.smoke_report).read_text())
    if (smoke.get("all_passed") is not True or smoke.get("real_cuda_tcn_smoke_completed") is not True
            or smoke.get("fingerprint") != bundle.fingerprint
            or smoke.get("config_sha256") != sha256_file(args.config)
            or smoke.get("source_sha256") != code_hashes()
            or smoke.get("baseline_source_sha256") != baseline_source_hashes(config)):
        raise RuntimeError("formal training requires the successful matching real-cache CUDA preflight")
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal training requires CUDA; CPU helpers are for unit tests only")
    if args.stage == "relation":
        train_relation(bundle, config, args.seed, args.output, args.device)
    else:
        if args.strategy is None or args.relation_dir is None:
            parser.error("TCN stage requires --strategy and --relation-dir")
        train_tcn(bundle, config, args.seed, args.strategy, args.relation_dir, args.output, args.device)


if __name__ == "__main__":
    main()
