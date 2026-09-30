"""Aggregate a finished semantic-window experiment without exporting clip rows.

This script deliberately keeps logits and group identifiers in memory. The
output contains only aggregate metrics and exploratory group-bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.full_data import FullClipDataset
from experiments.metrics import DEFAULT_CLASS_NAMES
from experiments.semantic_window_evidence import SemanticWindowCorroborator
from experiments.train_semantic_window import _trained_original
from experiments.train_adafocus import validate_splits


RUNS = (("corroborated", 42), ("corroborated", 14),
        ("corroborated", 15), ("plain_mean", 42))


def _classification(logits: np.ndarray, labels: np.ndarray) -> dict:
    pred = logits.argmax(axis=1)
    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (labels, pred), 1)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = cm.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(7), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(7), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(7), where=precision + recall > 0)
    weights = support / max(int(support.sum()), 1)
    overall = {
        "accuracy": float(tp.sum() / support.sum()),
        "balanced_accuracy": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "micro_precision": float(tp.sum() / support.sum()),
        "micro_recall": float(tp.sum() / support.sum()),
        "micro_f1": float(tp.sum() / support.sum()),
        "weighted_precision": float(np.dot(weights, precision)),
        "weighted_recall": float(np.dot(weights, recall)),
        "weighted_f1": float(np.dot(weights, f1)),
    }
    per_class = []
    for index, name in enumerate(DEFAULT_CLASS_NAMES):
        fp = int(predicted[index] - tp[index])
        tn = int(len(labels) - support[index] - fp)
        per_class.append({
            "class_id": index, "class_name": name, "support": int(support[index]),
            "precision": float(precision[index]), "recall": float(recall[index]),
            "f1": float(f1[index]),
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
        })
    return {"support": int(len(labels)), **overall, "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "sweep_to_reperfusion": int(cm[3, 4]),
            "reperfusion_to_sweep": int(cm[4, 3])}


def _binary_ranking(logits: np.ndarray, labels: np.ndarray) -> dict:
    mask = (labels == 3) | (labels == 4)
    y = (labels[mask] == 4).astype(np.int64)
    score = logits[mask, 4] - logits[mask, 3]
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    order = np.argsort(-score, kind="stable")
    sorted_y = y[order]
    cumulative_tp = np.cumsum(sorted_y)
    positions = np.flatnonzero(sorted_y) + 1
    ap = float(np.mean(cumulative_tp[positions - 1] / positions))
    positive_scores, negative_scores = score[y == 1], score[y == 0]
    auc = float(((positive_scores[:, None] > negative_scores[None, :]).sum()
                 + 0.5 * (positive_scores[:, None] == negative_scores[None, :]).sum())
                / (pos * neg))
    return {"positive_class": 4, "negative_class": 3,
            "positive_support": pos, "negative_support": neg,
            "auroc": auc, "average_precision": ap}


def _calibration(logits: np.ndarray, labels: np.ndarray) -> dict:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    prob = exp / exp.sum(axis=1, keepdims=True)
    confidence = prob.max(axis=1)
    correct = (prob.argmax(axis=1) == labels).astype(float)
    nll = float(-np.log(prob[np.arange(len(labels)), labels].clip(1e-12)).mean())
    one_hot = np.eye(7)[labels]
    brier = float(np.mean(np.sum((prob - one_hot) ** 2, axis=1)))
    ece = 0.0
    for lower in np.linspace(0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence >= lower) & ((confidence <= upper) if upper >= 1 else (confidence < upper))
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return {"nll": nll, "brier_multiclass_sum": brier, "ece_10_equal_width": ece}


def _slices(logits: np.ndarray, labels: np.ndarray, records: list) -> dict:
    durations = np.asarray([r.clip_duration_sec for r in records])
    sources = np.asarray([r.source_collection for r in records])
    result = {}
    masks = {"duration_le_10s": durations <= 10,
             "duration_gt_10s": durations > 10,
             "duration_lt_0_1s": durations < 0.1}
    masks.update({"source_" + str(source): sources == source for source in sorted(set(sources))})
    for name, mask in masks.items():
        if not mask.any():
            continue
        metric = _classification(logits[mask], labels[mask])
        present = [row["f1"] for row in metric["per_class"] if row["support"]]
        result[name] = {"support": metric["support"], "accuracy": metric["accuracy"],
                        "present_class_macro_f1": float(np.mean(present)),
                        "present_class_count": len(present)}
    return result


def _bootstrap_delta(logits_a: np.ndarray, logits_b: np.ndarray,
                     labels: np.ndarray, records: list, seed: int = 42) -> dict:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.group_id, []).append(index)
    indices = list(groups.values())
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(1000):
        chosen = rng.integers(0, len(indices), size=len(indices))
        rows = np.asarray([row for choice in chosen for row in indices[int(choice)]])
        fa = _classification(logits_a[rows], labels[rows])["macro_f1"]
        fb = _classification(logits_b[rows], labels[rows])["macro_f1"]
        samples.append(fb - fa)
    actual = (_classification(logits_b, labels)["macro_f1"] -
              _classification(logits_a, labels)["macro_f1"])
    return {"paired_delta_macro_f1": actual,
            "group_bootstrap_95_percentile": np.quantile(samples, [0.025, 0.975]).tolist(),
            "group_count": len(groups), "replicates": len(samples),
            "exploratory_due_to_reused_validation": True}


def _run(args: argparse.Namespace) -> dict:
    split = validate_splits(args.manifest_dir, include_test=False)
    device = torch.device("cuda:0")
    dataset = FullClipDataset(args.manifest_dir / "val.jsonl", args.cache_dir,
                              num_frames=36, crop_size=224, sampling="uniform")
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=True)
    baseline = _trained_original(args.baseline_checkpoint, device, split)
    heads = {}
    expected_f1 = {}
    for mode, seed in RUNS:
        directory = args.results_root / mode / f"seed_{seed}"
        result = json.loads((directory / "result.json").read_text())
        if result.get("test_metrics") is not None:
            raise ValueError("unexpected independent test metrics")
        state = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        if state["epoch"] != result["best_epoch"]:
            raise ValueError("best epoch differs between result and checkpoint")
        key = f"{mode}_{seed}"
        head = SemanticWindowCorroborator(corroboration=mode == "corroborated")
        head.load_state_dict(state["head"], strict=True)
        heads[key] = head.to(device).eval()
        expected_f1[key] = result["best_val_macro_f1"]

    collected = {key: [] for key in ("original", *heads)}
    labels_list, masks_list, magnitudes = [], {key: [] for key in heads}, {key: [] for key in heads}
    started = time.perf_counter()
    for batch in loader:
        frames = batch["video"].to(device, non_blocking=True)
        labels_list.append(batch["label"].numpy())
        captured: dict[str, list[torch.Tensor]] = {"global": [], "local": []}
        handles = [
            baseline.core.global_CNN.new_fc.register_forward_hook(
                lambda _m, _i, out: captured["global"].append(out)),
            baseline.core.local_CNN.new_fc.register_forward_hook(
                lambda _m, _i, out: captured["local"].append(out)),
        ]
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                original = baseline(frames)
        finally:
            for handle in handles:
                handle.remove()
        batch_size, frame_count = frames.shape[:2]
        global_votes = captured["global"][0].reshape(batch_size, 8, 7)
        local_votes = captured["local"][0].reshape(batch_size, 12, 7)
        global_pos = torch.linspace(0, frame_count - 1, 8, device=device).round() / (frame_count - 1)
        global_pos = global_pos[None].expand(batch_size, -1)
        input_pos = torch.linspace(0, frame_count - 1, 36, device=device).round() / (frame_count - 1)
        focus = original["eval_outputs"][8].long().reshape(-1)
        local_pos = input_pos[None].expand(batch_size, -1).reshape(-1).index_select(0, focus).reshape(batch_size, 12)
        base_logits = original["logits"].float()
        collected["original"].append(base_logits.cpu().numpy())
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for key, head in heads.items():
                evidence = head(global_votes, local_votes, global_pos, local_pos,
                                base_logits, frames)
                correction = evidence["correction"].float()
                logits = base_logits + correction
                collected[key].append(logits.cpu().numpy())
                masks_list[key].append(evidence["corroboration_gate"].cpu().numpy())
                magnitudes[key].append(correction.abs().mean(dim=1).cpu().numpy())
    elapsed = time.perf_counter() - started
    labels = np.concatenate(labels_list)
    logits = {key: np.concatenate(value) for key, value in collected.items()}
    summary = {"schema_version": "fsn-semantic-final-audit-1.0",
               "selection_set": "823 development validation clips, not independent test",
               "num_clips": len(dataset), "trained_backbone_seeds": [42],
               "shared_backbone_for_head_seeds": [42, 14, 15],
               "inference_seconds_one_shared_backbone_pass": elapsed,
               "inference_clips_per_second_shared_pass": len(dataset) / elapsed,
               "models": {}, "paired": {}}
    for key, values in logits.items():
        metrics = _classification(values, labels)
        if key in expected_f1 and not math.isclose(metrics["macro_f1"], expected_f1[key], abs_tol=1e-6):
            raise RuntimeError(f"best macro-F1 not reproduced for {key}: {metrics['macro_f1']} vs {expected_f1[key]}")
        summary["models"][key] = {
            "classification": metrics, "binary_focus_ranking": _binary_ranking(values, labels),
            "calibration": _calibration(values, labels), "slices": _slices(values, labels, dataset.records),
        }
    base_pred = logits["original"].argmax(axis=1)
    for key in heads:
        pred = logits[key].argmax(axis=1)
        changed = pred != base_pred
        summary["models"][key]["branch_audit"] = {
            "gate_fraction": float(np.concatenate(masks_list[key]).mean()),
            "mean_abs_logit_correction": float(np.concatenate(magnitudes[key]).mean()),
            "changed_predictions": int(changed.sum()),
            "corrected": int((changed & (pred == labels) & (base_pred != labels)).sum()),
            "harmed": int((changed & (pred != labels) & (base_pred == labels)).sum()),
        }
        summary["paired"][f"original_vs_{key}"] = _bootstrap_delta(
            logits["original"], logits[key], labels, dataset.records)
    summary["paired"]["plain_mean_42_vs_corroborated_42"] = _bootstrap_delta(
        logits["plain_mean_42"], logits["corroborated_42"], labels, dataset.records)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    summary = _run(args)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "num_clips": summary["num_clips"],
                      "macro_f1": {key: row["classification"]["macro_f1"]
                                   for key, row in summary["models"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
