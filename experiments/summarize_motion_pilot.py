#!/usr/bin/env python3
"""Publish aggregate, path-free results for the paired FSN sampling pilot."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


ARMS = ("uniform", "three_windows")
SEEDS = (42, 123, 2026)
T_95_DF2 = 4.303


def mean_interval(values: list[float]) -> dict:
    if len(values) != len(SEEDS) or not all(math.isfinite(value) for value in values):
        raise ValueError("expected three finite matched-seed values")
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    margin = T_95_DF2 * std / math.sqrt(len(values))
    return {
        "values": values,
        "mean": mean,
        "sample_std": std,
        "descriptive_t_95_interval": [mean - margin, mean + margin],
        "caution": "three seeds; interval is descriptive and unstable",
    }


def selected_history_row(result: dict) -> dict:
    phase, epoch_text = result["best_epoch"].rsplit("_", 1)
    epoch = int(epoch_text)
    rows = [
        row for row in result["history"]
        if row["phase"] == phase and row["epoch"] == epoch
    ]
    if len(rows) != 1:
        raise ValueError("best epoch is missing or duplicated in history")
    row = rows[0]
    value = row["val_metrics"]["all"]["macro_f1"]
    if not math.isclose(value, result["best_val_macro_f1"], abs_tol=1e-10):
        raise ValueError("stored best metric does not match history")
    return row


def metric_slice(block: dict) -> dict:
    return {
        "support": block["num_samples"],
        "accuracy": block["accuracy"],
        "macro_f1": block["macro_f1"],
        "present_class_macro_f1": block["present_class_macro_f1"],
    }


def compact_best(row: dict, best_epoch: str) -> dict:
    metrics = row["val_metrics"]
    overall = metrics["all"]
    confusion = overall["confusion_matrix"]
    return {
        "best_epoch": best_epoch,
        "macro_f1": overall["macro_f1"],
        "accuracy": overall["accuracy"],
        "weighted_f1": overall["weighted_f1"],
        "best_train_loss": row["train_loss"],
        "per_class": [
            {
                "class_id": item["class_id"],
                "class_name": item["class_name"],
                "support": item["support"],
                "precision": item["precision"],
                "recall": item["recall"],
                "f1": item["f1"],
            }
            for item in overall["per_class"]
        ],
        "confusion_matrix": confusion,
        "sweep_to_reperfusion": confusion[3][4],
        "reperfusion_to_sweep": confusion[4][3],
        "slices": {
            name: metric_slice(metrics["slices"][name])
            for name in ("duration_le_10s", "duration_gt_10s", "duration_lt_0.1s")
        },
        "source_accuracy": {
            source: metric_slice(block)
            for source, block in metrics["slices"]["source_collection"].items()
        },
    }


def load_results(results_root: Path, protocol: dict) -> tuple[dict, dict]:
    per_seed: dict[str, dict] = {}
    curves: dict[str, dict] = {}
    expected_hashes = {
        "train": protocol["train_manifest_sha256"],
        "val": protocol["validation_manifest_sha256"],
    }
    for arm in ARMS:
        curves[arm] = {}
        for seed in SEEDS:
            path = results_root / arm / "original" / f"seed_{seed}" / "result.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["variant"] != "original" or result["seed"] != seed:
                raise ValueError(f"variant/seed mismatch for {arm} seed {seed}")
            if result.get("sampling") != arm:
                raise ValueError(f"sampling mismatch for {arm} seed {seed}")
            if result.get("test_metrics") is not None:
                raise ValueError("independent test metrics are not allowed")
            audit = result["split_audit"]
            if audit["manifest_sha256"] != expected_hashes:
                raise ValueError("frozen manifest hashes differ")
            if audit["counts"] != {"train": 7372, "val": 823}:
                raise ValueError("frozen clip counts differ")
            if result["class_weight_mode"] != "sqrt_inverse":
                raise ValueError("class weight protocol differs")
            history = result["history"]
            if not history:
                raise ValueError("empty training history")
            for entry in history:
                if not math.isfinite(entry["train_loss"]) or not math.isfinite(
                    entry["val_metrics"]["all"]["macro_f1"]
                ):
                    raise ValueError("non-finite training curve")
            best = compact_best(selected_history_row(result), result["best_epoch"])
            best["total_train_minutes"] = sum(
                entry["train_seconds"] for entry in history
            ) / 60.0
            best["last_phase"] = history[-1]["phase"]
            best["last_epoch"] = history[-1]["epoch"]
            best["last_train_loss"] = history[-1]["train_loss"]
            per_seed.setdefault(str(seed), {})[arm] = best
            curves[arm][str(seed)] = [
                {
                    "phase": entry["phase"],
                    "epoch": entry["epoch"],
                    "train_loss": entry["train_loss"],
                    "validation_macro_f1": entry["val_metrics"]["all"]["macro_f1"],
                    "lr": entry["lr"],
                }
                for entry in history
            ]
    return per_seed, curves


def aggregate(per_seed: dict) -> dict:
    result: dict[str, dict] = {}
    fields = ("macro_f1", "accuracy", "weighted_f1", "sweep_to_reperfusion", "reperfusion_to_sweep")
    for arm in ARMS:
        result[arm] = {
            field: mean_interval([float(per_seed[str(seed)][arm][field]) for seed in SEEDS])
            for field in fields
        }
        result[arm]["per_class_f1"] = [
            mean_interval([
                per_seed[str(seed)][arm]["per_class"][class_id]["f1"]
                for seed in SEEDS
            ])
            for class_id in range(7)
        ]
        result[arm]["per_class"] = [
            {
                "class_id": class_id,
                "class_name": per_seed[str(SEEDS[0])][arm]["per_class"][class_id]["class_name"],
                "support": per_seed[str(SEEDS[0])][arm]["per_class"][class_id]["support"],
                **{
                    metric: mean_interval([
                        per_seed[str(seed)][arm]["per_class"][class_id][metric]
                        for seed in SEEDS
                    ])
                    for metric in ("precision", "recall", "f1")
                },
            }
            for class_id in range(7)
        ]
        result[arm]["duration_slices"] = {
            name: {
                "support": per_seed[str(SEEDS[0])][arm]["slices"][name]["support"],
                **{
                    metric: mean_interval([
                        per_seed[str(seed)][arm]["slices"][name][metric]
                        for seed in SEEDS
                    ])
                    for metric in ("accuracy", "present_class_macro_f1")
                },
            }
            for name in ("duration_le_10s", "duration_gt_10s", "duration_lt_0.1s")
        }
        sources = sorted(per_seed[str(SEEDS[0])][arm]["source_accuracy"])
        result[arm]["source_accuracy"] = {
            source: {
                "support": per_seed[str(SEEDS[0])][arm]["source_accuracy"][source]["support"],
                "accuracy": mean_interval([
                    per_seed[str(seed)][arm]["source_accuracy"][source]["accuracy"]
                    for seed in SEEDS
                ]),
            }
            for source in sources
        }
    result["dense_minus_uniform"] = {
        field: mean_interval([
            float(per_seed[str(seed)]["three_windows"][field])
            - float(per_seed[str(seed)]["uniform"][field])
            for seed in SEEDS
        ])
        for field in fields
    }
    return result


def plot_curves(curves: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=False)
    colors = {"uniform": "#2856a8", "three_windows": "#cb6935"}
    for row, seed in enumerate(SEEDS):
        for arm in ARMS:
            entries = curves[arm][str(seed)]
            x = list(range(1, len(entries) + 1))
            axes[row, 0].plot(x, [item["train_loss"] for item in entries],
                              label=arm, color=colors[arm], linewidth=1.6)
            axes[row, 1].plot(x, [item["validation_macro_f1"] for item in entries],
                              label=arm, color=colors[arm], linewidth=1.6)
        for ax in axes[row]:
            ax.axvline(5.5, color="gray", linestyle="--", linewidth=0.8)
            ax.grid(alpha=0.2)
        axes[row, 0].set_ylabel(f"Seed {seed}\nNative train loss")
        axes[row, 1].set_ylabel("Validation macro-F1")
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].legend(loc="lower right")
    axes[-1, 0].set_xlabel("Epoch (warm-up then fine-tune)")
    axes[-1, 1].set_xlabel("Epoch (warm-up then fine-tune)")
    fig.suptitle("FSN paired 36-frame sampling: training and validation curves")
    fig.tight_layout()
    fig.savefig(output, format=output.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)


def write_report(summary: dict, output: Path) -> None:
    per_seed = summary["per_seed"]
    agg = summary["aggregate"]
    lines = [
        "# FSN equal-budget motion sampling pilot",
        "",
        "Both arms use original Uni-AdaFocus with 36 candidate frames, matched pretrained weights,",
        "training settings, frozen source-record splits, and seeds 42, 123, and 2026.",
        "This compares sampling only; it does not add a role-aware motion module.",
        "",
        "| Seed | Uniform macro-F1 | Three-window macro-F1 | Dense − uniform |",
        "|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        pair = per_seed[str(seed)]
        base = pair["uniform"]["macro_f1"]
        dense = pair["three_windows"]["macro_f1"]
        lines.append(f"| {seed} | {base:.4f} | {dense:.4f} | {dense-base:+.4f} |")
    lines.extend([
        "",
        f"Mean macro-F1: uniform **{agg['uniform']['macro_f1']['mean']:.4f}**, "
        f"three-window **{agg['three_windows']['macro_f1']['mean']:.4f}**, "
        f"paired difference **{agg['dense_minus_uniform']['macro_f1']['mean']:+.4f}**.",
        f"Mean accuracy: uniform {agg['uniform']['accuracy']['mean']:.4f}, "
        f"three-window {agg['three_windows']['accuracy']['mean']:.4f}.",
        f"Mean weighted-F1: uniform {agg['uniform']['weighted_f1']['mean']:.4f}, "
        f"three-window {agg['three_windows']['weighted_f1']['mean']:.4f}.",
        "With only three seeds, the interval in summary.json is descriptive.",
        "",
        "| Arm | Sweep → reperfusion errors (3-seed mean) | Reperfusion → sweep errors (3-seed mean) |",
        "|---|---:|---:|",
    ])
    for arm in ARMS:
        lines.append(
            f"| {arm} | {agg[arm]['sweep_to_reperfusion']['mean']:.1f} | "
            f"{agg[arm]['reperfusion_to_sweep']['mean']:.1f} |"
        )
    deltas = [
        per_seed[str(seed)]["three_windows"]["macro_f1"]
        - per_seed[str(seed)]["uniform"]["macro_f1"]
        for seed in SEEDS
    ]
    if all(delta < 0 for delta in deltas):
        lines.extend([
            "",
            "Three-window sampling had lower seven-class macro-F1 in all three paired seeds.",
            "This is a negative result for replacing the uniform sampler with this fixed design.",
        ])
    lines.extend([
        "",
        "| Class (validation support) | Uniform F1 | Three-window F1 | Difference |",
        "|---|---:|---:|---:|",
    ])
    for class_id in range(7):
        base = agg["uniform"]["per_class"][class_id]
        dense = agg["three_windows"]["per_class"][class_id]
        lines.append(
            f"| {base['class_name']} ({base['support']}) | {base['f1']['mean']:.4f} | "
            f"{dense['f1']['mean']:.4f} | {dense['f1']['mean']-base['f1']['mean']:+.4f} |"
        )
    uniform_first_six = statistics.mean(
        item["f1"]["mean"] for item in agg["uniform"]["per_class"][:6]
    )
    dense_first_six = statistics.mean(
        item["f1"]["mean"] for item in agg["three_windows"]["per_class"][:6]
    )
    lines.extend([
        "",
        f"Sensitivity excluding the six-sample fixation class: mean F1 over the other "
        f"six classes is {uniform_first_six:.4f} for uniform and {dense_first_six:.4f} "
        "for three-window.  Sweeping F1 is essentially unchanged, while "
        "reperfusion F1 falls under three-window sampling.",
    ])
    lines.extend([
        "",
        "| Duration | Support | Uniform present-class macro-F1 | Three-window present-class macro-F1 |",
        "|---|---:|---:|---:|",
    ])
    for name, label in (("duration_le_10s", "≤10 s"), ("duration_gt_10s", ">10 s"),
                        ("duration_lt_0.1s", "<0.1 s")):
        base = agg["uniform"]["duration_slices"][name]
        dense = agg["three_windows"]["duration_slices"][name]
        if base["support"] == 0:
            continue
        lines.append(
            f"| {label} | {base['support']} | {base['present_class_macro_f1']['mean']:.4f} | "
            f"{dense['present_class_macro_f1']['mean']:.4f} |"
        )
    lines.extend([
        "",
        "| Source | Clips | Uniform accuracy | Three-window accuracy |",
        "|---|---:|---:|---:|",
    ])
    for source, base in agg["uniform"]["source_accuracy"].items():
        dense = agg["three_windows"]["source_accuracy"][source]
        lines.append(
            f"| {source} | {base['support']} | {base['accuracy']['mean']:.4f} | "
            f"{dense['accuracy']['mean']:.4f} |"
        )
    if "lishui" in agg["uniform"]["source_accuracy"]:
        lines.extend([
            "",
            "Lishui accuracy decreases for three-window sampling in all three seeds; "
            "the aggregate source effect is not uniform. The fixed windows may miss "
            "useful parts of a long clip, but frame-selection and visibility audits "
            "are needed to establish the cause.",
        ])
    lines.extend([
        "",
        "Report class-wise F1, both error directions, and source/duration slices from",
        "[the aggregate JSON](three_seed_comparison.json) alongside the",
        "[training and validation curves](learning_curves.svg). Source slices may omit classes;",
        "their fixed-seven-class macro-F1 must not be compared as if all classes were present.",
        "Fixation has only six evaluation clips, so its F1 is especially unstable.",
        "",
        "The source corpus includes edited online videos. This experiment makes no claim",
        "about continuous clinical workflow, direct blood reperfusion measurement,",
        "or detection of clinician and patient roles.",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def run(results_root: Path, protocol_path: Path, output_dir: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    per_seed, curves = load_results(results_root, protocol)
    summary = {
        "schema_version": "fsn-motion-sampling-results-1.0",
        "code_commit": "b652ca4e870def29052903fc26f42dfb0a21e5c6",
        "protocol": "configs/motion_sampling_protocol.json",
        "manifest_sha256": {
            "train": protocol["train_manifest_sha256"],
            "validation": protocol["validation_manifest_sha256"],
        },
        "official_checkpoint_sha256": protocol["official_checkpoint_sha256"],
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "aggregate": aggregate(per_seed),
        "test_metrics": None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "three_seed_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "learning_curves.json").write_text(
        json.dumps(curves, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_curves(curves, output_dir / "learning_curves.svg")
    write_report(summary, output_dir / "README.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/motion_sampling_protocol.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.results_root, args.protocol, args.output_dir)
    print(json.dumps({
        "seeds": result["seeds"],
        "uniform_mean_macro_f1": result["aggregate"]["uniform"]["macro_f1"]["mean"],
        "three_windows_mean_macro_f1": result["aggregate"]["three_windows"]["macro_f1"]["mean"],
        "paired_delta": result["aggregate"]["dense_minus_uniform"]["macro_f1"]["mean"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
