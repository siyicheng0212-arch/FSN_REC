#!/usr/bin/env python3
"""Publish epoch-level learning curves without private training artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_run(name: str, path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("test_metrics") is not None:
        raise ValueError(f"independent test metrics must be null for {name}")
    points = [
        {
            "phase": row["phase"],
            "phase_epoch": int(row["epoch"]),
            "overall_epoch": index,
            "train_loss": float(row["train_loss"]),
            "validation_macro_f1": float(row["val_metrics"]["all"]["macro_f1"]),
            "learning_rate": float(row["lr"]),
        }
        for index, row in enumerate(result["history"], 1)
    ]
    if not points:
        raise ValueError(f"empty training history for {name}")
    best = max(points, key=lambda point: point["validation_macro_f1"])
    if abs(best["validation_macro_f1"] - result["best_val_macro_f1"]) > 1e-8:
        raise ValueError(f"best validation score differs from history for {name}")
    return {
        "name": name,
        "variant": result["variant"],
        "seed": int(result["seed"]),
        "manifest_sha256": result["split_audit"]["manifest_sha256"],
        "best_epoch": result["best_epoch"],
        "best_validation_macro_f1": best["validation_macro_f1"],
        "epochs": points,
        "test_metrics": None,
    }


def render_chart(runs: list[dict], output_base: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(13, 4.5))
    for run in runs:
        points = run["epochs"]
        x = [point["overall_epoch"] for point in points]
        loss_ax.plot(x, [point["train_loss"] for point in points], label=run["name"])
        f1_ax.plot(
            x,
            [point["validation_macro_f1"] for point in points],
            label=run["name"],
        )
        best = max(points, key=lambda point: point["validation_macro_f1"])
        f1_ax.scatter(
            best["overall_epoch"], best["validation_macro_f1"], s=26, zorder=3
        )
    warmup_count = sum(point["phase"] == "head_warmup" for point in runs[0]["epochs"])
    for ax in (loss_ax, f1_ax):
        ax.axvline(warmup_count + 0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Overall epoch (warm-up first)")
        ax.grid(alpha=0.2)
    loss_ax.set_ylabel("Train native multi-branch loss")
    f1_ax.set_ylabel("Validation macro-F1")
    f1_ax.set_ylim(0, 1)
    f1_ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=160)
    fig.savefig(output_base.with_suffix(".svg"))
    plt.close(fig)


def parse_named_path(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
        if not name:
            raise ValueError("empty name")
        return name, Path(path)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected NAME=PATH") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_named_path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = [load_run(name, path) for name, path in args.run]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "fsn-public-learning-curves-1.0",
        "protocol_warning": "823 held-out clips are validation; no independent test exists",
        "loss_definition": (
            "Native weighted Uni-AdaFocus multi-branch training loss, not final-head CE"
        ),
        "runs": runs,
        "test_metrics": None,
    }
    (output / "learning_curves.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    by_seed = [run for run in runs if run["variant"] == "original" and run["name"].startswith("original_seed")]
    comparison = [run for run in runs if run["seed"] == 42 and run["name"] in {"original_seed42", "fsn_v1_seed42", "fsn_v2_seed42", "active_mean_seed42"}]
    if by_seed:
        render_chart(by_seed, output / "original_three_seeds", "Original Uni-AdaFocus: three seeds")
    if comparison:
        render_chart(comparison, output / "seed42_model_comparison", "Model comparison: seed 42")


if __name__ == "__main__":
    main()
