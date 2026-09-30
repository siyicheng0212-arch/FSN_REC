"""Build privacy-safe semantic-window aggregates and SVG curves from private JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


RUNS = (("corroborated", 42), ("corroborated", 14),
        ("corroborated", 15), ("plain_mean", 42))
COLORS = ("#087e8b", "#5c6bc0", "#c75c18", "#2f7d32")


def _polyline(values: list[float], left: float, top: float, width: float,
              height: float, minimum: float, maximum: float, color: str) -> str:
    count = len(values)
    points = " ".join(
        f"{left + index * width / max(count - 1, 1):.1f},"
        f"{top + height - (value - minimum) * height / (maximum - minimum):.1f}"
        for index, value in enumerate(values)
    )
    return f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>'


def _svg(curves: list[dict], baseline_f1: float) -> str:
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="940" height="560" viewBox="0 0 940 560">',
        '<rect width="940" height="560" fill="white"/>',
        '<text x="55" y="35" font-family="sans-serif" font-size="20">Semantic window head training curves</text>',
        '<text x="55" y="56" font-family="sans-serif" font-size="12">823-clip development validation; shared frozen Original-42 backbone</text>',
    ]
    panels = (("Development validation macro F1", "validation_macro_f1", 0.75, 0.79, 85),
              ("Head training loss", "train_loss", 0.09, 0.15, 310))
    for title, field, low, high, top in panels:
        left, width, height = 80, 760, 155
        pieces.append(f'<text x="80" y="{top - 13}" font-family="sans-serif" font-size="14">{escape(title)}</text>')
        pieces.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="none" stroke="#777"/>')
        for tick in range(5):
            y = top + height - tick * height / 4
            value = low + tick * (high - low) / 4
            pieces.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" stroke="#eee"/>')
            pieces.append(f'<text x="28" y="{y + 4:.1f}" font-family="sans-serif" font-size="11">{value:.3f}</text>')
        for curve, color in zip(curves, COLORS):
            values = [row[field] for row in curve["history"]]
            # All runs share the same epoch scale, so a shorter trace ends earlier.
            points = " ".join(
                f"{left + (row['epoch'] - 1) * width / 16:.1f},"
                f"{top + height - (row[field] - low) * height / (high - low):.1f}"
                for row in curve["history"]
            )
            pieces.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
        if field == "validation_macro_f1":
            y = top + height - (baseline_f1 - low) * height / (high - low)
            pieces.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" stroke="#333" stroke-dasharray="5 4"/>')
        pieces.append(f'<text x="80" y="{top + height + 19}" font-family="sans-serif" font-size="11">1</text>')
        pieces.append(f'<text x="817" y="{top + height + 19}" font-family="sans-serif" font-size="11">17 epochs</text>')
    for index, (curve, color) in enumerate(zip(curves, COLORS)):
        x = 80 + index * 200
        pieces.append(f'<line x1="{x}" y1="530" x2="{x + 25}" y2="530" stroke="{color}" stroke-width="3"/>')
        pieces.append(f'<text x="{x + 31}" y="534" font-family="sans-serif" font-size="12">{escape(curve["mode"])} {curve["seed"]}</text>')
    pieces.append('</svg>')
    return "\n".join(pieces) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.private_dir / "semantic_final_audit.json").read_text())
    curves = []
    manifest_hashes = set()
    code_commits = set()
    baseline_hashes = set()
    for mode, seed in RUNS:
        path = args.private_dir / f"semantic_{mode}_{seed}_result.json"
        result = json.loads(path.read_text())
        if result.get("test_metrics") is not None:
            raise ValueError("independent test metrics unexpectedly present")
        if result["mode"] != mode or result["seed"] != seed:
            raise ValueError("result identity mismatch")
        manifest_hashes.add(tuple(result["split_audit"]["manifest_sha256"][key]
                                  for key in ("train", "val")))
        code_commits.add(result["code_commit"])
        baseline_hashes.add(result["baseline_checkpoint_sha256"])
        key = f"{mode}_{seed}"
        if abs(result["best_val_macro_f1"] - audit["models"][key]["classification"]["macro_f1"]) > 1e-6:
            raise ValueError("saved best and re-inference disagree")
        curves.append({
            "mode": mode, "seed": seed, "best_epoch": result["best_epoch"],
            "best_validation_macro_f1": result["best_val_macro_f1"],
            "trainable_parameters": result["trainable_parameters"],
            "train_seconds": sum(row["train_seconds"] for row in result["history"]),
            "history": [{"epoch": row["epoch"], "train_loss": row["train_loss"],
                         "validation_macro_f1": row["val_metrics"]["all"]["macro_f1"]}
                        for row in result["history"]],
        })
    if not (len(manifest_hashes) == len(code_commits) == len(baseline_hashes) == 1):
        raise ValueError("four runs do not share one frozen protocol")
    audit["manifest_sha256"] = dict(zip(("train", "val"), next(iter(manifest_hashes))))
    audit["code_commit"] = next(iter(code_commits))
    audit["baseline_checkpoint_sha256"] = next(iter(baseline_hashes))
    audit["metric_note"] = "Best epoch selected on the same 823-clip development validation; not an independent test."
    (args.output_dir / "final_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "learning_curves.json").write_text(
        json.dumps({"schema_version": "fsn-semantic-learning-curves-1.0",
                    "selection_set": audit["selection_set"], "curves": curves},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "learning_curves.svg").write_text(
        _svg(curves, audit["models"]["original"]["classification"]["macro_f1"]), encoding="utf-8")


if __name__ == "__main__":
    main()
