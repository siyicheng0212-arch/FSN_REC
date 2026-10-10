"""Aggregate-only tables/curves. No private identities, paths or raw logs."""
import csv
from html import escape
import json
from pathlib import Path

from experiments.input_evidence.worker import public_safe
from experiments.videomae_tcn.io import write_json, write_text
from .model import CONFIG, TASKS


def curve_svg(series, title):
    """Small scientific vector plot; no plotting/runtime dependency required."""
    colors = ("#2563eb", "#ea580c", "#059669", "#9333ea", "#be123c", "#475569")
    if not series or len(series) > 6 or any(len(v) != 60 for v in series.values()):
        raise ValueError("curve panels need one to six complete registered histories")
    pieces = ['<svg xmlns="http://www.w3.org/2000/svg" width="850" height="400" viewBox="0 0 850 400">',
              '<rect width="850" height="400" fill="white"/>',
              f'<text x="45" y="28" font-family="sans-serif" font-size="18">{escape(title)}</text>']
    for tick in (0, .25, .5, .75, 1):
        yy = 340 - tick * 280
        pieces += [f'<line x1="55" x2="600" y1="{yy}" y2="{yy}" stroke="#ddd"/>',
                   f'<text x="10" y="{yy + 5}" font-family="sans-serif" font-size="12">{tick:.2f}</text>']
    for i, (name, values) in enumerate(series.items()):
        if any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in values):
            raise ValueError("curves must contain finite full-seven-class F1")
        points = " ".join(f"{55 + j * 545 / 59:.2f},{340 - value * 280:.2f}" for j, value in enumerate(values))
        color = colors[i]
        pieces += [f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>',
                   f'<text x="620" y="{70 + i * 27}" fill="{color}" font-family="sans-serif" font-size="12">{escape(name)}</text>']
    pieces += ['<text x="55" y="365" font-family="sans-serif" font-size="12">1</text>',
               '<text x="580" y="365" font-family="sans-serif" font-size="12">60</text>',
               '<text x="270" y="389" font-family="sans-serif" font-size="12">epoch (development validation)</text>', '</svg>']
    return "\n".join(pieces) + "\n"


def aggregate(p):
    out = Path(p["output"])
    reports = {b: json.loads((Path(child["output"]) / "evaluation/public_safe_summary.json").read_text())
               for b, child in p["subplans"].items()}
    for b, report in reports.items():
        if report.get("training_completed") is not True or report.get("source_commit") != p["source_commit"] or \
                report.get("test_read") is not False or set(report["models"]) != set(TASKS):
            raise RuntimeError("aggregate requires every explicitly requested backbone/cell to complete")
        public_safe(report)
    result = {"schema": CONFIG["schema"], "source_commit": p["source_commit"], "training_completed": True,
              "test_read": False, "backbones": reports, "task_counts": p["task_counts"],
              "no_independent_test": True, "single_seed_frozen_A_head_experiments_only": True,
              "different_backbone_pretraining_views_optimizers_not_equal_budget": True,
              "cross_backbone_absolute_scores_are_not_temporal_module_effects": True}
    write_json(out / "public_safe_summary.json", public_safe(result))
    table = ["# Frozen-backbone temporal comparison", "", "Exploratory val823 development results; not an independent test.", "",
             "| Backbone | Model | F1 (%) | Accuracy (%) | A fixes / harms | New training |", "|---|---|---:|---:|---|---|"]
    with (out / "aggregate_metrics.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["backbone", "model", "macro_f1", "accuracy", "fixes_vs_A", "harms_vs_A", "best_epoch",
                         "epochs", "optimizer_steps", "parameters", "elapsed_seconds", "new_training"])
        for b, report in reports.items():
            for task, row in report["models"].items():
                m, r = row["metrics"], row["training_result"]
                paired = m["paired_vs_A"]
                writer.writerow([b, task, m["macro_f1"], m["accuracy"], paired["improved"], paired["worsened"],
                    r["best_epoch"], r["epochs_completed"], r["optimizer_steps"], row["parameters"]["parameters"],
                    r["resources"]["elapsed_seconds"], r["new_training"]])
                table.append(f'| {b} | {task} | {100*m["macro_f1"]:.3f} | {100*m["accuracy"]:.3f} | '
                             f'{paired["improved"]} / {paired["worsened"]} | {r["new_training"]} |')
    with (out / "per_class_metrics.csv").open("x", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["backbone", "model", "label_id", "precision", "recall", "f1", "support"])
        for b, report in reports.items():
            for task, row in report["models"].items():
                for c in row["metrics"]["per_class"]:
                    writer.writerow([b, task, *[c[k] for k in ("label_id", "precision", "recall", "f1", "support")]])
    table += ["", "## Interpretation limits", "", "- All seven classes are retained, including the six-sample class.",
              "- Different head capacity/loss/compute are reported, not declared identical.",
              "- The loss ablation changes stage supervision AND smoothing together; it cannot isolate smoothing alone.",
              "- Reused checkpoints are old completed runs, not newly trained results.",
              "- Recording bootstrap is conditional development uncertainty, not patient independence or confirmatory significance.",
              "- Masked A scores do not make the remaining visual evidence out-of-fold.",
              "- Synthetic challenges do not establish natural editing-boundary detection.",
              "- No further architecture/seed/visual training is automatically started."]
    for b, child in p["subplans"].items():
        histories = {}
        for task in TASKS:
            ref = p["reuse"][b].get(task)
            directory = Path(ref["run"]) / ref["variant"] if ref else Path(child["output"]) / task
            histories[task] = [r["val_macro_f1"] for r in json.loads((directory / "history.json").read_text())]
        groups = {"input": ("logits", "logits_aug", "visual", "visual_aug"),
                  "heads": ("logits", "clip_logits", "mstcnpp_logits", "asformer_logits"),
                  "masked_heads": ("visual", "clip_visual", "mstcnpp_visual", "asformer_visual"),
                  "loss": ("mstcnpp_logits", "mstcnpp_logits_final_ce", "asformer_logits", "asformer_logits_final_ce")}
        table += ["", f"## {b} development curves", ""]
        for name, tasks in groups.items():
            filename = b + "_" + name + "_curve.svg"
            write_text(out / filename, curve_svg({t: histories[t] for t in tasks}, b + " / " + name))
            table += [f"![{b} {name}]({filename})", ""]
    write_text(out / "REPORT.md", "\n".join(table) + "\n")
