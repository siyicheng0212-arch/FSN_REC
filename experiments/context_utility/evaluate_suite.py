"""Four matched groups, original validation and fixed synthetic challenges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.fsn_tcn.io import fresh_output, write_json, write_jsonl
from experiments.fsn_tcn.perturb import CONDITIONS, build_challenge
from .config import VARIANTS, build_base, load_config
from .data import load_bundle
from .evaluate import evaluate_chains, evaluate_same_checkpoint_gate_modes, summarize_comparison
from .train import load_trained, sealed_inputs


def evaluate_suite(bundle, config, checkpoint, seed, models_root, output, device="cpu"):
    if seed not in config["seeds"]:
        raise ValueError("evaluation seed was not predeclared")
    root = Path(models_root)
    for variant in VARIANTS:
        directory = root / variant
        result = json.loads((directory / "result.json").read_text())
        if result.get("variant") != variant or result.get("seed") != seed:
            raise ValueError("all four matching training results are required")
        if not (directory / "best.pt").is_file():
            raise ValueError("missing best checkpoint")
    conditions = {"original": bundle.chains["val"]}
    challenge_audits = {}
    for condition in CONDITIONS:
        conditions[condition], challenge_audits[condition] = build_challenge(
            bundle.chains["val"], bundle.metadata, condition=condition, seed=config["challenge_seed"])
    base, _ = build_base(config, bundle.index.global_dim + bundle.index.local_dim, checkpoint,
                         device, fingerprint=bundle.fingerprint)
    out = fresh_output(output)
    write_json(out / "protocol.json", {"schema": "fsn-context-utility-evaluation-v1",
               **sealed_inputs(bundle, config, checkpoint), "seed": seed,
               "challenge_audits": challenge_audits}, exclusive=True)
    initial_reports, initial_rows = {}, {}
    for condition, chains in conditions.items():
        initial_reports[condition], initial_rows[condition] = evaluate_chains(bundle, chains, base, device=device)
        write_jsonl(out / f"private_initial_{condition}.jsonl", initial_rows[condition])
    del base
    groups, prediction_rows, plan_seals = {}, {}, {}
    for variant in VARIANTS:
        model, state, _ = load_trained(root / variant / "best.pt", bundle, config, checkpoint,
                                       variant, seed, device)
        history = json.loads((root / variant / "history.json").read_text())
        if not history:
            raise ValueError("training history is empty")
        plan_seals[variant] = state["protocol"]["plan_sha256_per_epoch"]
        reports, private = {}, {}
        for condition, chains in conditions.items():
            reports[condition], private[condition] = evaluate_chains(bundle, chains, model, device=device)
            reports[condition]["paired_vs_initial_TCN"] = summarize_comparison(private[condition], initial_rows[condition])
            write_jsonl(out / f"private_{variant}_{condition}.jsonl", private[condition])
        gate_controls = {}
        if variant == "dynamic_aug":
            for condition, chains in conditions.items():
                gate_controls[condition], control_rows = evaluate_same_checkpoint_gate_modes(
                    bundle, chains, model, device=device, permutation_seed=config["challenge_seed"])
                for mode in ("unit", "permuted"):
                    write_jsonl(out / f"private_dynamic_{condition}_{mode}.jsonl", control_rows[mode])
        training = json.loads((root / variant / "result.json").read_text())
        groups[variant] = {"conditions": reports, "same_checkpoint_gate_controls": gate_controls,
                           "training": {key: training[key] for key in (
                               "best_epoch", "epochs_completed", "elapsed_seconds", "trainable_parameters",
                               "added_parameters", "A_trainable_parameters", "peak_cuda_allocated_bytes")},
                           "actual_optimizer_steps": sum(row["optimizer_steps"] for row in history),
                           "actual_original_clip_visits": sum(row["original_clip_visits"] for row in history),
                           "actual_anchor_view_visits": sum(row["anchor_view_visits"] for row in history)}
        prediction_rows[variant] = private
        del model
    if any(value != plan_seals["continue"] for value in plan_seals.values()):
        raise ValueError("the four groups did not use the same predeclared anchor/donor plans")
    for variant in VARIANTS:
        for condition in conditions:
            report = groups[variant]["conditions"][condition]
            report["paired_vs_continued_TCN"] = summarize_comparison(
                prediction_rows[variant][condition], prediction_rows["continue"][condition])
            report["paired_vs_augmented_TCN"] = summarize_comparison(
                prediction_rows[variant][condition], prediction_rows["aug"][condition])
            report["paired_vs_scalar_TCN"] = summarize_comparison(
                prediction_rows[variant][condition], prediction_rows["scalar_aug"][condition])
    summary = {"schema": "fsn-context-utility-safe-summary-v1", "seed": seed,
               "baseline_kind": config["baseline"]["kind"],
               "evaluation_role": "val823_development_not_independent_test",
               "initial_TCN": initial_reports, "groups": groups,
               "same_predeclared_plans": True, "A_frozen": True,
               "maximum_budget_and_stopping_rule_matched": True,
               "actual_training_exposure_may_differ_due_to_early_stopping": True,
               "strict_boundary_isolation_claimed": False,
               "true_edit_boundary_accuracy_claimed": False,
               "independent_test_read": False}
    write_json(out / "safe_summary.json", summary, exclusive=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-index", "source-protocol-dir", "config", "base-checkpoint", "models-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    bundle = load_bundle(args.feature_index, args.source_protocol_dir)
    evaluate_suite(bundle, config, args.base_checkpoint, args.seed, args.models_root, args.output, args.device)


if __name__ == "__main__":
    main()
