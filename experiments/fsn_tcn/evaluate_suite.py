"""Four paired groups on original val823 and sealed synthetic-only challenges."""
import argparse
import json
from pathlib import Path

import torch

from experiments.relation.data import sha256_file
from .config import STRATEGIES, baseline_source_hashes, build_base, load_config
from .evaluate import evaluate_chains
from .io import code_hashes, fresh_output, load_bundle, write_json, write_jsonl
from .perturb import CONDITIONS, build_challenge
from .train import load_relation, score_pairs


def evaluate_suite(bundle, config, seed, relation_dir, models_root, output, device):
    relation_dir, models_root = Path(relation_dir), Path(models_root)
    relation, saved = load_relation(relation_dir / "best.pt", bundle, device)
    if saved["seed"] != seed or saved["protocol"]["config"] != config:
        raise ValueError("R evaluation configuration differs from training")
    # Challenge generator uses a predeclared fixed seed across every model/seed.
    conditions = {"original": (bundle.chains["val"], {"synthetic": False})}
    for condition in CONDITIONS:
        conditions[condition] = build_challenge(bundle.chains["val"], bundle.metadata,
                                                condition=condition, seed=config["challenge_seed"])
    pairs = {(left, right) for chains, _ in conditions.values() for chain in chains
             for left, right, valid in zip(chain["ordered_clip_ids"], chain["ordered_clip_ids"][1:], chain["eligible"])
             if valid}
    scores = score_pairs(relation, bundle.index, pairs, device, config["relation"]["batch_size"])
    out = fresh_output(output)
    write_jsonl(out / "relation_scores_private.jsonl", [{"left_clip_id": pair[0], "right_clip_id": pair[1], "score": value}
                                                       for pair, value in sorted(scores.items())])
    for condition, (chains, audit) in conditions.items():
        write_jsonl(out / f"chains_{condition}_private.jsonl", chains)
        write_json(out / f"audit_{condition}.json", audit, exclusive=True)
    reports = {}
    for strategy in STRATEGIES:
        checkpoint = models_root / strategy / "best.pt"
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if (state["fingerprint"] != bundle.fingerprint or state["seed"] != seed
                or state["strategy"] != strategy or state["config"] != config
                or state["protocol"]["code_sha256"] != code_hashes()
                or state["protocol"]["baseline_source_sha256"] != baseline_source_hashes(config)
                or state["protocol"]["R_best_sha256"] != sha256_file(relation_dir / "best.pt")):
            raise ValueError("TCN checkpoint belongs to another experiment")
        training_result = json.loads((models_root / strategy / "result.json").read_text())
        if training_result["best_sha256"] != sha256_file(checkpoint):
            raise ValueError("TCN checkpoint differs from completed training result")
        base = build_base(config, bundle.index.global_dim + bundle.index.local_dim, device)
        base.load_state_dict(state["model"], strict=True)
        by_condition = {}
        for condition, (chains, _) in conditions.items():
            aggregate, predictions = evaluate_chains(bundle, chains, base, scores, strategy=strategy,
                                                     threshold=config["threshold"], seed=seed, device=device)
            aggregate.update(best_epoch=state["epoch"], seed=seed, strategy=strategy,
                             trainable_TCN_parameters=training_result["trainable_TCN_parameters"],
                             training_elapsed_seconds=training_result["elapsed_seconds"])
            by_condition[condition] = aggregate
            write_jsonl(out / f"{strategy}_{condition}_predictions_private.jsonl", predictions)
        # Same weights, different cut positions: diagnostic, not independently trained controls.
        switch_audit = {}
        for switch in STRATEGIES:
            aggregate, _ = evaluate_chains(bundle, bundle.chains["val"], base, scores, strategy=switch,
                                           threshold=config["threshold"], seed=seed, device=device)
            switch_audit[switch] = aggregate
        reports[strategy] = {"conditions": by_condition, "same_checkpoint_gate_switch_audit": switch_audit}
    summary = {"schema": "fsn-tcn-public-safe-summary-v1", "seed": seed,
               "evaluation_role": "val823_development_not_independent_test",
               "synthetic_challenges_are_not_real_edit_boundary_ground_truth": True,
               "A_is_fixed_across_seeds": True, "threshold": config["threshold"],
               "threshold_calibrated": False, "challenge_seed": config["challenge_seed"],
               "challenge_audits": {condition: audit for condition, (_, audit) in conditions.items()},
               "groups": reports}
    write_json(out / "public_safe_summary.json", summary, exclusive=True)
    write_json(out / "result.json", {"status": "complete", "seed": seed,
                                      "summary_sha256": sha256_file(out / "public_safe_summary.json")}, exclusive=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-index", "source-protocol-dir", "config", "relation-dir", "models-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.seed not in config["seeds"]:
        parser.error("undeclared seed")
    bundle = load_bundle(args.feature_index, args.source_protocol_dir)
    evaluate_suite(bundle, config, args.seed, args.relation_dir, args.models_root, args.output, args.device)


if __name__ == "__main__":
    main()
