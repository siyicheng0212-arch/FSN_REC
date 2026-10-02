"""Read-only plan by default; checked four-GPU runs require --execute."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import statistics

from experiments.relation.data import sha256_file
from experiments.relation.run_suite import assert_idle_gpus
from .config import STRATEGIES, baseline_source_hashes, load_config
from .hard_negatives import build_hard_negatives
from .io import code_hashes, fresh_output, load_bundle, write_json
from .perturb import CONDITIONS, build_challenge

ROOT = Path(__file__).resolve().parents[2]


def aggregate_seeds(summaries):
    """Report every declared run and paired gains; one seed has no estimated SD."""
    groups = {}
    for strategy in STRATEGIES:
        groups[strategy] = {}
        for condition in ("original", *CONDITIONS):
            runs = [summary["groups"][strategy]["conditions"][condition] for summary in summaries]
            values = {
                "macro_f1": [run["metrics"]["macro_f1"] for run in runs],
                "accuracy": [run["metrics"]["accuracy"] for run in runs],
                "macro_f1_gain_vs_A": [run["metrics"]["macro_f1"] - run["A_metrics"]["macro_f1"] for run in runs],
                "accuracy_gain_vs_A": [run["metrics"]["accuracy"] - run["A_metrics"]["accuracy"] for run in runs],
            }
            groups[strategy][condition] = {name: {"values": array, "mean": statistics.mean(array),
                "sample_std": statistics.stdev(array) if len(array) > 1 else None} for name, array in values.items()}
    return {"count": len(summaries), "fixed_A_not_full_visual_seed_variation": True, "groups": groups}


def repository_state():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    return {"commit": commit, "clean": not dirty}


def gpu_indices(value):
    try:
        values = [int(x) for x in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("four distinct physical GPU indices required") from error
    if len(values) != 4 or len(set(values)) != 4 or min(values) < 0:
        raise argparse.ArgumentTypeError("use four distinct GPUs, e.g. 0,1,2,3; not DDP")
    return values


def build_commands(args, config):
    out = Path(args.output).resolve()
    python = [str(Path(args.python).resolve()), "-m"]
    common = ["--feature-index", args.feature_index, "--source-protocol-dir", args.source_protocol_dir,
              "--config", args.config]
    commands = {
        "unit_tests": python + ["pytest", "-q", "-p", "no:cacheprovider", "tests/test_fsn_tcn_model.py", "tests/test_fsn_tcn_negatives.py",
                                 "tests/test_fsn_tcn_evaluate.py", "tests/test_fsn_tcn_suite.py"],
        "raw_cache_smoke": python + ["experiments.relation.smoke", "--manifest-dir", args.manifest_dir,
            "--cache-root", args.cache_root, "--original-checkpoint", args.original_checkpoint,
            "--source-protocol-dir", args.source_protocol_dir, "--output", str(out / "raw_cache_smoke.json"),
            "--device", f"cuda:{args.gpus[0]}"],
        "tcn_smoke": python + ["experiments.fsn_tcn.smoke"] + common + ["--raw-cache-report", str(out / "raw_cache_smoke.json"),
            "--output", str(out / "tcn_smoke.json"), "--device", f"cuda:{args.gpus[0]}"],
    }
    for seed in config["seeds"]:
        seed_root = out / f"seed_{seed}"
        rdir = seed_root / "relation"
        training = python + ["experiments.fsn_tcn.train"]
        smoke = ["--smoke-report", str(out / "tcn_smoke.json")]
        commands[f"relation_{seed}"] = training + ["relation"] + common + smoke + ["--seed", str(seed),
            "--output", str(rdir), "--device", f"cuda:{args.gpus[0]}"]
        for strategy, gpu in zip(STRATEGIES, args.gpus):
            commands[f"{strategy}_{seed}"] = training + ["tcn"] + common + smoke + ["--seed", str(seed),
                "--strategy", strategy, "--relation-dir", str(rdir), "--output", str(seed_root / strategy),
                "--device", f"cuda:{gpu}"]
        commands[f"evaluate_{seed}"] = python + ["experiments.fsn_tcn.evaluate_suite"] + common + [
            "--seed", str(seed), "--relation-dir", str(rdir), "--models-root", str(seed_root),
            "--output", str(seed_root / "evaluation"), "--device", f"cuda:{args.gpus[0]}"]
    return commands


def prepare_plan(args):
    # Worker cwd is fixed to ROOT; preserve the caller's path interpretation.
    for name in ("python", "feature_index", "source_protocol_dir", "manifest_dir", "cache_root",
                 "original_checkpoint", "config", "output"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    output = Path(args.output).resolve()
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError("use a fresh private output outside the code worktree")
    for path in (args.python, args.feature_index, args.config, args.original_checkpoint):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if not Path(args.cache_root).is_dir():
        raise FileNotFoundError(args.cache_root)
    config = load_config(args.config)
    bundle = load_bundle(args.feature_index, args.source_protocol_dir)
    if sha256_file(args.original_checkpoint) != bundle.index.checkpoint_sha256:
        raise ValueError("raw-cache A checkpoint differs from frozen feature export")
    manifests = {s: sha256_file(Path(args.manifest_dir) / f"{s}.jsonl") for s in ("train", "val")}
    if manifests != bundle.fingerprint["manifest_sha256"]:
        raise ValueError("raw-cache manifests differ from frozen full7372/823")
    # Check feasibility before any training; never discover a missing control after a favorable result.
    negatives = {str(seed): build_hard_negatives(bundle.metadata, bundle.edges, seed=seed,
                                                ratio=config["hard_negative_ratio"])[1] for seed in config["seeds"]}
    challenges = {condition: build_challenge(bundle.chains["val"], bundle.metadata,
                                             condition=condition, seed=config["challenge_seed"])[1]
                  for condition in CONDITIONS}
    state = repository_state()
    if args.expected_commit and state["commit"] != args.expected_commit:
        raise ValueError("HEAD differs from the specified fixed commit")
    if args.execute and not state["clean"]:
        raise ValueError("execution requires clean committed source; do not pull during a run")
    return {"schema": "fsn-tcn-minimal-suite-v1", "training_completed": False,
            "repository": state, "fingerprint": bundle.fingerprint, "config": config,
            "config_sha256": sha256_file(args.config), "source_sha256": code_hashes(),
            "external_baseline_sha256": baseline_source_hashes(config),
            "original_checkpoint_sha256": bundle.index.checkpoint_sha256,
            "arguments": vars(args), "hard_negative_audits": negatives, "challenge_audits": challenges,
            "commands": build_commands(args, config), "GPU_mapping": dict(zip(STRATEGIES, args.gpus)),
            "A_fixed_across_all_seeds": True, "DDP": False,
            "old_local_tcn_compatibility_verified": False,
            "baseline_kind": config["baseline"]["kind"],
            "note": "reference is a newly explicit baseline; external factory must supply the historical two-input scorer contract"}


def validate_runtime(plan):
    args = plan["arguments"]
    if repository_state() != plan["repository"] or code_hashes() != plan["source_sha256"]:
        raise RuntimeError("source/git changed during the run")
    if sha256_file(args["config"]) != plan["config_sha256"]:
        raise RuntimeError("config changed after the threshold/seeds were sealed")
    if baseline_source_hashes(plan["config"]) != plan["external_baseline_sha256"]:
        raise RuntimeError("external baseline code changed")
    if sha256_file(args["original_checkpoint"]) != plan["original_checkpoint_sha256"]:
        raise RuntimeError("Original checkpoint changed")
    for split, digest in plan["fingerprint"]["manifest_sha256"].items():
        if sha256_file(Path(args["manifest_dir"]) / f"{split}.jsonl") != digest:
            raise RuntimeError("frozen manifest changed")
    if sha256_file(args["feature_index"]) != plan["fingerprint"]["feature_index_sha256"]:
        raise RuntimeError("frozen feature index changed")
    directory = Path(args["source_protocol_dir"])
    if sha256_file(directory / "audit.json") != plan["fingerprint"]["source_audit_sha256"]:
        raise RuntimeError("source audit changed")
    for name, digest in plan["fingerprint"]["source_files_sha256"].items():
        if sha256_file(directory / name) != digest:
            raise RuntimeError("sealed source chains/metadata/edges changed")


def execute_plan(plan, run=None):
    out = fresh_output(plan["arguments"]["output"])
    logs = out / "launcher_logs"
    logs.mkdir()
    write_json(out / "run_protocol.json", plan, exclusive=True)
    write_json(out / "frozen_config.json", plan["config"], exclusive=True)
    (out / ".launch_once").write_text(plan["repository"]["commit"] + "\n")
    (out / "run_commit.txt").write_text(plan["repository"]["commit"] + "\n")
    (logs / "pids.tsv").write_text("stage\tpid\n")
    (logs / "exit_codes.tsv").write_text("stage\texit_code\n")
    lock, outcomes = threading.Lock(), {}

    def launch(stage):
        if run is None:
            validate_runtime(plan)
        print(json.dumps({"stage": stage, "status": "starting"}), flush=True)
        if run is not None:  # test injection is unavailable through the CLI
            code = run(stage, plan["commands"][stage])
        else:
            env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            env.update({key: plan["GPU_environment"][key] for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")})
            with (logs / f"{stage}.log").open("x") as log:
                process = subprocess.Popen(plan["commands"][stage], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                with lock, (logs / "pids.tsv").open("a") as stream:
                    stream.write(f"{stage}\t{process.pid}\n")
                code = process.wait()
        with lock:
            outcomes[stage] = code
            with (logs / "exit_codes.tsv").open("a") as stream:
                stream.write(f"{stage}\t{code}\n")
        if code != 0:
            raise RuntimeError(f"{stage} failed with exit {code}; no automatic restart or protocol change")
        return code

    try:
        for stage in ("unit_tests", "raw_cache_smoke", "tcn_smoke"):
            launch(stage)
        if run is None:
            report = json.loads((out / "tcn_smoke.json").read_text())
            if (report.get("all_passed") is not True or report.get("real_cuda_tcn_smoke_completed") is not True
                    or report.get("fingerprint") != plan["fingerprint"]
                    or report.get("baseline_source_sha256") != plan["external_baseline_sha256"]
                    or report.get("source_sha256") != plan["source_sha256"]):
                raise RuntimeError("TCN CUDA preflight was not verified")
        summaries = []
        for seed in plan["config"]["seeds"]:
            launch(f"relation_{seed}")
            # Exactly four independent jobs, each with one distinct GPU.
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(launch, f"{strategy}_{seed}") for strategy in STRATEGIES]
                errors = []
                for future in futures:
                    try:
                        future.result()
                    except Exception as error:
                        errors.append(error)
                if errors:
                    raise errors[0]
            launch(f"evaluate_{seed}")
            if run is None:
                path = out / f"seed_{seed}" / "evaluation" / "public_safe_summary.json"
                summaries.append(json.loads(path.read_text()))
        if run is None:
            write_json(out / "public_safe_all_seeds.json", {"schema": "fsn-tcn-all-declared-seeds-v1",
                       "evaluation_role": "val823_development_not_independent_test",
                       "A_fixed": True, "declared_seeds": plan["config"]["seeds"], "results": summaries,
                       "aggregate": aggregate_seeds(summaries)}, exclusive=True)
        write_json(out / "suite_result.json", {"status": "complete", "training_completed": True,
                   "exit_codes": outcomes, "seeds": plan["config"]["seeds"]}, exclusive=True)
    except Exception as error:
        write_json(out / "suite_result.json", {"status": "failed", "training_completed": False,
                   "exit_codes": outcomes, "error_type": type(error).__name__, "error": str(error)}, exclusive=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-index", "source-protocol-dir", "manifest-dir", "cache-root", "original-checkpoint", "config", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", type=gpu_indices, default=[0, 1, 2, 3])
    parser.add_argument("--expected-commit")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = prepare_plan(args)
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    plan["GPU_environment"] = assert_idle_gpus(args.gpus)
    execute_plan(plan)


if __name__ == "__main__":
    main()
