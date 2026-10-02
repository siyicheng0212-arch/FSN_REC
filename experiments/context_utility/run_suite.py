"""Read-only planning, then an explicitly authorized two/four-GPU utility suite.

Every worker sees one physical GPU UUID as cuda:0. A failed wave is allowed to
finish naturally; it blocks all later waves and evaluation. No automatic retry,
checkpoint resume, git update, uploads, or process termination is performed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading

from experiments.relation.data import sha256_file
from experiments.relation.run_suite import assert_idle_gpus
from experiments.fsn_tcn.io import fresh_output, write_json
from experiments.fsn_tcn.perturb import CONDITIONS, build_challenge
from .augmentation import build_epoch_plan
from .data import load_bundle


ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("continue", "aug", "scalar_aug", "dynamic_aug")


def repository_state():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    return {"commit": commit, "clean": not dirty}


def gpu_indices(value):
    try:
        values = [int(part) for part in value.split(",")]
    except (ValueError, AttributeError) as error:
        raise argparse.ArgumentTypeError("use two or four distinct physical GPU indices") from error
    if len(values) not in (2, 4) or len(set(values)) != len(values) or min(values) < 0:
        raise argparse.ArgumentTypeError("use 0,1 or 0,1,2,3; these are independent experiments, not DDP")
    return values


def wave_schedule(gpus):
    if len(gpus) not in (2, 4) or len(set(gpus)) != len(gpus) or any(type(g) is not int or g < 0 for g in gpus):
        raise ValueError("two or four distinct physical GPUs are required")
    return [[{"variant": variant, "physical_gpu": gpu} for variant, gpu in zip(VARIANTS[start:], gpus)]
            for start in range(0, len(VARIANTS), len(gpus))]


def build_commands(args, config):
    """Build commands only; do not query GPUs, import CUDA, or create outputs."""
    out = Path(args.output).resolve()
    command = [str(Path(args.python).resolve()), "-m"]
    common = ["--feature-index", args.feature_index, "--source-protocol-dir", args.source_protocol_dir,
              "--config", args.config, "--base-checkpoint", args.base_checkpoint]
    tests = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "tests").glob("test_context_utility*.py"))
    if not tests:
        raise ValueError("context utility test files are missing")
    result = {"unit_tests": command + ["pytest", "-q", "-p", "no:cacheprovider", *tests],
              "cuda_smoke": command + ["experiments.context_utility.smoke", *common,
                  "--output", str(out / "cuda_smoke.json"), "--device", "cuda:0"]}
    for seed in config["seeds"]:
        seed_root = out / f"seed_{seed}"
        for variant in VARIANTS:
            result[f"{variant}_{seed}"] = command + ["experiments.context_utility.train", *common,
                "--seed", str(seed), "--variant", variant, "--smoke-report", str(out / "cuda_smoke.json"),
                "--output", str(seed_root / variant), "--device", "cuda:0"]
        result[f"evaluate_{seed}"] = command + ["experiments.context_utility.evaluate_suite", *common,
            "--seed", str(seed), "--models-root", str(seed_root), "--output", str(seed_root / "evaluation"),
            "--device", "cuda:0"]
    return result


def prepare_plan(args):
    from .config import baseline_source_hashes, build_base, code_hashes, load_config
    for name in ("python", "feature_index", "source_protocol_dir", "config", "base_checkpoint", "output"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("output already exists; preserve it and choose a fresh directory")
    if output.is_relative_to(ROOT):
        raise ValueError("private outputs must be outside the source worktree")
    for name in ("python", "feature_index", "config", "base_checkpoint"):
        if not Path(getattr(args, name)).is_file():
            raise FileNotFoundError(getattr(args, name))
    config = load_config(args.config)
    if config["baseline"]["kind"] == "legacy":
        paths = [config["baseline"]["legacy_audit"], *config["baseline"]["source_dependencies"]]
        if any(not Path(path).is_absolute() for path in paths):
            raise ValueError("legacy audit/dependency paths must be absolute so worker cwd cannot change their meaning")
    bundle = load_bundle(args.feature_index, args.source_protocol_dir)
    first = next(iter(bundle.metadata))
    arrays = bundle.index.read(first)
    input_dim = arrays["global_tokens"].shape[1] + arrays["local_tokens"].shape[1]
    # Strict actual-base loading and its audit belong to the config adapter. A
    # missing legacy contract must fail here, before outputs or GPU inspection.
    base, base_audit = build_base(config, input_dim, args.base_checkpoint, device="cpu", fingerprint=bundle.fingerprint)
    base_parameters = sum(parameter.numel() for parameter in base.parameters())
    from .model import ContextUtilityTCN
    parameter_counts = {}
    for variant, mode in zip(VARIANTS, ("plain", "plain", "scalar", "dynamic")):
        adapted = ContextUtilityTCN(deepcopy(base), input_dim, config["baseline"]["temporal_paths"],
                                   mode=mode, dim=config["module"]["dim"], gate_seed=config["seeds"][0])
        parameter_counts[variant] = {"added": adapted.added_parameter_count(),
            "trainable": sum(p.numel() for p in adapted.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in adapted.parameters())}
    augmentation = config["augmentation"]
    audits = {}
    for seed in config["seeds"]:
        _, audit = build_epoch_plan(bundle, seed=seed, epoch=1,
            anchors_per_segment=augmentation["anchors_per_segment"], max_neighbors=augmentation["max_neighbors"])
        if audit["actual_replacements"] <= 0:
            raise ValueError("no train-only clinical donor replacements; do not silently alter the protocol")
        audits[str(seed)] = audit
    challenges = {condition: build_challenge(bundle.chains["val"], bundle.metadata,
                    condition=condition, seed=config["challenge_seed"])[1] for condition in CONDITIONS}
    state = repository_state()
    if args.expected_commit and state["commit"] != args.expected_commit:
        raise ValueError("HEAD differs from the specified fixed commit")
    if args.execute and not state["clean"]:
        raise ValueError("execution requires clean committed source; preserve changes and commit first")
    return {"schema": "fsn-context-utility-suite-v1", "training_completed": False,
            "repository": state, "arguments": vars(args), "config": config,
            "config_sha256": sha256_file(args.config), "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
            "fingerprint": bundle.fingerprint, "source_sha256": code_hashes(),
            "external_baseline_sha256": baseline_source_hashes(config), "base_load_audit": base_audit,
            "legacy_audit_sha256": base_audit["legacy_audit"]["sha256"] if base_audit.get("legacy_audit") else None,
            "base_parameters": base_parameters, "parameter_counts": parameter_counts, "first_epoch_plan_audits": audits,
            "challenge_audits": challenges, "commands": build_commands(args, config),
            "waves": wave_schedule(args.gpus),
            "GPU_mapping": {job["variant"]: job["physical_gpu"] for wave in wave_schedule(args.gpus) for job in wave},
            "DDP": False, "A_frozen": True, "evaluation_role": "val823_development_not_independent_test",
            "baseline_kind": config["baseline"]["kind"],
            "note": "one suite uses one actual base checkpoint; reference and legacy are separate experiments"}


def validate_runtime(plan):
    from .config import baseline_source_hashes, code_hashes
    args = plan["arguments"]
    if repository_state() != plan["repository"] or code_hashes() != plan["source_sha256"]:
        raise RuntimeError("source/git changed during execution; no automatic update or restart")
    if sha256_file(args["config"]) != plan["config_sha256"]:
        raise RuntimeError("config changed after planning")
    if baseline_source_hashes(plan["config"]) != plan["external_baseline_sha256"]:
        raise RuntimeError("external baseline code changed after planning")
    if sha256_file(args["base_checkpoint"]) != plan["base_checkpoint_sha256"]:
        raise RuntimeError("base checkpoint changed after planning")
    legacy_audit = plan["base_load_audit"].get("legacy_audit")
    if legacy_audit and sha256_file(legacy_audit["path"]) != legacy_audit["sha256"]:
        raise RuntimeError("verified legacy audit changed after planning")
    fingerprint = plan["fingerprint"]
    if sha256_file(args["feature_index"]) != fingerprint["feature_index_sha256"]:
        raise RuntimeError("frozen feature index changed after planning")
    directory = Path(args["source_protocol_dir"])
    if sha256_file(directory / "audit.json") != fingerprint["source_audit_sha256"]:
        raise RuntimeError("source audit changed after planning")
    for name, digest in fingerprint["source_files_sha256"].items():
        if sha256_file(directory / name) != digest:
            raise RuntimeError("sealed source metadata/chains/edges changed after planning")


def worker_environment(plan, physical_gpu):
    """Physical GPU UUID is isolated, so every child intentionally uses cuda:0."""
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    if physical_gpu is not None:
        inventory = plan["GPU_environment"]["physical_gpu_uuid"]
        # JSON serialization can turn integer dictionary keys into strings.
        uuid = inventory.get(physical_gpu, inventory.get(str(physical_gpu)))
        if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
            raise RuntimeError("missing physical GPU UUID mapping")
        env.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    else:
        env.update(CUDA_VISIBLE_DEVICES="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    return env


def _verify_smoke(plan, path):
    report = json.loads(Path(path).read_text())
    checks = {"all_passed": True, "real_cuda_context_utility_smoke_completed": True,
              "fingerprint": plan["fingerprint"], "config_sha256": plan["config_sha256"],
              "base_checkpoint_sha256": plan["base_checkpoint_sha256"], "source_sha256": plan["source_sha256"],
              "external_baseline_sha256": plan["external_baseline_sha256"],
              "legacy_audit_sha256": plan["legacy_audit_sha256"]}
    if any(report.get(key) != value for key, value in checks.items()):
        raise RuntimeError("real CUDA smoke report does not match this sealed suite")


def _verify_group_results(output, seed):
    """All four exit codes alone are insufficient: require parseable results."""
    for variant in VARIANTS:
        directory = Path(output) / f"seed_{seed}" / variant
        path = directory / "result.json"
        result = json.loads(path.read_text())
        if not isinstance(result, dict) or result.get("status") in {"failed", "error"}:
            raise RuntimeError(f"{variant} has no valid completed result.json")
        metrics = result.get("metrics", {})
        if not isinstance(metrics, dict) or metrics.get("count") != 823 or result.get("counts", {}).get("clips") != 823:
            raise RuntimeError(f"{variant} result does not cover all val823 clips")
        for key in ("macro_f1", "accuracy"):
            value = metrics.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise RuntimeError(f"{variant} result has invalid {key}")
        if not (directory / "best.pt").is_file():
            raise RuntimeError(f"{variant} completed without best.pt")
        history = json.loads((directory / "history.json").read_text())
        if not isinstance(history, list) or not history:
            raise RuntimeError(f"{variant} completed without a nonempty history")


def _recheck_idle(plan, indices):
    current = assert_idle_gpus(indices)
    initial = plan["GPU_environment"]["physical_gpu_uuid"]
    for index in indices:
        expected = initial.get(index, initial.get(str(index)))
        if current["physical_gpu_uuid"].get(index) != expected:
            raise RuntimeError("physical GPU inventory changed after launch planning")


def execute_plan(plan, run=None):
    """Test injection is not exposed in the CLI and never claims real CUDA."""
    output = fresh_output(plan["arguments"]["output"])
    logs = output / "launcher_logs"
    logs.mkdir()
    write_json(output / "run_protocol.json", plan, exclusive=True)
    write_json(output / "frozen_config.json", plan["config"], exclusive=True)
    (output / "run_commit.txt").write_text(plan["repository"]["commit"] + "\n")
    (output / ".launch_once").write_text(plan["repository"]["commit"] + "\n")
    (logs / "pids.tsv").write_text("stage\tphysical_gpu\tpid\n")
    (logs / "exit_codes.tsv").write_text("stage\tphysical_gpu\texit_code\n")
    lock, outcomes = threading.Lock(), {}
    first_gpu = plan["arguments"]["gpus"][0]

    def launch(stage, physical_gpu=None):
        if run is None:
            validate_runtime(plan)
        print(json.dumps({"stage": stage, "physical_gpu": physical_gpu, "status": "starting"}), flush=True)
        if run is not None:
            code = run(stage, plan["commands"][stage])
        else:
            with (logs / f"{stage}.log").open("x") as log:
                process = subprocess.Popen(plan["commands"][stage], cwd=ROOT,
                    env=worker_environment(plan, physical_gpu), stdout=log, stderr=subprocess.STDOUT)
                with lock, (logs / "pids.tsv").open("a") as stream:
                    stream.write(f"{stage}\t{physical_gpu}\t{process.pid}\n")
                code = process.wait()
        with lock:
            outcomes[stage] = code
            with (logs / "exit_codes.tsv").open("a") as stream:
                stream.write(f"{stage}\t{physical_gpu}\t{code}\n")
        if code != 0:
            raise RuntimeError(f"{stage} failed with exit {code}; later waves will not start")

    try:
        launch("unit_tests")
        launch("cuda_smoke", first_gpu)
        if run is None:
            _verify_smoke(plan, output / "cuda_smoke.json")
        for seed in plan["config"]["seeds"]:
            for wave in plan["waves"]:
                if run is None:
                    _recheck_idle(plan, [job["physical_gpu"] for job in wave])
                errors = []
                with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                    futures = [pool.submit(launch, f"{job['variant']}_{seed}", job["physical_gpu"]) for job in wave]
                    for future in futures:
                        try:
                            future.result()
                        except Exception as error:
                            errors.append(error)
                # Exiting the pool waits for existing jobs without killing them.
                if errors:
                    raise errors[0]
            if run is None:
                _verify_group_results(output, seed)
                _recheck_idle(plan, [first_gpu])
            launch(f"evaluate_{seed}", first_gpu)
        write_json(output / "suite_result.json", {"status": "complete", "training_completed": run is None,
                   "test_injection_only": run is not None, "exit_codes": outcomes,
                   "seeds": plan["config"]["seeds"]}, exclusive=True)
    except Exception as error:
        write_json(output / "suite_result.json", {"status": "failed", "training_completed": False,
                   "exit_codes": outcomes, "error_type": type(error).__name__, "error": str(error)}, exclusive=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-index", "source-protocol-dir", "config", "base-checkpoint", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", type=gpu_indices, default=[0, 1, 2, 3])
    parser.add_argument("--expected-commit")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = prepare_plan(args)
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
        return
    plan["GPU_environment"] = assert_idle_gpus(args.gpus)
    execute_plan(plan)


if __name__ == "__main__":
    main()
