"""Run four matched gates over a strictly verified, frozen historical A+TCN.

Planning is read-only; execution requires an exact clean commit, an unused
private output directory, idle physical GPUs, unit tests, and real CUDA smoke.
Two GPUs run two waves, four GPUs run one wave.  Each independent child sees
only its assigned physical GPU as cuda:0.  A failed wave blocks later waves;
there are no automatic retries or silently adjusted experimental settings.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading

from experiments.context_utility.config import baseline_source_hashes, load_config as load_legacy_config
from experiments.fsn_tcn.io import code_hashes as tcn_code_hashes, fresh_output, write_json
from experiments.relation.data import sha256_file
from experiments.relation.run_suite import assert_idle_gpus
from .revision_config import VARIANTS, load_config


ROOT = Path(__file__).resolve().parents[2]


def source_hashes():
    hashes = tcn_code_hashes()
    for relative in ("experiments/context_utility", "experiments/legacy_tcn_audit",
                     "experiments/selective_context"):
        for path in sorted((ROOT / relative).glob("*.py")):
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    return hashes


def repository_state():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    return {"commit": commit, "clean": not dirty}


def gpu_indices(raw):
    try:
        indices = [int(item) for item in raw.split(",")]
    except (AttributeError, ValueError) as error:
        raise argparse.ArgumentTypeError("use two or four distinct physical GPU indices") from error
    if len(indices) not in (2, 4) or len(set(indices)) != len(indices) or min(indices) < 0:
        raise argparse.ArgumentTypeError("use two or four distinct physical GPU indices")
    return indices


def wave_schedule(gpus):
    if (len(gpus) not in (2, 4) or len(set(gpus)) != len(gpus)
            or any(type(gpu) is not int or gpu < 0 for gpu in gpus)):
        raise ValueError("two or four distinct physical GPUs are required")
    return [[{"variant": variant, "physical_gpu": gpu}
             for variant, gpu in zip(VARIANTS[start:start + len(gpus)], gpus)]
            for start in range(0, len(VARIANTS), len(gpus))]


def _resolve_files(args):
    for name in ("python", "feature_index", "legacy_config", "legacy_checkpoint",
                 "historical_predictions", "config"):
        path = Path(getattr(args, name)).resolve()
        setattr(args, name, str(path))
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    if args.historical_column_map is not None:
        path = Path(args.historical_column_map).resolve()
        args.historical_column_map = str(path)
        if not path.is_file():
            raise FileNotFoundError(f"historical_column_map: {path}")
    args.source_protocol_dir = str(Path(args.source_protocol_dir).resolve())
    if not Path(args.source_protocol_dir).is_dir():
        raise FileNotFoundError(args.source_protocol_dir)
    args.output = str(Path(args.output).resolve())
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("suite output already exists; preserve it and choose a fresh directory")
    if destination.is_relative_to(ROOT):
        raise ValueError("private training output must be outside the source worktree")
    if args.chain_layout not in ("full_chain", "eligible_segments"):
        raise ValueError("historical chain layout must be declared and verified")


def build_commands(args, config):
    prefix = [str(Path(args.python).resolve()), "-m"]
    output = Path(args.output).resolve()
    common = ["--feature-index", args.feature_index,
              "--source-protocol-dir", args.source_protocol_dir,
              "--legacy-config", args.legacy_config,
              "--legacy-checkpoint", args.legacy_checkpoint,
              "--historical-predictions", args.historical_predictions,
              "--chain-layout", args.chain_layout, "--config", args.config]
    if args.historical_column_map is not None:
        common += ["--historical-column-map", args.historical_column_map]
    if args.historical_logits_key is not None:
        common += ["--historical-logits-key", args.historical_logits_key]
    tests = sorted({str(path.relative_to(ROOT)) for pattern in (
        "test_selective_context_revision*.py", "test_legacy_tcn*.py",
        "test_context_utility_config.py", "test_tcn_revision_training.py")
        for path in (ROOT / "tests").glob(pattern)})
    if not tests:
        raise ValueError("selective revision tests are missing")
    commands = {
        "unit_tests": prefix + ["pytest", "-q", "-p", "no:cacheprovider", *tests],
        "cuda_smoke": prefix + ["experiments.selective_context.revision_smoke", *common,
                                  "--device", "cuda:0", "--output", str(output / "cuda_smoke.json")],
    }
    for seed in config["seeds"]:
        for variant in VARIANTS:
            commands[f"{variant}_{seed}"] = prefix + ["experiments.selective_context.revision_train", *common,
                "--variant", variant, "--seed", str(seed),
                "--smoke-report", str(output / "cuda_smoke.json"),
                "--output", str(output / f"seed_{seed}" / variant), "--device", "cuda:0"]
    return commands


def prepare_plan(args):
    """Verify actual legacy assets without writing output or inspecting GPUs."""
    _resolve_files(args)
    config = load_config(args.config)
    if config["seeds"] != [42]:
        raise ValueError("this first experiment fixes seed42; never add seeds silently")
    legacy = load_legacy_config(args.legacy_config)
    if legacy["baseline"]["kind"] != "legacy":
        raise ValueError("the old A+TCN checkpoint requires its audited historical model")
    for name in (legacy["baseline"]["legacy_audit"],
                 *legacy["baseline"]["source_dependencies"]):
        if not Path(name).is_absolute():
            raise ValueError("historical source dependencies/audit must have fixed absolute paths")
    from .revision_train import prepare_revision_inputs, revision_smoke_seal
    prepared = prepare_revision_inputs(args, device="cpu")
    seals = revision_smoke_seal(args, prepared)
    if not isinstance(seals, dict):
        raise ValueError("revision trainer did not produce an input seal")
    state = repository_state()
    if args.expected_commit is not None and state["commit"] != args.expected_commit:
        raise ValueError("HEAD differs from specified fixed commit")
    if args.execute and (not args.expected_commit or not state["clean"]):
        raise ValueError("execution requires --expected-commit and a clean committed worktree")
    bundle = prepared["bundle"]
    baseline_sources = baseline_source_hashes(legacy)
    old_audit = prepared["base_audit"].get("legacy_audit")
    if not isinstance(old_audit, dict) or not old_audit.get("sha256"):
        raise ValueError("historical TCN does not have a verified checkpoint audit")
    source_sha = source_hashes()
    for name, expected in {
        "fingerprint": bundle.fingerprint,
        "source_sha256": source_sha,
        "external_baseline_sha256": baseline_sources,
        "legacy_audit_sha256": old_audit["sha256"],
        "config_sha256": sha256_file(args.config),
        "legacy_config_sha256": sha256_file(args.legacy_config),
        "legacy_checkpoint_sha256": sha256_file(args.legacy_checkpoint),
        "historical_predictions_sha256": sha256_file(args.historical_predictions),
        "historical_column_map_sha256": (sha256_file(args.historical_column_map)
                                           if args.historical_column_map else None),
        "chain_layout": args.chain_layout,
    }.items():
        if seals.get(name) != expected:
            raise ValueError(f"revision smoke seal disagrees on {name}")
    export_protocol = Path(args.feature_index).parent / "protocol.json"
    if not export_protocol.is_file():
        raise ValueError("canonical frozen A export protocol is missing")
    return {"schema": "fsn-tcn-revision-suite-v1", "training_completed": False,
        "repository": state, "arguments": vars(args).copy(), "config": config,
        "smoke_seal": seals, "fingerprint": bundle.fingerprint,
        "source_sha256": source_sha, "external_baseline_sha256": baseline_sources,
        "legacy_audit_sha256": old_audit["sha256"],
        "feature_export_protocol_sha256": sha256_file(export_protocol),
        "base_load_audit": prepared["base_audit"],
        "historical_parity": prepared["parity"],
        "commands": build_commands(args, config), "waves": wave_schedule(args.gpus),
        "GPU_mapping": {job["variant"]: job["physical_gpu"]
                        for wave in wave_schedule(args.gpus) for job in wave},
        "DDP": False, "A_frozen": True, "old_TCN_frozen": True,
        "evaluation_role": "val823_development_not_independent_test"}


def validate_runtime(plan):
    args, sealed = plan["arguments"], plan["smoke_seal"]
    if repository_state() != plan["repository"] or source_hashes() != plan["source_sha256"]:
        raise RuntimeError("code/commit changed after planning; no source changes during training")
    paths = {
        "config": (args["config"], sealed["config_sha256"]),
        "legacy config": (args["legacy_config"], sealed["legacy_config_sha256"]),
        "legacy checkpoint": (args["legacy_checkpoint"], sealed["legacy_checkpoint_sha256"]),
        "historical predictions": (args["historical_predictions"], sealed["historical_predictions_sha256"]),
        "feature index": (args["feature_index"], plan["fingerprint"]["feature_index_sha256"]),
        "feature export protocol": (str(Path(args["feature_index"]).parent / "protocol.json"),
                                    plan["feature_export_protocol_sha256"]),
        "legacy audit": (plan["base_load_audit"]["legacy_audit"]["path"],
                         plan["legacy_audit_sha256"]),
        "source protocol audit": (str(Path(args["source_protocol_dir"]) / "audit.json"),
                                  plan["fingerprint"]["source_audit_sha256"]),
    }
    if args["historical_column_map"] is not None:
        paths["historical column map"] = (args["historical_column_map"],
                                          sealed["historical_column_map_sha256"])
    paths.update({f"source protocol {name}": (str(Path(args["source_protocol_dir"]) / name), digest)
                  for name, digest in plan["fingerprint"]["source_files_sha256"].items()})
    for name, (path, digest) in paths.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"sealed {name} changed during this run")
    if baseline_source_hashes(load_legacy_config(args["legacy_config"])) != plan["external_baseline_sha256"]:
        raise RuntimeError("historical source dependency changed during this run")


def worker_environment(plan, physical_gpu):
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    if physical_gpu is None:
        environment.update(CUDA_VISIBLE_DEVICES="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    else:
        uuids = plan["GPU_environment"]["physical_gpu_uuid"]
        uuid = uuids.get(physical_gpu, uuids.get(str(physical_gpu)))
        if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
            raise RuntimeError("physical GPU UUID mapping is missing")
        environment.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    return environment


def _verify_smoke(plan, path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"all_passed": True, "real_cuda_tcn_revision_smoke_completed": True,
                **plan["smoke_seal"]}
    if any(key not in report or report[key] != value for key, value in expected.items()):
        raise RuntimeError("real CUDA revision smoke does not certify this sealed suite")


def _verify_group_results(output, seed):
    summary = {}
    comparator = None
    for variant in VARIANTS:
        directory = Path(output) / f"seed_{seed}" / variant
        report = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        if report.get("status") in {"error", "failed"} or report.get("counts", {}).get("clips") != 823:
            raise RuntimeError(f"{variant} did not complete validation of all 823 clips")
        comparison = report.get("metrics")
        if not isinstance(comparison, dict):
            raise RuntimeError(f"{variant} lacks old-TCN/A paired comparison")
        for name in ("A", "old_TCN"):
            baseline = comparison.get(name)
            if not isinstance(baseline, dict) or baseline.get("count") != 823:
                raise RuntimeError(f"{variant} lacks complete {name} paired comparison")
            for key in ("macro_f1", "accuracy"):
                value = baseline.get(key)
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise RuntimeError(f"{variant} has invalid {name} {key}")
        current_comparator = {name: {key: comparison[name][key]
                                     for key in ("count", "macro_f1", "accuracy", "confusion")}
                              for name in ("A", "old_TCN")}
        if comparator is None:
            comparator = current_comparator
        elif comparator != current_comparator:
            raise RuntimeError("matched arms disagree on frozen A/old-TCN val823 baselines")
        metrics = comparison.get("new")
        if not isinstance(metrics, dict) or metrics.get("count") != 823:
            raise RuntimeError(f"{variant} missing 823-clip revised metrics")
        for key in ("macro_f1", "accuracy"):
            value = metrics.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise RuntimeError(f"{variant} has invalid {key}")
        if not (directory / "best.pt").is_file():
            raise RuntimeError(f"{variant} completed without best.pt")
        history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        if not isinstance(history, list) or not history:
            raise RuntimeError(f"{variant} completed without nonempty history")
        summary[variant] = {
            "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
            "A_macro_f1": comparison["A"]["macro_f1"],
            "old_TCN_macro_f1": comparison["old_TCN"]["macro_f1"],
            "delta_macro_f1_vs_old_TCN": metrics["macro_f1"] - comparison["old_TCN"]["macro_f1"],
            "delta_macro_f1_vs_A": metrics["macro_f1"] - comparison["A"]["macro_f1"],
            "val_clips": 823, "best_epoch": report.get("best_epoch"),
            "gate_trainable_parameters": report.get("gate_trainable_parameters"),
        }
    return summary


def _recheck_idle(plan, indices):
    current = assert_idle_gpus(indices)
    baseline = plan["GPU_environment"]["physical_gpu_uuid"]
    for index in indices:
        if current["physical_gpu_uuid"].get(index) != baseline.get(index, baseline.get(str(index))):
            raise RuntimeError("physical GPU UUID mapping changed since launch planning")


def execute_plan(plan, run=None):
    """`run` is an internal test hook and cannot attest real CUDA/training."""
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

    def launch(stage, gpu=None):
        if run is None:
            validate_runtime(plan)
        print(json.dumps({"stage": stage, "physical_gpu": gpu, "status": "starting"}), flush=True)
        if run is not None:
            code = int(run(stage, plan["commands"][stage]))
        else:
            with (logs / f"{stage}.log").open("x") as stream:
                worker = subprocess.Popen(plan["commands"][stage], cwd=ROOT,
                    env=worker_environment(plan, gpu), stdout=stream, stderr=subprocess.STDOUT)
                with lock, (logs / "pids.tsv").open("a") as pids:
                    pids.write(f"{stage}\t{gpu}\t{worker.pid}\n")
                code = worker.wait()
        with lock, (logs / "exit_codes.tsv").open("a") as exits:
            outcomes[stage] = code
            exits.write(f"{stage}\t{gpu}\t{code}\n")
        if code:
            raise RuntimeError(f"{stage} failed with exit {code}; preserve artifacts and do not retry")

    try:
        launch("unit_tests")
        launch("cuda_smoke", first_gpu)
        if run is None:
            _verify_smoke(plan, output / "cuda_smoke.json")
        for seed in plan["config"]["seeds"]:
            for wave in plan["waves"]:
                if run is None:
                    _recheck_idle(plan, [item["physical_gpu"] for item in wave])
                errors = []
                with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                    futures = [pool.submit(launch, f"{item['variant']}_{seed}", item["physical_gpu"])
                               for item in wave]
                    for future in futures:
                        try:
                            future.result()
                        except Exception as error:
                            errors.append(error)
                if errors:
                    raise errors[0]
            if run is None:
                metrics = _verify_group_results(output, seed)
                write_json(output / f"seed_{seed}" / "public_safe_summary.json",
                    {"schema": "fsn-tcn-revision-summary-v1", "role": plan["evaluation_role"],
                     "variants": metrics}, exclusive=True)
        write_json(output / "suite_result.json", {"status": "complete", "training_completed": run is None,
            "test_injection_only": run is not None, "seeds": plan["config"]["seeds"],
            "exit_codes": outcomes}, exclusive=True)
    except Exception as error:
        write_json(output / "suite_result.json", {"status": "failed", "training_completed": False,
            "exit_codes": outcomes, "error_type": type(error).__name__, "error": str(error)}, exclusive=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-index", "source-protocol-dir", "legacy-config", "legacy-checkpoint",
                 "historical-predictions", "config", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--historical-column-map")
    parser.add_argument("--historical-logits-key")
    parser.add_argument("--chain-layout", choices=("full_chain", "eligible_segments"), required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", type=gpu_indices, default=[0, 1])
    parser.add_argument("--expected-commit")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = prepare_plan(args)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        return
    plan["GPU_environment"] = assert_idle_gpus(args.gpus)
    execute_plan(plan)


if __name__ == "__main__":
    main()
