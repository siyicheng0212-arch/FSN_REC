"""Plan or run one source-policy suite; default is read-only planning.

One frozen Original export, two independent R jobs, five same-logit controls.
No Original retraining, old launcher, test access, auto-resume or extra seeds.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from .data import sha256_file
from .source_edges import build_source_relations, load_fixed_manifests
from .protocol import relation_source_hashes

ROOT = Path(__file__).resolve().parents[2]


def gpu_indices(value):
    try:
        result = [int(x) for x in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--gpus must be two nonnegative indices, e.g. 0,1") from error
    if len(result) != 2 or min(result) < 0 or len(set(result)) != 2:
        raise argparse.ArgumentTypeError("two distinct GPU indices are required; this is not DDP")
    return result


def source_hashes():
    return relation_source_hashes()


def repository_state():
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    return {"commit": sha, "clean": not dirty}


def build_commands(args):
    out = Path(args.output).resolve()
    command = [str(Path(args.python).resolve()), "-m"]
    device = f"cuda:{args.gpus[0]}"
    protocol = out / "source_protocol"
    index = out / "features" / "index.jsonl"
    commands = {
        "source_protocol": command + ["experiments.relation.source_edges", "--manifest-dir", args.manifest_dir,
            "--source-map", args.source_map, "--output", str(protocol), "--max-gap-seconds", str(args.max_gap_seconds)],
        "cuda_smoke": command + ["experiments.relation.smoke", "--manifest-dir", args.manifest_dir,
            "--cache-root", args.cache_root, "--original-checkpoint", args.original_checkpoint,
            "--source-protocol-dir", str(protocol), "--output", str(out / "cuda_smoke.json"), "--device", device],
        "export": command + ["experiments.relation.export", "--train-manifest", str(Path(args.manifest_dir) / "train.jsonl"),
            "--val-manifest", str(Path(args.manifest_dir) / "val.jsonl"), "--cache-root", args.cache_root,
            "--checkpoint", args.original_checkpoint, "--output", str(out / "features"),
            "--source-protocol-dir", str(protocol), "--device", device, "--batch-size", "4", "--workers", "4"],
        "fit_transition": command + ["experiments.relation.fit_transition", "--index", str(index),
            "--edges", str(protocol / "edges.jsonl"), "--source-protocol-dir", str(protocol),
            "--output", str(out / "transition"), "--smoothing", "1"],
    }
    for mode, gpu in zip(("mlp", "dual"), args.gpus):
        commands[mode] = command + ["experiments.relation.train", "--index", str(index),
            "--edges", str(protocol / "edges.jsonl"), "--source-protocol-dir", str(protocol),
            "--output", str(out / mode), "--mode", mode, "--device", f"cuda:{gpu}",
            "--seed", str(args.seed), "--epochs", str(args.epochs), "--patience", "5",
            "--batch-size", "16", "--lr", "0.001", "--dim", "64", "--heads", "4",
            "--endpoint-tokens", "2", "--dropout", "0.1", "--balance-loss", "train_ratio",
            "--threshold", str(args.threshold)]
    commands["evaluate"] = command + ["experiments.relation.evaluate_suite", "--index", str(index),
        "--protocol-dir", str(protocol), "--transition", str(out / "transition"),
        "--mlp-checkpoint", str(out / "mlp" / "best.pt"), "--dual-checkpoint", str(out / "dual" / "best.pt"),
        "--output", str(out / "evaluation"), "--device", device,
        "--threshold", str(args.threshold), "--strength", "1"]
    return commands


def prepare_plan(args):
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("suite output already exists; preserve it and choose a new directory")
    if out.is_relative_to(ROOT):
        raise ValueError("private outputs must be outside the source worktree")
    for value in (args.python, args.original_checkpoint, args.source_map):
        if not Path(value).is_file():
            raise FileNotFoundError(value)
    if not Path(args.cache_root).is_dir():
        raise FileNotFoundError(args.cache_root)
    if args.epochs < 1 or args.seed != 42 or not math.isfinite(args.threshold) or not 0 < args.threshold < 1:
        raise ValueError("this first suite fixes seed42, positive epochs and a threshold in (0,1)")
    rows, manifests = load_fixed_manifests(args.manifest_dir)
    source_map = json.loads(Path(args.source_map).read_text())
    relations = build_source_relations(rows, source_map, max_gap_seconds=args.max_gap_seconds)
    for split in ("train", "val"):
        counts = relations["audit"]["edge_counts"][split]
        if not counts["C"] or not counts["D"]:
            raise ValueError(f"{split} has no usable C or D candidates; report it without inventing pairs or resplitting")
    state = repository_state()
    if args.expected_commit and state["commit"] != args.expected_commit:
        raise ValueError("HEAD differs from the expected frozen commit")
    if args.execute and not state["clean"]:
        raise ValueError("execution requires a clean committed isolated worktree")
    return {"schema": "fsn-source-relation-suite-v1", "training_completed": False,
            "repository": state, "source_sha256": source_hashes(),
            "original_checkpoint_sha256": sha256_file(args.original_checkpoint),
            "manifest_sha256": manifests, "source_map_sha256": sha256_file(args.source_map),
            "arguments": vars(args), "label_policy": "source_rule_v1",
            "clip_counts": relations["audit"]["clip_counts"], "edge_counts": relations["audit"]["edge_counts"],
            "excluded_adjacencies": relations["audit"]["excluded_adjacencies"],
            "evaluation_role": "val823_development_not_independent_test", "seed": 42,
            "commands": build_commands(args), "GPU_R_mapping": dict(zip(("mlp", "dual"), args.gpus)),
            "settings": {"R_epochs_max": args.epochs, "R_patience": 5, "R_batch_size": 16,
                         "R_lr": .001, "R_weight_decay": .01, "R_balance": "train_ratio_only",
                         "threshold": args.threshold, "threshold_calibrated": False, "flow_strength": 1,
                         "max_gap_seconds": args.max_gap_seconds, "no_extra_seeds": True}}


def assert_idle_gpus(indices):
    inventory = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, check=True)
    physical = {}
    for line in inventory.stdout.splitlines():
        index, uuid = (part.strip() for part in line.split(","))
        physical[int(index)] = uuid
    if sorted(physical) != list(range(len(physical))) or any(gpu not in physical for gpu in indices):
        raise RuntimeError("requested GPU is absent or physical GPU inventory is not contiguous")
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    identity = ",".join(str(i) for i in sorted(physical))
    if inherited is not None and inherited != identity:
        raise RuntimeError("inherited CUDA_VISIBLE_DEVICES is not the full identity mapping; inspect it and unset it before launch")
    for gpu in indices:
        query = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        if query.stdout.strip():
            raise RuntimeError(f"GPU{gpu} already has compute processes; do not interrupt or duplicate them")
    # UUID order explicitly makes cuda:N correspond to nvidia-smi physical index N.
    return {"CUDA_VISIBLE_DEVICES": ",".join(physical[i] for i in sorted(physical)),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "physical_gpu_uuid": physical}


def validate_runtime_inputs(plan):
    args = plan["arguments"]
    expected = {args["original_checkpoint"]: plan["original_checkpoint_sha256"],
                args["source_map"]: plan["source_map_sha256"]}
    expected.update({str(Path(args["manifest_dir"]) / (split + ".jsonl")): digest
                     for split, digest in plan["manifest_sha256"].items()})
    if any(sha256_file(path) != digest for path, digest in expected.items()):
        raise RuntimeError("external checkpoint/source map/manifests changed after planning; preserve artifacts")
    state = repository_state()
    if not state["clean"] or state["commit"] != plan["repository"]["commit"] or source_hashes() != plan["source_sha256"]:
        raise RuntimeError("runtime source or git state changed; stop without restarting")


def validate_stage_seal(stage, plan, output):
    audit = json.loads((output / "source_protocol" / "audit.json").read_text())
    if (audit.get("source_map_sha256") != plan["source_map_sha256"]
            or audit.get("manifest_sha256") != plan["manifest_sha256"]
            or audit.get("clip_counts") != plan["clip_counts"]
            or audit.get("edge_counts") != plan["edge_counts"]
            or audit.get("max_gap_seconds") != plan["settings"]["max_gap_seconds"]):
        raise RuntimeError("source protocol differs from the planned source map/manifests/settings")
    if stage == "export":
        seal = json.loads((output / "features" / "protocol.json").read_text())
        manifests = {split: seal.get(split + "_manifest_sha256") for split in ("train", "val")}
        if (seal.get("status") != "complete" or seal.get("A_frozen") is not True
                or seal.get("checkpoint_sha256") != plan["original_checkpoint_sha256"]
                or seal.get("source_audit_sha256") != sha256_file(output / "source_protocol" / "audit.json")
                or manifests != plan["manifest_sha256"]
                or seal.get("index_sha256") != sha256_file(output / "features" / "index.jsonl")):
            raise RuntimeError("export differs from the checkpoint/source protocol verified by smoke")


def execute_plan(plan, output, run=None):
    """Serialized dependencies; only the two independent R jobs run in parallel."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "launcher_logs"
    logs.mkdir()
    (output / "run_protocol.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    (output / ".launch_once").write_text(plan["repository"]["commit"] + "\n")
    (output / "run_commit.txt").write_text(plan["repository"]["commit"] + "\n")
    (logs / "pids.tsv").write_text("stage\tpid\n")
    (logs / "exit_codes.tsv").write_text("stage\texit_code\n")
    lock = threading.Lock()
    outcomes = {}

    def launch(stage):
        if run is None:
            validate_runtime_inputs(plan)
        print(json.dumps({"stage": stage, "status": "starting"}), flush=True)
        if run is not None:  # internal unit-test injection; no CLI bypass
            code = run(stage, plan["commands"][stage])
        else:
            env = {**os.environ, "OMP_NUM_THREADS": "6", "MKL_NUM_THREADS": "6",
                   **{key: plan["GPU_environment"][key] for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}}
            with (logs / f"{stage}.log").open("x") as log:
                process = subprocess.Popen(plan["commands"][stage], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                with lock, (logs / "pids.tsv").open("a") as pids:
                    pids.write(f"{stage}\t{process.pid}\n")
                code = process.wait()
        with lock:
            outcomes[stage] = int(code)
            with (logs / "exit_codes.tsv").open("a") as exits:
                exits.write(f"{stage}\t{code}\n")
        if code:
            raise RuntimeError(f"{stage} exited {code}; preserve artifacts; no auto-restart")
        return code

    status = {"training_completed": False, "outcomes": outcomes, "started_unix": time.time()}
    try:
        for stage in ("source_protocol", "cuda_smoke", "export", "fit_transition"):
            launch(stage)
            if stage in ("source_protocol", "export"):
                validate_stage_seal(stage, plan, output)
            if stage == "cuda_smoke":
                smoke = json.loads((output / "cuda_smoke.json").read_text())
                if (smoke.get("all_passed") is not True or smoke.get("checkpoint_sha256") != plan["original_checkpoint_sha256"]
                        or smoke.get("source_sha256") != plan["source_sha256"]
                        or smoke.get("real_cache_cuda_smoke_completed") is not True
                        or smoke.get("manifest_sha256") != plan["manifest_sha256"]
                        or smoke.get("source_audit_sha256") != sha256_file(output / "source_protocol" / "audit.json")
                        or smoke.get("sampling") != "36/8/12/128"
                        or smoke.get("device") != f"cuda:{plan['GPU_R_mapping']['mlp']}"):
                    raise RuntimeError("smoke report does not match this source/checkpoint protocol")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(launch, mode) for mode in ("mlp", "dual")]
            errors = []
            for future in futures:
                try:
                    future.result()
                except Exception as error:
                    errors.append(error)
            if errors:
                raise errors[0]
        for mode in ("mlp", "dual"):
            result = json.loads((output / mode / "metrics.json").read_text())
            if (not (output / mode / "best.pt").is_file() or result.get("epochs_completed", 0) < 1
                    or not math.isfinite(result["val"]["bce"])):
                raise RuntimeError(f"{mode} result is incomplete")
        launch("evaluate")
        summary = json.loads((output / "evaluation" / "public_safe_summary.json").read_text())
        if set(summary["variants"]) != {"A_only", "all_candidate", "source_rule", "learned_mlp", "learned_dual"}:
            raise RuntimeError("evaluation lacks the five paired controls")
        status.update(training_completed=True, finished_unix=time.time(), public_summary="evaluation/public_safe_summary.json")
    except BaseException as error:
        status.update(error_type=type(error).__name__, error=str(error), finished_unix=time.time())
        raise
    finally:
        (output / "suite_result.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    return status


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest-dir", "cache-root", "original-checkpoint", "output"):
        result.add_argument(f"--{name}", required=True)
    result.add_argument("--source-map", default=str(Path(__file__).with_name("source_map.json")))
    result.add_argument("--python", default=sys.executable)
    result.add_argument("--gpus", type=gpu_indices, default=[0, 1])
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--epochs", type=int, default=30)
    result.add_argument("--threshold", type=float, default=.9)
    result.add_argument("--max-gap-seconds", type=float, default=5.)
    result.add_argument("--expected-commit")
    result.add_argument("--execute", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for key in ("manifest_dir", "cache_root", "original_checkpoint", "output", "source_map", "python"):
        setattr(args, key, str(Path(getattr(args, key)).resolve()))
    plan = prepare_plan(args)
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return plan
    plan["GPU_environment"] = assert_idle_gpus(args.gpus)
    return execute_plan(plan, args.output)


if __name__ == "__main__":
    main()
