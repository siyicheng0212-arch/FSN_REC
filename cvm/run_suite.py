"""Generate a frozen CVM run plan; launching always requires --execute.

Four GPUs run independent jobs. There is no DDP, automatic promotion, resume,
extra seed wave, or test evaluation. A plan and launch lock are never replaced.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from cvm.train import ROOT, atomic_json, sha256_file, source_digest, source_hashes

FORMAL_SEEDS = (42, 2026, 2027)
CORE_BACKBONES = ("r2plus1d_18", "mvit_v2_s", "videomamba_tiny16")


def declared_configurations(stage: str, backbones: tuple[str, ...] | None = None,
                            visual_taxonomy: str | None = None) -> list[dict[str, str]]:
    selected = backbones or (("r2plus1d_18",) if stage in ("pilot", "smoke") else CORE_BACKBONES)
    if any(backbone not in CORE_BACKBONES for backbone in selected) or len(set(selected)) != len(selected):
        raise ValueError("backbones must be unique declared core architectures")
    configurations = []
    for backbone in selected:
        modes = ("flat", "capacity_control", "aux_flat", "hierarchy") if stage in ("pilot", "smoke") else ("flat", "hierarchy")
        configurations.extend({"backbone": backbone, "mode": mode, "taxonomy": "clinical"} for mode in modes)
    if stage == "formal" and "r2plus1d_18" in selected:
        configurations.extend({"backbone": "r2plus1d_18", "mode": mode, "taxonomy": "clinical"}
                              for mode in ("capacity_control", "aux_flat"))
        configurations.extend({"backbone": "r2plus1d_18", "mode": "hierarchy", "taxonomy": taxonomy}
                              for taxonomy in ("random_17", "random_29", "random_43"))
    if visual_taxonomy:
        configurations.append({"backbone": "r2plus1d_18", "mode": "hierarchy", "taxonomy": "visual_train_only", "taxonomy_file": visual_taxonomy})
    return configurations


def generate_plan(args: argparse.Namespace) -> Path:
    from cvm.protocol import load_protocol
    protocol = load_protocol(args.protocol)
    weights = json.loads(args.weights_json.read_text(encoding="utf-8"))
    if not isinstance(weights, dict):
        raise ValueError("weights JSON must map declared backbone names to official local checkpoint paths")
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if not gpus or len(set(gpus)) != len(gpus) or any(not value.isdigit() for value in gpus):
        raise ValueError("GPU IDs must be unique nonnegative integers")
    selected = tuple(args.backbones.split(",")) if args.backbones else None
    configurations = declared_configurations(args.stage, selected, str(args.taxonomy_file.resolve()) if args.taxonomy_file else None)
    seeds = FORMAL_SEEDS if args.stage == "formal" else (42,)
    settings = {"batch_size": args.batch_size, "accum_steps": args.accum_steps,
                "epochs": 1 if args.stage == "smoke" else args.epochs,
                "warmup_epochs": 0 if args.stage == "smoke" else args.warmup_epochs,
                "patience": args.patience, "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
                "warmup_lr": args.warmup_lr, "weight_decay": args.weight_decay, "class_weights": args.class_weights,
                "workers": args.workers, "optimizer": args.optimizer, "amp": args.amp,
                "clip_grad": args.clip_grad, "augmentation": args.augmentation,
                "aux_weight": 1., "group_weight": 1., "conditional_weight": 1.}
    if settings["batch_size"] <= 0 or settings["accum_steps"] <= 0:
        raise ValueError("batch and accumulation must be positive")
    jobs = []
    for configuration in configurations:
        backbone = configuration["backbone"]
        entry = weights.get(backbone, weights.get("videomamba_tiny") if backbone == "videomamba_tiny16" else None)
        if isinstance(entry, str):
            entry = {"checkpoint": entry}
        if not isinstance(entry, dict) or not isinstance(entry.get("checkpoint"), str):
            raise ValueError(f"weights JSON lacks an explicit local checkpoint for {backbone}")
        checkpoint = Path(entry["checkpoint"]).expanduser().resolve()
        checkpoint_sha = sha256_file(checkpoint) if checkpoint.is_file() else None
        for seed in seeds:
            name = f"{backbone}__{configuration['mode']}__{configuration['taxonomy']}__seed{seed}"
            jobs.append({**configuration, "seed": seed, "name": name,
                         "output": str((args.output / "runs" / name).resolve()),
                         "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
                         "videomamba_root": str(Path(entry.get("external_repo", args.videomamba_root)).expanduser().resolve()) if entry.get("external_repo", args.videomamba_root) else None,
                         "main_decoder": "soft" if configuration["mode"] == "hierarchy" else "flat"})
    plan = {"schema_version": "fsn-cvm-plan-1", "stage": args.stage,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "protocol": str(args.protocol.resolve()), "protocol_sha256": protocol.protocol_sha256,
            "cache_root": str(args.cache_root.resolve()), "source_hashes": source_hashes(), "code_hash": source_digest(),
            "gpus": list(gpus), "settings": settings, "jobs": jobs,
            "fixed_formal_seeds": list(FORMAL_SEEDS), "test_evaluated": False,
            "full_declared_formal_matrix": args.stage == "formal" and (selected is None or set(selected) == set(CORE_BACKBONES)),
            "formal_claim_ready": False, "launch_requires_explicit_execute": True,
            "local_readiness": {"all_checkpoint_files_exist": all(job["checkpoint_sha256"] for job in jobs),
                                "CUDA_and_dependencies": "not checked by plan generation"}}
    try:
        plan["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        plan["git_commit"] = None
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "run_plan.json"
    atomic_json(path, plan)
    print(json.dumps({"plan": str(path), "jobs": len(jobs), "stage": args.stage, "training_started": False,
                      "formal_claim_ready": False}, ensure_ascii=False), flush=True)
    return path


def trainer_command(plan: dict[str, Any], job: dict[str, Any], python: str) -> list[str]:
    command = [python, "-m", "cvm.train", "train", "--protocol", plan["protocol"],
               "--cache-root", plan["cache_root"], "--output", job["output"],
               "--backbone", job["backbone"], "--mode", job["mode"],
               "--seed", str(job["seed"]), "--checkpoint", job["checkpoint"],
               "--main-decoder", job["main_decoder"], "--expected-protocol-sha256", plan["protocol_sha256"],
               "--expected-code-hash", plan["code_hash"]]
    if job.get("taxonomy_file"):
        command.extend(("--taxonomy-file", job["taxonomy_file"]))
    else:
        command.extend(("--taxonomy", job["taxonomy"]))
    if job.get("videomamba_root"):
        command.extend(("--videomamba-root", job["videomamba_root"]))
    for key, value in plan["settings"].items():
        command.extend(("--" + key.replace("_", "-"), str(value)))
    return command


def preflight(plan: dict[str, Any]) -> None:
    import torch
    from cvm.protocol import load_protocol
    from cvm.models import build_model
    from cvm.train import CachedProtocolDataset, model_arguments
    if source_digest() != plan["code_hash"]:
        raise RuntimeError("implementation changed after planning; create a new plan")
    protocol = load_protocol(Path(plan["protocol"]))
    if protocol.protocol_sha256 != plan["protocol_sha256"]:
        raise RuntimeError("protocol changed after planning")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no jobs started")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        raise RuntimeError("unset inherited CUDA_VISIBLE_DEVICES; the launcher assigns each job's physical GPU explicitly")
    for identifier in plan["gpus"]:
        index = int(identifier)
        if index >= torch.cuda.device_count():
            raise RuntimeError("a declared GPU is unavailable; no jobs started")
        if plan["settings"]["amp"] == "bfloat16":
            with torch.cuda.device(index):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("a declared GPU lacks bf16 support; no silent fallback")
    for job in plan["jobs"]:
        if Path(job["output"]).exists():
            raise FileExistsError("a planned run already exists; overwriting/resume is prohibited")
        if not job["checkpoint_sha256"] or not Path(job["checkpoint"]).is_file() or sha256_file(Path(job["checkpoint"])) != job["checkpoint_sha256"]:
            raise RuntimeError("a declared checkpoint is absent or changed; create a new ready plan before any jobs")
    CachedProtocolDataset(protocol, "train", Path(plan["cache_root"]))
    CachedProtocolDataset(protocol, "val", Path(plan["cache_root"]))
    # Validate ALL formal checkpoints and dependencies before launching any job.
    tested = set()
    for job in plan["jobs"]:
        key = (job["backbone"], job["checkpoint"], job.get("videomamba_root"))
        if key not in tested:
            config = {**job, "allow_download": False}
            if job.get("taxonomy_file"):
                definition = json.loads(Path(job["taxonomy_file"]).read_text(encoding="utf-8"))
                if definition.get("protocol_sha256") != protocol.protocol_sha256 or definition.get("training_manifest_sha256") != sha256_file(protocol.manifest_paths["train"]) or definition.get("provenance", {}).get("split") != "train":
                    raise RuntimeError("custom visual taxonomy lacks matching train-only provenance")
                config["taxonomy_definition"] = definition
            model = build_model(**model_arguments(config, pretrained=True))
            del model
            gc.collect()
            tested.add(key)


def execute_plan(plan_path: Path, *, python: str, execute: bool) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "fsn-cvm-plan-1":
        raise ValueError("unsupported plan schema")
    if not execute:
        print(json.dumps({"plan": str(plan_path), "jobs": len(plan["jobs"]), "training_started": False,
                          "requires": "--execute", "stage": plan["stage"]}), flush=True)
        return
    preflight(plan)
    root = plan_path.resolve().parent
    lock = root / ".launch_once"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"launcher_pid": os.getpid(), "plan_sha256": sha256_file(plan_path),
                                 "code_hash": plan["code_hash"], "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    logs = root / "launcher_logs"
    logs.mkdir(exist_ok=False)
    (root / "run_commit.txt").write_text(f"git_commit={plan.get('git_commit')}\ncode_hash={plan['code_hash']}\n", encoding="utf-8")
    queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, job in enumerate(plan["jobs"]):
        queues[plan["gpus"][index % len(plan["gpus"])]].append(job)
    stop = threading.Event()
    mutex = threading.Lock()
    active: dict[str, subprocess.Popen] = {}
    exits: list[dict[str, Any]] = []

    def handle_stop(_number: int, _frame: Any) -> None:
        stop.set()
        with mutex:
            for process in active.values():
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)

    old_handlers = {number: signal.signal(number, handle_stop) for number in (signal.SIGINT, signal.SIGTERM)}

    def worker(gpu: str) -> None:
        for job in queues[gpu]:
            if stop.is_set():
                break
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
            log_path = logs / (job["name"] + ".log")
            with log_path.open("x", encoding="utf-8") as stream:
                process = subprocess.Popen(trainer_command(plan, job, python), cwd=ROOT, env=environment,
                                           stdout=stream, stderr=subprocess.STDOUT)
                with mutex:
                    active[gpu] = process
                    with (logs / "pids.tsv").open("a", encoding="utf-8") as handle:
                        handle.write(f"{job['name']}\t{gpu}\t{process.pid}\n")
                    if stop.is_set() and process.poll() is None:
                        process.send_signal(signal.SIGINT)
                return_code = process.wait()
            with mutex:
                active.pop(gpu, None)
                exits.append({"name": job["name"], "gpu": gpu, "exit_code": return_code})
                with (logs / "exit_codes.tsv").open("a", encoding="utf-8") as handle:
                    handle.write(f"{job['name']}\t{gpu}\t{return_code}\n")
            if return_code != 0:
                # Keep other currently running jobs intact; don't start queued jobs.
                stop.set()
                break

    try:
        with ThreadPoolExecutor(max_workers=len(plan["gpus"])) as executor:
            futures = [executor.submit(worker, gpu) for gpu in plan["gpus"] if queues[gpu]]
            for future in futures:
                future.result()
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)
        complete = len(exits) == len(plan["jobs"]) and all(item["exit_code"] == 0 for item in exits)
        atomic_json(root / "launcher_result.json", {"status": "completed" if complete else "stopped_or_failed",
                                                    "all_jobs_completed": complete, "exits": exits,
                                                    "planned_jobs": len(plan["jobs"]), "test_evaluated": False})
    if not complete:
        raise SystemExit(1)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    for name in ("protocol", "cache-root", "output"):
        plan.add_argument("--" + name, type=Path, required=True)
    plan.add_argument("--weights-json", "--checkpoints", dest="weights_json", type=Path, required=True)
    plan.add_argument("--stage", "--phase", dest="stage", choices=("smoke", "pilot", "formal"), default="pilot")
    plan.add_argument("--videomamba-root", type=Path)
    plan.add_argument("--gpus", default="0,1,2,3")
    plan.add_argument("--backbones", help="Declared feasibility subset; a subset is marked incomplete for formal claims")
    plan.add_argument("--taxonomy-file", type=Path)
    plan.add_argument("--batch-size", type=int, default=2)
    plan.add_argument("--accum-steps", type=int, default=16)
    plan.add_argument("--epochs", type=int, default=60)
    plan.add_argument("--warmup-epochs", type=int, default=5)
    plan.add_argument("--patience", type=int, default=10)
    plan.add_argument("--backbone-lr", type=float, default=1e-5)
    plan.add_argument("--head-lr", type=float, default=1e-4)
    plan.add_argument("--warmup-lr", type=float, default=1e-3)
    plan.add_argument("--weight-decay", type=float, default=.05)
    plan.add_argument("--class-weights", choices=("none", "sqrt_inverse", "inverse"), default="none")
    plan.add_argument("--workers", type=int, default=4)
    plan.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    plan.add_argument("--amp", choices=("none", "bfloat16", "float16"), default="bfloat16")
    plan.add_argument("--clip-grad", type=float, default=5.)
    plan.add_argument("--augmentation", choices=("none", "horizontal_flip"), default="horizontal_flip")
    launch = sub.add_parser("launch")
    launch.add_argument("--plan", type=Path, required=True)
    launch.add_argument("--python", default=sys.executable)
    launch.add_argument("--execute", action="store_true")
    return command


def main() -> None:
    args = parser().parse_args()
    if args.command == "plan":
        generate_plan(args)
    else:
        execute_plan(args.plan, python=args.python, execute=args.execute)


if __name__ == "__main__":
    main()
