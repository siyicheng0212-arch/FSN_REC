"""Read-only plan by default; execute one fixed VideoMAE + plain TCN run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import torch

from experiments.relation.tcn_baseline import FrozenEvidenceTCN, TCNConfig
from .evidence import SCHEMA, TRAINING, prepare
from .io import (ROOT, append_line, atomic_json, repository_state, source_hashes,
                 unchanged, write_json, write_text)


def fresh_output(output, index, source):
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise FileExistsError("output must be new and its parent must already exist")
    for protected in (ROOT, index.parent, source):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("private output must be separate from code and sealed inputs")


def plan(args):
    output, index, source = (Path(value).resolve() for value in (
        args.output, args.index, args.source_protocol_dir))
    fresh_output(output, index, source)
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit):
        raise ValueError("provide a full 40-character expected commit")
    state = repository_state()
    if state != {"commit": args.expected_commit, "clean": True}:
        raise ValueError("require a clean independent worktree at the exact fixed commit")
    if args.gpu < 0 or not re.fullmatch(r"GPU-[0-9a-fA-F-]{8,}", args.gpu_uuid):
        raise ValueError("provide one nonnegative physical GPU number and its full UUID")
    python = Path(args.python).resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    _, _, evidence = prepare(index, source)
    with torch.random.fork_rng(devices=[]):
        model = FrozenEvidenceTCN(TCNConfig(**evidence["model_config"]))
        parameters = sum(p.numel() for p in model.parameters())
    commands = {"unit_tests": [str(python), "-m", "unittest",
                "tests.test_relation_tcn_baseline", "tests.test_videomae_tcn", "-v"]}
    for stage in ("smoke", "train", "evaluate"):
        commands[stage] = [str(python), "-m", "experiments.videomae_tcn.worker",
                           "--stage", stage, "--run-protocol", str(output / "run_protocol.json")]
    return {"schema": SCHEMA, "source_commit": state["commit"], "source_sha256": source_hashes(),
            "index": str(index), "source_protocol_dir": str(source), "output": str(output),
            "python": str(python), "gpu_physical_index": args.gpu, "gpu_uuid": args.gpu_uuid,
            "gpu_mapping_verified": False, "worker_cuda_device": "cuda:0", "ddp": False,
            "gpu_release_guard": {"compute_pid_allowed": False, "utilization_required_percent": 0,
                                  "maximum_driver_memory_mib": 64, "timeout_seconds": 30,
                                  "consecutive_idle_observations": 2},
            "training": TRAINING, "evidence": evidence, "trainable_TCN_parameters": parameters,
            "A_parameters_trained": 0, "precision": "float32_same_as_old_TCN",
            "commands": commands, "training_completed": False,
            "validation_role": "val823_development_not_independent_test", "test_read": False,
            "contains_private_paths": True, "plan_only_does_not_probe_gpu": True}


class ReleaseTelemetryBusy(RuntimeError):
    """No compute owner, but recent device-use telemetry has not settled."""


def check_gpu(index, expected_uuid):
    inventory = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits"], check=True,
                               capture_output=True, text=True)
    devices = {}
    for line in inventory.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 5:
            raise RuntimeError("unrecognized GPU inventory")
        number = int(fields[0])
        devices[number] = {"physical_index": number, "uuid": fields[1], "name": fields[2],
                           "utilization_percent": int(fields[3]), "memory_used_mib": int(fields[4])}
    gpu = devices.get(index)
    if gpu is None or gpu["uuid"] != expected_uuid or "RTX 3090" not in gpu["name"]:
        raise RuntimeError("requested physical RTX3090 or UUID does not match inventory")
    apps = subprocess.run(["nvidia-smi", f"--id={expected_uuid}", "--query-compute-apps=pid",
                           "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    # Permit driver bookkeeping (e.g. 1 MiB); compute users and active use block launch.
    if apps.stdout.strip():
        raise RuntimeError("requested GPU is busy with a compute process; preserve other tasks and stop")
    if gpu["utilization_percent"] != 0 or gpu["memory_used_mib"] > 64:
        raise ReleaseTelemetryBusy("GPU has no compute process but release telemetry is busy: " +
                                  json.dumps(gpu, sort_keys=True))
    return gpu


def wait_released_gpu(index, expected_uuid):
    """Wait for telemetry after our child exits, never bypassing idle checks.

    Utilization covers a recent sampling window and can stay nonzero just
    after a successful CUDA worker exits. Compute PIDs fail immediately; two
    strictly idle samples are required. Sustained activity stops the run.
    """
    deadline, consecutive, latest = time.monotonic() + 30, 0, None
    while True:
        try:
            latest = check_gpu(index, expected_uuid)
            consecutive += 1
            if consecutive == 2:
                return latest
        except ReleaseTelemetryBusy as error:
            latest, consecutive = str(error), 0
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU did not become strictly idle within 30 seconds: {latest}")
        time.sleep(1)


def execute(fixed):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("unset outer CUDA_VISIBLE_DEVICES; launcher uses one physical GPU UUID")
    unchanged(fixed)
    gpu = check_gpu(fixed["gpu_physical_index"], fixed["gpu_uuid"])
    time.sleep(2)
    check_gpu(fixed["gpu_physical_index"], fixed["gpu_uuid"])
    fresh_output(Path(fixed["output"]), Path(fixed["index"]), Path(fixed["source_protocol_dir"]))
    output = Path(fixed["output"])
    output.mkdir(mode=0o700, exist_ok=False)  # atomic once-only reservation
    write_text(output / ".launch_once", str(os.getpid()) + "\n")
    write_text(output / "run_commit.txt", fixed["source_commit"] + "\n")
    fixed = {**fixed, "gpu_mapping_verified": True, "gpu_inventory": gpu,
             "launcher_pid": os.getpid()}
    write_json(output / "run_protocol.json", fixed)
    logs = output / "launcher_logs"
    logs.mkdir(mode=0o700)
    write_text(logs / "pids.tsv", "stage\tpid\tphysical_gpu\tgpu_uuid\n")
    write_text(logs / "exit_codes.tsv", "stage\texit_code\n")
    exits = {}
    try:
        for stage, command in fixed["commands"].items():
            unchanged(fixed)
            if stage != "unit_tests":
                wait_released_gpu(fixed["gpu_physical_index"], fixed["gpu_uuid"])
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       CUDA_VISIBLE_DEVICES="" if stage == "unit_tests" else fixed["gpu_uuid"])
            with (logs / f"{stage}.log").open("x", encoding="utf-8") as stream:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                mapping = "cpu" if stage == "unit_tests" else str(fixed["gpu_physical_index"])
                uuid = "none" if stage == "unit_tests" else fixed["gpu_uuid"]
                append_line(logs / "pids.tsv", f"{stage}\t{child.pid}\t{mapping}\t{uuid}")
                atomic_json(output / "status.json", {"status": "running", "stage": stage,
                            "launcher_pid": os.getpid(), "worker_pid": child.pid,
                            "physical_gpu": mapping, "gpu_uuid": uuid,
                            "training_completed": False, "exit_code": None})
                print(json.dumps({"stage": stage, "pid": child.pid,
                                  "physical_gpu": mapping, "gpu_uuid": uuid}), flush=True)
                exits[stage] = child.wait()
            append_line(logs / "exit_codes.tsv", f"{stage}\t{exits[stage]}")
            if exits[stage] != 0:
                raise RuntimeError(f"{stage} failed with exit {exits[stage]}; preserve output")
            if stage == "smoke":
                if json.loads((output / "cuda_smoke.json").read_text()).get("all_passed") is not True:
                    raise RuntimeError("CUDA smoke report did not pass")
            if stage == "train":
                run = json.loads((output / "tcn" / "result.json").read_text())
                if run.get("training_completed") is not True:
                    raise RuntimeError("worker exited zero without complete TCN result")
        unchanged(fixed)
        report = json.loads((output / "evaluation" / "result.json").read_text())
        if report.get("status") != "complete" or report.get("val_clip_count") != fixed["evidence"]["counts"]["val"]:
            raise RuntimeError("final best-checkpoint evaluation is incomplete")
        write_json(output / "suite_result.json", {"schema": SCHEMA, "status": "complete",
                   "training_completed": True, "exit_codes": exits, "test_read": False})
        atomic_json(output / "status.json", {"status": "completed", "exit_code": 0,
                    "training_completed": True, "test_read": False})
        return 0
    except BaseException as error:
        atomic_json(output / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt)
                    else "failed", "exit_code": 130 if isinstance(error, KeyboardInterrupt) else 1,
                    "training_completed": False, "completed_stage_exit_codes": exits,
                    "error_type": type(error).__name__, "error": str(error)})
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "source-protocol-dir", "output", "expected-commit", "gpu-uuid"):
        result.add_argument("--" + name, required=True)
    result.add_argument("--gpu", type=int, default=0, help="one physical nvidia-smi GPU index")
    result.add_argument("--python", default=sys.executable)
    result.add_argument("--execute", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    fixed = plan(args)
    if not args.execute:
        print(json.dumps(fixed, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
        return 0
    return execute(fixed)


if __name__ == "__main__":
    raise SystemExit(main())
