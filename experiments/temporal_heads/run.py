"""Sealed read-only plan by default; three independent heads, one GPU queue."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.direction_gate import run as old_run
from experiments.relation.data import sha256_file
from experiments.videomae_tcn.io import ROOT, append_line, atomic_json, write_json, write_text
from experiments.videomae_tcn.run import fresh_output, wait_released_gpu
from .model import CONFIG, VARIANTS

SCHEMA = "fsn-frozen-videomae-mature-temporal-heads-v1"
PRIOR_FILES = ("run_protocol.json", "suite_result.json", "evaluation/public_safe_summary.json")


def hashes():
    names = [str(f.relative_to(ROOT)) for f in (ROOT / "experiments/temporal_heads").rglob("*")
             if f.is_file() and "__pycache__" not in f.parts]
    return {**old_run.hashes(), **{n: sha256_file(ROOT / n) for n in names},
            "tests/test_temporal_heads.py": sha256_file(ROOT / "tests/test_temporal_heads.py")}


def unchanged(p):
    if hashes() != p["source_sha256"]:
        raise RuntimeError("temporal-head source changed")
    old_run.unchanged({**p, "source_sha256": {n: p["source_sha256"][n] for n in old_run.hashes()}})
    if {n: sha256_file(Path(p["prior_context_run"]) / n) for n in PRIOR_FILES} != p["prior_sha256"]:
        raise RuntimeError("completed prior context suite changed")


def plan(args):
    p = old_run.plan(args)
    prior = Path(args.prior_context_run).resolve()
    out = Path(p["output"])
    if out.is_relative_to(prior) or prior.is_relative_to(out):
        raise ValueError("fresh output overlaps completed context suite")
    status = json.loads((prior / "suite_result.json").read_text())
    protocol = json.loads((prior / "run_protocol.json").read_text())
    summary = json.loads((prior / "evaluation/public_safe_summary.json").read_text())
    if status.get("status") != "complete" or status.get("training_completed") is not True or \
            not status.get("exit_codes") or any(v != 0 for v in status["exit_codes"].values()) or \
            summary.get("training_completed") is not True or summary.get("test_read") is not False or \
            protocol.get("evidence") != p["evidence"] or protocol.get("base_sha256") != p["base_sha256"]:
        raise ValueError("prior context suite is not complete with the same A and ordinary TCN")
    commands = {"unit_tests": [p["python"], "-m", "unittest", "tests.test_relation_tcn_baseline",
                              "tests.test_videomae_tcn", "tests.test_direction_gate", "tests.test_temporal_heads", "-v"]}
    for stage in ("audit", "smoke", *("train_" + v for v in VARIANTS), "evaluate"):
        commands[stage] = [p["python"], "-m", "experiments.temporal_heads.worker", "--stage", stage,
                           "--run-protocol", str(out / "run_protocol.json")]
    return {**p, "schema": SCHEMA, "source_sha256": hashes(), "training": CONFIG,
            "variants": list(VARIANTS), "commands": commands, "prior_context_run": str(prior),
            "prior_sha256": {n: sha256_file(prior / n) for n in PRIOR_FILES},
            "checkpoint_role": "historical_replay_and_comparison_only_all_new_heads_fresh",
            "architecture_and_loss_differ_not_architecture_causality": True,
            "upstream": {"mstcnpp": "sj-li/MS-TCN2@f423a9e65f4ccb1cd7322eb9f94946a19e787993",
                         "asformer": "ChinaYi/ASFormer@e1bbe4f3ed083748f91467c51a63ac2a8b9277ad"},
            "adaptation": "clip_positions_not_dense_frame_segmentation_exact_length_grouping",
            "optimizer_and_epoch_budget_are_shared_not_compute_or_parameter_matched": True}


def execute(p):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("unset outer CUDA_VISIBLE_DEVICES; launcher uses physical UUID")
    unchanged(p)
    wait_released_gpu(p["gpu_physical_index"], p["gpu_uuid"])
    out = Path(p["output"])
    fresh_output(out, Path(p["index"]), Path(p["source_protocol_dir"]))
    out.mkdir(mode=0o700, exist_ok=False)
    write_text(out / ".launch_once", str(os.getpid()) + "\n")
    write_text(out / "run_commit.txt", p["source_commit"] + "\n")
    write_json(out / "run_protocol.json", {**p, "launcher_pid": os.getpid()})
    logs = out / "launcher_logs"; logs.mkdir(mode=0o700)
    write_text(logs / "pids.tsv", "stage\tpid\tphysical_gpu\tgpu_uuid\n")
    write_text(logs / "exit_codes.tsv", "stage\texit_code\n")
    exits = {}
    try:
        for stage, command in p["commands"].items():
            unchanged(p)
            if stage != "unit_tests":
                wait_released_gpu(p["gpu_physical_index"], p["gpu_uuid"])
            mapping = "cpu" if stage == "unit_tests" else str(p["gpu_physical_index"])
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       CUDA_VISIBLE_DEVICES="" if stage == "unit_tests" else p["gpu_uuid"])
            with (logs / (stage + ".log")).open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                append_line(logs / "pids.tsv", f"{stage}\t{child.pid}\t{mapping}\t{p['gpu_uuid']}")
                atomic_json(out / "status.json", {"status": "running", "stage": stage,
                            "launcher_pid": os.getpid(), "worker_pid": child.pid,
                            "physical_gpu": mapping, "gpu_uuid": p["gpu_uuid"], "training_completed": False})
                print(json.dumps({"stage": stage, "pid": child.pid, "physical_gpu": mapping}), flush=True)
                exits[stage] = child.wait()
            append_line(logs / "exit_codes.tsv", f"{stage}\t{exits[stage]}")
            if exits[stage] != 0:
                raise RuntimeError(f"{stage} failed; preserve output; no automatic retry or later head")
            if stage == "smoke" and not json.loads((out / "cuda_smoke.json").read_text()).get("all_passed"):
                raise RuntimeError("CUDA smoke did not pass")
            if stage.startswith("train_") and not json.loads((out / stage[6:] / "result.json").read_text()).get("training_completed"):
                raise RuntimeError("head result incomplete despite zero exit")
        report = json.loads((out / "evaluation/public_safe_summary.json").read_text())
        if not report.get("training_completed") or set(report["variants"]) != set(VARIANTS):
            raise RuntimeError("aggregate evaluation incomplete")
        write_json(out / "suite_result.json", {"schema": SCHEMA, "status": "complete",
                   "training_completed": True, "exit_codes": exits, "test_read": False})
        atomic_json(out / "status.json", {"status": "completed", "exit_code": 0, "training_completed": True})
    except BaseException as error:
        atomic_json(out / "status.json", {"status": "failed", "training_completed": False,
                    "exit_codes": exits, "error_type": type(error).__name__, "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "source-protocol-dir", "base-run", "prior-context-run", "output", "expected-commit", "gpu-uuid"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    p = plan(parser.parse_args())
    if "--execute" in sys.argv:
        execute(p)
    else:
        print(json.dumps(p, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
