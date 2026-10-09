"""Read-only sealed plan by default; one-GPU serial execution with --execute."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from experiments.relation.data import sha256_file
from experiments.videomae_tcn.evidence import prepare
from experiments.videomae_tcn.io import (ROOT, append_line, atomic_json, repository_state,
                                        source_hashes as base_source_hashes, write_json, write_text)
from experiments.videomae_tcn.run import check_gpu, fresh_output, wait_released_gpu
from .model import VARIANTS, INITIAL_ALPHA, SHRINK_PSEUDOCOUNT, Q_NAMES

SCHEMA = "fsn-direction-confidence-gate-v1"
TRAINING = {"seed": 42, "epochs": 30, "batch_size_clips": 256, "optimizer": "AdamW",
            "lr": 0.001, "weight_decay": 0.01, "clip_grad_norm": 5.0,
            "loss": "unweighted_seven_class_cross_entropy", "neighbor_augmentation": False,
            "early_stopping": False, "select_best": "val_full_seven_class_macro_f1_earliest_tie",
            "initial_alpha": INITIAL_ALPHA, "shrink_pseudocount": SHRINK_PSEUDOCOUNT}
BASE_FILES = ("run_protocol.json", "suite_result.json", "tcn/result.json", "tcn/best.pt",
              "evaluation/result.json", "evaluation/private_val_predictions.jsonl")


def hashes():
    extra = ("experiments/direction_gate/__init__.py", "experiments/direction_gate/model.py",
             "experiments/direction_gate/run.py", "experiments/direction_gate/worker.py",
             "tests/test_direction_gate.py")
    return {**base_source_hashes(), **{n: sha256_file(ROOT / n) for n in extra}}


def unchanged(p):
    if repository_state() != {"commit": p["source_commit"], "clean": True} or hashes() != p["source_sha256"]:
        raise RuntimeError("fixed source changed")
    index, source, base = map(Path, (p["index"], p["source_protocol_dir"], p["base_run"]))
    paths = {"index": index, "export_protocol": index.parent / "protocol.json",
             "source_audit": source / "audit.json"}
    paths.update({n: source / n for n in ("edges.jsonl", "chains_train.jsonl", "chains_val.jsonl",
                                         "manifest_metadata.jsonl")})
    if {n: sha256_file(f) for n, f in paths.items()} != p["evidence"]["input_sha256"] or \
            {n: sha256_file(base / n) for n in BASE_FILES} != p["base_sha256"]:
        raise RuntimeError("sealed inputs or frozen base artifacts changed")


def plan(args):
    index, source, output, base = map(lambda x: Path(x).resolve(),
                                    (args.index, args.source_protocol_dir, args.output, args.base_run))
    fresh_output(output, index, source)
    if output.is_relative_to(base) or base.is_relative_to(output):
        raise ValueError("new output must not overlap the old TCN run")
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit) or \
            repository_state() != {"commit": args.expected_commit, "clean": True}:
        raise ValueError("require clean independent checkout at full fixed commit")
    if args.gpu < 0 or not re.fullmatch(r"GPU-[0-9a-fA-F-]{8,}", args.gpu_uuid):
        raise ValueError("require physical GPU and UUID")
    python = Path(args.python).resolve()
    if not python.is_file():
        raise ValueError("CUDA Python missing")
    _, _, info = prepare(index, source)
    bp = json.loads((base / "run_protocol.json").read_text())
    br = json.loads((base / "tcn/result.json").read_text())
    bs = json.loads((base / "suite_result.json").read_text())
    be = json.loads((base / "evaluation/result.json").read_text())
    if bs.get("status") != "complete" or bs.get("training_completed") is not True or \
            any(v != 0 for v in bs.get("exit_codes", {}).values()) or \
            set(bs.get("exit_codes", {})) != {"unit_tests", "smoke", "train", "evaluate"} or \
            br.get("training_completed") is not True or be.get("status") != "complete" or \
            be.get("val_clip_count") != info["counts"]["val"] or \
            bp.get("evidence") != info or br.get("test_read") is not False or \
            sha256_file(base / "tcn/best.pt") != br.get("best_checkpoint_sha256"):
        raise ValueError("old ordinary TCN run is incomplete or not the same sealed evidence")
    commands = {"unit_tests": [str(python), "-m", "unittest", "tests.test_relation_tcn_baseline",
                              "tests.test_videomae_tcn", "tests.test_direction_gate", "-v"]}
    for stage in ("diagnose", "smoke", *("train_" + v for v in VARIANTS), "evaluate"):
        commands[stage] = [str(python), "-m", "experiments.direction_gate.worker", "--stage", stage,
                           "--run-protocol", str(output / "run_protocol.json")]
    return {"schema": SCHEMA, "source_commit": args.expected_commit, "source_sha256": hashes(),
            "index": str(index), "source_protocol_dir": str(source), "base_run": str(base),
            "output": str(output), "python": str(python), "gpu_physical_index": args.gpu,
            "gpu_uuid": args.gpu_uuid, "worker_cuda_device": "cuda:0", "ddp": False,
            "base_sha256": {n: sha256_file(base / n) for n in BASE_FILES}, "evidence": info,
            "base_result": br, "training": TRAINING, "variants": list(VARIANTS),
            "q_names": list(Q_NAMES), "commands": commands, "test_read": False,
            "plan_only_does_not_probe_gpu": True, "contains_private_paths": True,
            "challenge": {"seed": 42, "one_replaced_neighbor_per_clinical_anchor": True,
                          "same_collection_preferred": True, "different_recording_required": True,
                          "donor_same_split_only": True, "synthetic_not_real_edit_boundaries": True}}


def execute(p):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("unset outer CUDA_VISIBLE_DEVICES")
    unchanged(p)
    check_gpu(p["gpu_physical_index"], p["gpu_uuid"])
    fresh_output(Path(p["output"]), Path(p["index"]), Path(p["source_protocol_dir"]))
    out = Path(p["output"])
    out.mkdir(mode=0o700, exist_ok=False)
    write_text(out / ".launch_once", str(os.getpid()) + "\n")
    write_text(out / "run_commit.txt", p["source_commit"] + "\n")
    write_json(out / "run_protocol.json", {**p, "launcher_pid": os.getpid()})
    logs = out / "launcher_logs"
    logs.mkdir(mode=0o700)
    write_text(logs / "pids.tsv", "stage\tpid\tphysical_gpu\tgpu_uuid\n")
    write_text(logs / "exit_codes.tsv", "stage\texit_code\n")
    exits = {}
    try:
        for stage, command in p["commands"].items():
            unchanged(p)
            if stage != "unit_tests":
                wait_released_gpu(p["gpu_physical_index"], p["gpu_uuid"])
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       CUDA_VISIBLE_DEVICES="" if stage == "unit_tests" else p["gpu_uuid"])
            with (logs / (stage + ".log")).open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                mapping = "cpu" if stage == "unit_tests" else str(p["gpu_physical_index"])
                append_line(logs / "pids.tsv", f"{stage}\t{child.pid}\t{mapping}\t{p['gpu_uuid']}")
                atomic_json(out / "status.json", {"status": "running", "stage": stage,
                            "launcher_pid": os.getpid(), "worker_pid": child.pid,
                            "physical_gpu": mapping, "gpu_uuid": p["gpu_uuid"], "training_completed": False})
                print(json.dumps({"stage": stage, "pid": child.pid, "physical_gpu": mapping}), flush=True)
                exits[stage] = child.wait()
            append_line(logs / "exit_codes.tsv", f"{stage}\t{exits[stage]}")
            if exits[stage] != 0:
                raise RuntimeError(f"{stage} failed; preserve scene, no retry")
            if stage == "smoke" and json.loads((out / "cuda_smoke.json").read_text())["all_passed"] is not True:
                raise RuntimeError("CUDA smoke report failed")
            if stage.startswith("train_") and json.loads((out / stage[6:] / "result.json").read_text())["training_completed"] is not True:
                raise RuntimeError("gate training incomplete")
        report = json.loads((out / "evaluation/public_safe_summary.json").read_text())
        if report["training_completed"] is not True or set(report["gates"]) != set(VARIANTS):
            raise RuntimeError("evaluation incomplete")
        write_json(out / "suite_result.json", {"status": "complete", "training_completed": True,
                   "exit_codes": exits, "test_read": False})
        atomic_json(out / "status.json", {"status": "completed", "exit_code": 0, "training_completed": True})
    except BaseException as error:
        atomic_json(out / "status.json", {"status": "failed", "training_completed": False,
                    "exit_codes": exits, "error_type": type(error).__name__, "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "source-protocol-dir", "base-run", "output", "expected-commit", "gpu-uuid"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    p = plan(args)
    if args.execute:
        execute(p)
    else:
        print(json.dumps(p, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
