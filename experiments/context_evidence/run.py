"""Default read-only sealed plan; explicit single-GPU, serial --execute."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.direction_gate import run as old_run
from experiments.relation.data import sha256_file
from experiments.videomae_tcn.io import ROOT, append_line, atomic_json, repository_state, write_json, write_text
from experiments.videomae_tcn.run import fresh_output, wait_released_gpu
from .model import VARIANTS

SCHEMA = "fsn-center-preserving-context-evidence-v1"
TRAINING = {"seed": 42, "epochs": 30, "early_stopping": False,
            "batch_size_segments": 16, "optimizer": "AdamW", "lr": .001,
            "weight_decay": .01, "clip_grad_norm": 5.0,
            "loss": "normal_valid_clip_CE_plus_unchanged_anchor_CE_equal_weights",
            "select_best": "val_full_seven_class_macro_f1_earliest_tie",
            "A_frozen": True, "TCN_trainable": True,
            "augmentation": "fixed_train_clinical_same_collection_preferred_different_recording_neighbor",
            "views_and_steps_matched_across_variants": True}


def hashes():
    names = ("experiments/context_evidence/__init__.py", "experiments/context_evidence/model.py",
             "experiments/context_evidence/run.py", "experiments/context_evidence/worker.py",
             "tests/test_context_evidence.py")
    return {**old_run.hashes(), **{n: sha256_file(ROOT / n) for n in names}}


def unchanged(p):
    if hashes() != p["source_sha256"]:
        raise RuntimeError("fixed context-evidence source changed")
    old_run.unchanged({**p, "source_sha256": {n: p["source_sha256"][n] for n in old_run.hashes()}})


def plan(args):
    p = old_run.plan(args)  # exact existing evidence + complete ordinary TCN seals
    commands = {"unit_tests": [p["python"], "-m", "unittest", "tests.test_relation_tcn_baseline",
                              "tests.test_videomae_tcn", "tests.test_direction_gate",
                              "tests.test_context_evidence", "-v"]}
    for stage in ("audit", "smoke", *("train_" + v for v in VARIANTS), "evaluate"):
        commands[stage] = [p["python"], "-m", "experiments.context_evidence.worker", "--stage", stage,
                           "--run-protocol", str(Path(p["output"]) / "run_protocol.json")]
    return {**p, "schema": SCHEMA, "source_sha256": hashes(), "training": TRAINING,
            "variants": list(VARIANTS), "commands": commands,
            "design_role": "exploratory_followup_informed_by_prior_val823_error_audit",
            "coefficient": {"inputs": "six_detached_A_and_context_candidate_confidence_statistics",
                            "parameters": 7, "range": "(0,2)", "initial_value": 1.0,
                            "not_a_connection_probability": True},
            "clip_only": {"hidden_width": 74, "parameters": 148229,
                          "ordinary_TCN_parameters": 148935, "initialization": "fresh_zero_classifier",
                          "not_same_checkpoint_or_exact_functional_capacity": True},
            "context_definition": "A + g * (TCN_full - TCN_single_clip_same_weights)",
            "shared_branch_dropout": True, "context_singleton_exact_A": True}


def execute(p):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("unset outer CUDA_VISIBLE_DEVICES; use physical GPU UUID mapping")
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
                raise RuntimeError(f"{stage} failed; preserve output; no automatic retry")
            if stage == "smoke" and not json.loads((out / "cuda_smoke.json").read_text()).get("all_passed"):
                raise RuntimeError("CUDA smoke did not pass")
            if stage.startswith("train_") and not json.loads((out / stage[6:] / "result.json").read_text()).get("training_completed"):
                raise RuntimeError("training result incomplete despite zero exit")
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
