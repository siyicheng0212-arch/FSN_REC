"""Default is a read-only plan; explicit execute queues four independent cells."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from experiments.anchor_reference import run_suite as shared
from experiments.relation.data import sha256_file
from experiments.relation.tcn_baseline import TCNConfig
from experiments.videomae_tcn.io import ROOT, append_line, atomic_json, write_json, write_text
from experiments.videomae_tcn.run import fresh_output, wait_released_gpu
from .model import VARIANTS, InputEvidenceTCN, input_policy

CONFIG_PATH = Path(__file__).with_name("config.json")
CONFIG = json.loads(CONFIG_PATH.read_text())


def waves(gpus, uuids):
    if len(gpus) not in (1, 2, 3, 4) or len(set(gpus)) != len(gpus) or \
            len(uuids) != len(gpus) or len(set(uuids)) != len(uuids) or any(g < 0 for g in gpus):
        raise ValueError("use 1-4 unique physical indices and matching distinct UUIDs")
    return [[{"variant": v, "gpu": gpus[i], "gpu_uuid": uuids[i]}
             for i, v in enumerate(VARIANTS[start:start + len(gpus)])]
            for start in range(0, len(VARIANTS), len(gpus))]


def plan(args):
    # Reuse the same full-split source/index and historical-checkpoint verifier.
    # It neither queries a GPU nor creates output. No template/new baseline is
    # accepted instead of the genuine historical assets.
    p = shared.plan(args)
    gpus = [int(x) for x in args.gpus.split(",")]
    uuids = args.gpu_uuids.split(",")
    if any(not re.fullmatch(r"GPU-[a-fA-F0-9-]{8,}", u) for u in uuids):
        raise ValueError("require UUIDs measured after the actual server boot")
    p.update({"schema": CONFIG["schema"], "training": deepcopy(CONFIG),
              "shared_config_sha256": p["config_sha256"],
              "config_sha256": sha256_file(CONFIG_PATH),
              "variants": list(VARIANTS), "waves": waves(gpus, uuids),
              "parameters": {v: InputEvidenceTCN(TCNConfig(**p["evidence"]["model_config"]), v).parameter_counts()
                             for v in VARIANTS},
              "input_policies": {v: input_policy(v) for v in VARIANTS},
              "no_new_gate": True, "existing_train_features_remain_in_sample": True,
              "historical_TCN_is_replayed_but_not_used_as_initialization": True,
              "follow_up_training_automatic": False})
    return p


def unchanged(p):
    if p["schema"] != CONFIG["schema"] or p["training"] != CONFIG or p["variants"] != list(VARIANTS):
        raise RuntimeError("registered diagnostic recipe/cells changed")
    if sha256_file(CONFIG_PATH) != p["config_sha256"]:
        raise RuntimeError("fixed diagnostic configuration changed")
    shared.unchanged({**p, "config_sha256": p["shared_config_sha256"]})


def validate_training_result(out, variant):
    directory = out / variant
    result = json.loads((directory / "result.json").read_text())
    history = json.loads((directory / "history.json").read_text())
    if result.get("training_completed") is not True or result.get("epochs_completed") != CONFIG["epochs"] or \
            len(history) != CONFIG["epochs"] or result.get("variant") != variant or \
            result.get("input_policy") != input_policy(variant) or not (directory / "best.pt").is_file() or \
            sha256_file(directory / "best.pt") != result.get("best_sha256"):
        raise RuntimeError("exit0 is not sufficient: complete sealed training artifacts are required")


def execute(p):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("unset outer CUDA_VISIBLE_DEVICES; each worker gets one physical UUID")
    unchanged(p)
    for slot in p["waves"][0]:
        wait_released_gpu(slot["gpu"], slot["gpu_uuid"])
    out = Path(p["output"])
    fresh_output(out, Path(p["index"]), Path(p["source_protocol_dir"]))
    out.mkdir(mode=0o700, exist_ok=False)
    write_text(out / ".launch_once", str(os.getpid()) + "\n")
    write_text(out / "run_commit.txt", p["source_commit"] + "\n")
    write_json(out / "run_protocol.json", {**p, "launcher_pid": os.getpid()})
    logs = out / "launcher_logs"
    logs.mkdir(mode=0o700)
    write_text(logs / "pids.tsv", "stage\tvariant\tpid\tphysical_gpu\tgpu_uuid\n")
    write_text(logs / "exit_codes.tsv", "stage\tvariant\texit_code\n")
    exits = {}

    def launch(stage, slot=None):
        unchanged(p)
        if slot is not None:
            wait_released_gpu(slot["gpu"], slot["gpu_uuid"])
        variant = slot["variant"] if slot else "all"
        if stage == "unit_tests":
            command = [p["python"], "-m", "unittest", "tests.test_input_evidence",
                       "tests.test_anchor_reference", "tests.test_relation_tcn_baseline", "-q"]
        else:
            command = [p["python"], "-m", "experiments.input_evidence.worker", "--stage", stage,
                       "--run-protocol", str(out / "run_protocol.json")]
            if slot:
                command += ["--variant", variant]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   CUDA_VISIBLE_DEVICES="" if slot is None else slot["gpu_uuid"])
        tag = stage + "_" + variant
        with (logs / (tag + ".log")).open("x") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        append_line(logs / "pids.tsv", f"{stage}\t{variant}\t{child.pid}\t"
                    f"{slot['gpu'] if slot else 'cpu'}\t{slot['gpu_uuid'] if slot else 'none'}")
        print(json.dumps({"stage": stage, "variant": variant, "pid": child.pid,
                          "physical_gpu": slot["gpu"] if slot else "cpu",
                          "gpu_uuid": slot["gpu_uuid"] if slot else None}), flush=True)
        return stage, variant, tag, child

    def collect(children):
        failed = []
        for stage, variant, tag, child in children:
            rc = child.wait()
            exits[tag] = rc
            append_line(logs / "exit_codes.tsv", f"{stage}\t{variant}\t{rc}")
            if rc:
                failed.append((tag, rc))
        if failed:
            raise RuntimeError(f"failed stages {failed}; peers finished naturally, later waves stopped")

    def group(stage, items):
        children = []
        try:
            for slot in items:
                children.append(launch(stage, slot))
        except BaseException:
            collect(children)
            raise
        collect(children)

    try:
        atomic_json(out / "status.json", {"status": "running", "stage": "unit_tests", "training_completed": False})
        collect([launch("unit_tests")])
        collect([launch("audit")])
        for i, wave in enumerate(p["waves"]):
            atomic_json(out / "status.json", {"status": "running", "stage": "smoke", "wave": i + 1,
                                              "training_completed": False})
            group("smoke", wave)
            for slot in wave:
                result = json.loads((out / "smoke" / (slot["variant"] + ".json")).read_text())
                if result.get("all_passed") is not True or result.get("input_policy") != input_policy(slot["variant"]):
                    raise RuntimeError("all four real-feature CUDA smokes must pass before training")
        for i, wave in enumerate(p["waves"]):
            atomic_json(out / "status.json", {"status": "running", "stage": "train", "wave": i + 1,
                                              "training_completed": False})
            group("train", wave)
            for slot in wave:
                validate_training_result(out, slot["variant"])
        group("evaluate", [{**p["waves"][0][0], "variant": "all"}])
        report = json.loads((out / "evaluation/public_safe_summary.json").read_text())
        if report.get("training_completed") is not True or set(report["models"]) != set(VARIANTS):
            raise RuntimeError("four-cell evaluation is incomplete")
        write_json(out / "suite_result.json", {"status": "complete", "training_completed": True,
                                               "exit_codes": exits, "test_read": False})
        atomic_json(out / "status.json", {"status": "completed", "training_completed": True, "exit_code": 0})
    except BaseException as error:
        atomic_json(out / "status.json", {"status": "failed", "training_completed": False,
                                          "exit_codes": exits, "error_type": type(error).__name__, "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "source-protocol-dir", "historical-run", "output", "expected-commit", "gpu-uuids"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--backbone", choices=("videomae2", "adafocus"), required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    p = plan(args)
    if args.execute:
        execute(p)
    else:
        print(json.dumps(p, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
