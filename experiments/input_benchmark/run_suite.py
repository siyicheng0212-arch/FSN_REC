"""One-boot bounded queue: private assets, read-only plan, explicit execution."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import torch

from experiments.input_evidence import run_suite as input_run
from experiments.relation.data import sha256_file
from experiments.relation.tcn_baseline import TCNConfig
from experiments.videomae_tcn.io import ROOT, append_line, atomic_json, write_json, write_text
from experiments.videomae_tcn.run import fresh_output, wait_released_gpu
from .model import CONFIG, CONFIG_PATH, TASKS, BenchmarkHead

RECIPE_KEYS = ("seed", "epochs", "early_stopping", "batch_size_segments", "optimizer", "learning_rate",
               "minimum_learning_rate", "scheduler", "weight_decay", "clip_gradient_norm", "select_best", "replacement_seed")
VENDOR = ROOT / "experiments/temporal_heads/vendor"


def verify_reuse(reference, task, p):
    """Never infer equivalence from a score or silently reuse a different recipe."""
    if task not in ("logits", "logits_aug", "visual", "visual_aug") or not isinstance(reference, dict) or \
            set(reference) != {"run", "variant"} or not isinstance(reference["run"], str) or \
            not Path(reference["run"]).is_absolute() or not isinstance(reference["variant"], str):
        raise ValueError("only completed identical native cells may be explicitly reused")
    run, old_variant = Path(reference["run"]).resolve(), reference["variant"]
    out = Path(p["verification_output"])
    if run.is_relative_to(out) or out.is_relative_to(run):
        raise ValueError("new output overlaps an explicitly reused old run")
    suite = json.loads((run / "suite_result.json").read_text())
    old = json.loads((run / "run_protocol.json").read_text())
    input_format = old.get("schema") == "fsn-input-evidence-native-v1"
    anchor_format = old.get("schema") == "fsn-anchor-reference-native-v1"
    allowed = old_variant == task if input_format else \
        (anchor_format and {"logits": "plain", "logits_aug": "plain_aug"}.get(task) == old_variant)
    if not allowed or suite.get("status") != "complete" or suite.get("training_completed") is not True or \
            not suite.get("exit_codes") or any(code != 0 for code in suite["exit_codes"].values()):
        raise ValueError("reuse requires a complete supported suite with all stage exits zero")
    if old.get("evidence") != p["evidence"] or old.get("backbone") != p["backbone"] or \
            old.get("test_read") is not False or any(old["training"].get(k) != CONFIG[k] for k in RECIPE_KEYS) or \
            old["training"].get("loss") != "0.5_normal_clip_CE_plus_0.5_preserved_anchor_CE":
        raise ValueError("reuse A/input seals or common training recipe do not match")
    critical = ["experiments/anchor_reference/model.py", "experiments/anchor_reference/data.py",
                "experiments/anchor_reference/worker.py", "experiments/relation/tcn_baseline.py"]
    if input_format:
        critical += ["experiments/input_evidence/model.py", "experiments/input_evidence/worker.py"]
    if any(old.get("source_sha256", {}).get(name) != sha256_file(ROOT / name) for name in critical):
        raise ValueError("reuse implementation is not the verified identical native computation/trainer")
    directory = run / old_variant
    result = json.loads((directory / "result.json").read_text())
    history = json.loads((directory / "history.json").read_text())
    ck = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    steps = CONFIG["epochs"] * math.ceil(p["evidence"]["segments"]["train"]["count"] / CONFIG["batch_size_segments"])
    if result.get("training_completed") is not True or result.get("epochs_completed") != CONFIG["epochs"] or \
            result.get("optimizer_steps") != steps or [r.get("epoch") for r in history] != list(range(1, 61)) or \
            history[-1].get("cumulative_steps") != steps or ck.get("variant") != old_variant or \
            ck.get("schema") != old["schema"] or ck.get("source_commit") != old.get("source_commit") or \
            ck.get("training") != old["training"] or ck.get("input_sha256") != p["evidence"]["input_sha256"] or \
            ck.get("epoch") != result.get("best_epoch") or ck.get("macro_f1") != result.get("best_macro_f1") or \
            ck.get("accuracy") != result.get("best_accuracy_same_F1_checkpoint") or \
            sha256_file(directory / "best.pt") != result.get("best_sha256"):
        raise ValueError("reuse requires sealed 60-epoch history, best checkpoint and complete result")
    # Strict state coverage, not an adapter silently dropping shared tensors.
    BenchmarkHead(TCNConfig(**p["evidence"]["model_config"]), task).core.load_state_dict(ck["model"], strict=True)
    names = ["suite_result.json", "run_protocol.json", "audit_seal.json", "private_plans.json",
             old_variant + "/best.pt", old_variant + "/result.json", old_variant + "/history.json"]
    files = {str(run / name): sha256_file(run / name) for name in names}
    return {"run": str(run), "variant": old_variant, "source_commit": old["source_commit"],
            "checkpoint": str(directory / "best.pt"), "result": str(directory / "result.json"),
            "audit_seal": str(run / "audit_seal.json"), "artifacts": files, "new_training": False}


def waves(jobs, gpus, uuids):
    # Reuse the established inventory syntax validation without querying GPUs.
    input_run.waves(gpus, uuids)
    return [[{**job, "gpu": gpus[i], "gpu_uuid": uuids[i]}
             for i, job in enumerate(jobs[start:start + len(gpus)])]
            for start in range(0, len(jobs), len(gpus))]


def plan(args):
    assets_path = Path(args.assets).resolve()
    assets = json.loads(assets_path.read_text())
    if not isinstance(assets, list) or not assets or len(assets) > 2 or \
            any(not isinstance(row, dict) for row in assets) or \
            any(set(row) - {"backbone", "index", "source_protocol_dir", "historical_run", "reuse"} for row in assets) or \
            any(not isinstance(row.get("backbone"), str) for row in assets) or \
            len({row.get("backbone") for row in assets}) != len(assets) or \
            any(row.get("backbone") not in ("videomae2", "adafocus") for row in assets):
        raise ValueError("private assets must explicitly name one or both supported frozen backbones")
    gpus, uuids = [int(x) for x in args.gpus.split(",")], args.gpu_uuids.split(",")
    if any(not re.fullmatch(r"GPU-[a-fA-F0-9-]{8,}", value) for value in uuids):
        raise ValueError("measure current physical UUIDs, never reuse boot-stale mappings")
    waves([], gpus, uuids)
    output = Path(args.output).resolve()
    if assets_path.is_relative_to(output):
        raise ValueError("private asset file and new output must not overlap")
    subplans, reuse = {}, {}
    for row in assets:
        required = ("index", "source_protocol_dir", "historical_run")
        if any(not isinstance(row.get(k), str) or not Path(row[k]).is_absolute() for k in required):
            raise ValueError("use actual absolute private asset paths")
        p = input_run.plan(SimpleNamespace(**{k: row[k] for k in required}, backbone=row["backbone"],
            output=str(output), expected_commit=args.expected_commit, gpus=args.gpus,
            gpu_uuids=args.gpu_uuids, python=args.python))
        p["verification_output"], p["output"] = str(output), str(output / row["backbone"])
        subplans[row["backbone"]] = p
        if not isinstance(row.get("reuse", {}), dict):
            raise ValueError("explicit reuse must be a task-to-run mapping")
        reuse[row["backbone"]] = {task: verify_reuse(ref, task, p) for task, ref in row.get("reuse", {}).items()}
    if len(subplans) == 2:
        # Different frozen encoders must not quietly mean different full splits.
        source_keys = ("source_audit", "edges.jsonl", "chains_train.jsonl", "chains_val.jsonl", "manifest_metadata.jsonl")
        seals = [{k: child["evidence"]["input_sha256"][k] for k in source_keys} for child in subplans.values()]
        if seals[0] != seals[1]:
            raise ValueError("two-backbone comparison requires byte-identical sealed full source protocol")
    jobs = [{"backbone": backbone, "variant": task} for task in TASKS for backbone in subplans]
    train_jobs = [j for j in jobs if j["variant"] not in reuse[j["backbone"]]]
    return {"schema": CONFIG["schema"], "source_commit": args.expected_commit, "training": deepcopy(CONFIG),
            "registry": deepcopy(TASKS), "config_sha256": sha256_file(CONFIG_PATH),
            "assets_file": str(assets_path), "assets_sha256": sha256_file(assets_path),
            "vendor_sha256": {str(f.relative_to(ROOT)): sha256_file(f) for f in sorted(VENDOR.iterdir()) if f.is_file()},
            "output": str(output), "python": args.python, "subplans": subplans, "reuse": reuse,
            "smoke_waves": waves(jobs, gpus, uuids), "train_waves": waves(train_jobs, gpus, uuids),
            "evaluation_slots": [{"backbone": b, "variant": "all", "gpu": gpus[0], "gpu_uuid": uuids[0]} for b in subplans],
            "task_counts": {"registered": len(jobs), "new_training": len(train_jobs), "reused": len(jobs) - len(train_jobs)},
            "parameters": {b: {v: BenchmarkHead(TCNConfig(**p["evidence"]["model_config"]), v).parameter_counts()
                                for v in TASKS} for b, p in subplans.items()},
            "contains_private_paths": True, "test_read": False, "ddp": False,
            "plan_does_not_probe_GPU_or_create_output": True, "automatic_additional_backbones_or_seeds": False}


def unchanged(p):
    if p["schema"] != CONFIG["schema"] or p["training"] != CONFIG or p["registry"] != TASKS or \
            sha256_file(CONFIG_PATH) != p["config_sha256"] or sha256_file(p["assets_file"]) != p["assets_sha256"]:
        raise RuntimeError("registered code/config/private assets changed")
    for child in p["subplans"].values():
        input_run.unchanged(child)
    if any(sha256_file(ROOT / f) != h for f, h in p["vendor_sha256"].items()):
        raise RuntimeError("attributed upstream code or license changed")
    for refs in p["reuse"].values():
        for ref in refs.values():
            if any(sha256_file(f) != h for f, h in ref["artifacts"].items()):
                raise RuntimeError("explicitly reused immutable result changed")


def execute(p):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("unset outer CUDA_VISIBLE_DEVICES; independent workers use one UUID")
    unchanged(p)
    for slot in p["smoke_waves"][0]:
        wait_released_gpu(slot["gpu"], slot["gpu_uuid"])
    out = Path(p["output"])
    for child in p["subplans"].values():
        fresh_output(out, Path(child["index"]), Path(child["source_protocol_dir"]))
    out.mkdir(mode=0o700, exist_ok=False)
    for backbone in p["subplans"]:
        (out / backbone).mkdir(mode=0o700)
    write_text(out / ".launch_once", str(os.getpid()) + "\n")
    write_text(out / "run_commit.txt", p["source_commit"] + "\n")
    write_json(out / "run_protocol.json", {**p, "launcher_pid": os.getpid()})
    logs = out / "launcher_logs"
    logs.mkdir(mode=0o700)
    write_text(logs / "pids.tsv", "stage\tbackbone\tvariant\tpid\tphysical_gpu\tgpu_uuid\n")
    write_text(logs / "exit_codes.tsv", "stage\tbackbone\tvariant\texit_code\n")
    exits = {}

    def launch(stage, slot):
        unchanged(p)
        if "gpu" in slot:
            wait_released_gpu(slot["gpu"], slot["gpu_uuid"])
        b, v = slot["backbone"], slot["variant"]
        tag = stage + "_" + b + "_" + v
        if stage == "unit_tests":
            command = [p["python"], "-m", "unittest", "tests.test_input_benchmark", "tests.test_input_evidence",
                       "tests.test_temporal_heads", "tests.test_anchor_reference", "-q"]
        else:
            command = [p["python"], "-m", "experiments.input_benchmark.worker", "--run-protocol",
                       str(out / "run_protocol.json"), "--stage", stage, "--backbone", b, "--variant", v]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   CUDA_VISIBLE_DEVICES=slot.get("gpu_uuid", ""))
        with (logs / (tag + ".log")).open("x") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        append_line(logs / "pids.tsv", f"{stage}\t{b}\t{v}\t{child.pid}\t{slot.get('gpu', 'cpu')}\t{slot.get('gpu_uuid', 'none')}")
        print(json.dumps({"stage": stage, **slot, "pid": child.pid}), flush=True)
        return stage, b, v, tag, child

    def collect(children):
        failed = []
        for stage, b, v, tag, child in children:
            rc = child.wait()
            exits[tag] = rc
            append_line(logs / "exit_codes.tsv", f"{stage}\t{b}\t{v}\t{rc}")
            if rc:
                failed.append((tag, rc))
        if failed:
            raise RuntimeError(f"failed {failed}; peers finished naturally and later waves stopped")

    def group(stage, wave):
        children = []
        try:
            for slot in wave:
                children.append(launch(stage, slot))
        except BaseException:
            collect(children)
            raise
        collect(children)

    try:
        atomic_json(out / "status.json", {"status": "running", "stage": "unit_tests", "training_completed": False})
        group("unit_tests", [{"backbone": "all", "variant": "all"}])
        for b in p["subplans"]:
            atomic_json(out / "status.json", {"status": "running", "stage": "audit", "backbone": b, "training_completed": False})
            group("audit", [{"backbone": b, "variant": "all"}])
        # Every architecture/input cell passes real CUDA before any training.
        for i, wave in enumerate(p["smoke_waves"]):
            atomic_json(out / "status.json", {"status": "running", "stage": "smoke", "wave": i + 1, "training_completed": False})
            group("smoke", wave)
            for slot in wave:
                report = json.loads((out / slot["backbone"] / "smoke" / (slot["variant"] + ".json")).read_text())
                if report.get("all_passed") is not True or report.get("task_specification") != TASKS[slot["variant"]]:
                    raise RuntimeError("all real CUDA smokes must pass before formal training")
        for i, wave in enumerate(p["train_waves"]):
            atomic_json(out / "status.json", {"status": "running", "stage": "train", "wave": i + 1, "training_completed": False})
            group("train", wave)
            for slot in wave:
                from .worker import validate_completed
                validate_completed(p, slot["backbone"], slot["variant"], torch.device("cpu"))
        for slot in p["evaluation_slots"]:
            atomic_json(out / "status.json", {"status": "running", "stage": "evaluate", "backbone": slot["backbone"], "training_completed": False})
            group("evaluate", [slot])
        atomic_json(out / "status.json", {"status": "running", "stage": "aggregate", "training_completed": False})
        group("aggregate", [{"backbone": "all", "variant": "all"}])
        report = json.loads((out / "public_safe_summary.json").read_text())
        if report.get("training_completed") is not True or set(report["backbones"]) != set(p["subplans"]):
            raise RuntimeError("complete cross-backbone aggregate is required")
        write_json(out / "suite_result.json", {"status": "complete", "training_completed": True, "exit_codes": exits,
                    "registered_tasks": p["task_counts"], "test_read": False})
        atomic_json(out / "status.json", {"status": "completed", "training_completed": True, "exit_code": 0})
    except BaseException as error:
        atomic_json(out / "status.json", {"status": "failed", "training_completed": False, "exit_codes": exits,
                                          "error_type": type(error).__name__, "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "output", "expected-commit", "gpu-uuids"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    p = plan(args)
    if args.execute:
        execute(p)
    else:
        print(json.dumps(p, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
