"""Four independent GPU jobs: two temporal controls at seeds10/1217; reuse42."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from . import run
from experiments.relation.data import sha256_file
from experiments.videomae_tcn.io import ROOT, atomic_json, write_json, write_text
from experiments.videomae_tcn.run import fresh_output, wait_released_gpu


def assignments(indices, uuids):
    if len(indices) != 4 or len(set(indices)) != 4 or len(uuids) != 4 or len(set(uuids)) != 4:
        raise ValueError("four distinct physical GPUs and UUIDs required")
    return [(seed, variant, index, uuid) for (seed, variant), index, uuid in zip(
        [(10, run.PAIRED_VARIANTS[0]), (10, run.PAIRED_VARIANTS[1]),
         (1217, run.PAIRED_VARIANTS[0]), (1217, run.PAIRED_VARIANTS[1])], indices, uuids)]


def seed42_seal(directory):
    suite = json.loads((directory / "suite_result.json").read_text())
    protocol = json.loads((directory / "run_protocol.json").read_text())
    if suite.get("status") != "complete" or not suite.get("training_completed") or any(suite["exit_codes"].values()):
        raise ValueError("seed42 is not a completed suite")
    if protocol["training"]["seed"] != 42:
        raise ValueError("wrong historical seed")
    aggregate = json.loads((directory / "evaluation/public_safe_summary.json").read_text())
    names = ["suite_result.json", "run_protocol.json", "evaluation/public_safe_summary.json"]
    for variant in run.PAIRED_VARIANTS:
        result = json.loads((directory / variant / "result.json").read_text())
        if not result["training_completed"] or result["epochs_completed"] != 30 or result["optimizer_steps"] != 5910:
            raise ValueError("historical budget incomplete")
        if sha256_file(directory / variant / "best.pt") != result["best_sha256"]:
            raise ValueError("historical checkpoint SHA changed")
        if aggregate["variants"][variant]["training_result"] != result or abs(
                aggregate["variants"][variant]["metrics"]["macro_f1"] - result["best_macro_f1"]) > 1e-12:
            raise ValueError("historical aggregate does not match checkpoint result")
        names.extend([variant + "/result.json", variant + "/history.json", variant + "/best.pt"])
    return {n: sha256_file(directory / n) for n in names}


def summary(rows):
    result = {}
    for variant in run.PAIRED_VARIANTS:
        values = [rows[str(seed)][variant]["metrics"]["macro_f1"] for seed in (42, 10, 1217)]
        acc = [rows[str(seed)][variant]["metrics"]["accuracy"] for seed in (42, 10, 1217)]
        result[variant] = {"macro_f1_mean": statistics.mean(values), "macro_f1_sample_std": statistics.stdev(values),
                           "accuracy_mean": statistics.mean(acc), "accuracy_sample_std": statistics.stdev(acc)}
    delta = [rows[str(s)][run.PAIRED_VARIANTS[1]]["metrics"]["macro_f1"] -
             rows[str(s)][run.PAIRED_VARIANTS[0]]["metrics"]["macro_f1"] for s in (42, 10, 1217)]
    result["paired_F1_delta"] = {"seeds": [42, 10, 1217], "values": delta,
                                "mean": statistics.mean(delta), "sample_std": statistics.stdev(delta)}
    return result


def plan(args):
    root, historical = Path(args.output).resolve(), Path(args.seed42_run).resolve()
    fresh_output(root, Path(args.index).resolve(), Path(args.source_protocol_dir).resolve())
    if root.is_relative_to(historical) or historical.is_relative_to(root):
        raise ValueError("outputs overlap historical run")
    seal = seed42_seal(historical)
    children = []
    for seed, variant, gpu, uuid in assignments(args.gpus, args.gpu_uuids):
        # Sibling directories keep the default plan strictly read-only: no output parent is created.
        output = str(root) + f"_seed{seed}_{variant}"
        child_args = argparse.Namespace(**{**vars(args), "output": output, "seed": seed,
                                           "variant": variant, "gpu": gpu, "gpu_uuid": uuid})
        child = run.plan(child_args)
        old = json.loads((historical / "run_protocol.json").read_text())
        if child["evidence"] != old["evidence"] or child["base_sha256"] != old["base_sha256"]:
            raise ValueError("historical/new frozen evidence or initial TCN differs")
        children.append(child)
    return {"output": str(root), "seed42_run": str(historical), "seed42_sha256": seal,
            "children": children, "source_commit": args.expected_commit, "ddp": False,
            "test_read": False, "no_A_retraining": True,
            "stability_scope": "fixed_A42_and_fixed_TCN42_temporal_continuation_only"}


def execute(p):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("unset outer CUDA_VISIBLE_DEVICES")
    historical = Path(p["seed42_run"])
    if seed42_seal(historical) != p["seed42_sha256"]:
        raise RuntimeError("historical seed42 artifacts changed")
    for child in p["children"]:
        run.unchanged(child)
        wait_released_gpu(child["gpu_physical_index"], child["gpu_uuid"])
    root = Path(p["output"])
    root.mkdir(mode=0o700, exist_ok=False)
    write_json(root / "run_protocol.json", p)
    write_text(root / ".launch_once", str(os.getpid()) + "\n")
    processes, exits = [], {}
    try:
        for child in p["children"]:
            key = f"seed{child['training']['seed']}_{child['variants'][0]}"
            # The child repeats seals, CPU tests, historical per-clip replay and real CUDA smoke.
            cmd = [child["python"], "-m", "experiments.context_evidence.run"]
            for flag, value in [("index", child["index"]), ("source-protocol-dir", child["source_protocol_dir"]),
                                ("base-run", child["base_run"]), ("output", child["output"]),
                                ("expected-commit", child["source_commit"]), ("gpu-uuid", child["gpu_uuid"]),
                                ("gpu", child["gpu_physical_index"]), ("seed", child["training"]["seed"]),
                                ("variant", child["variants"][0]), ("python", child["python"])]:
                cmd.extend(["--" + flag, str(value)])
            cmd.append("--execute")
            log = (root / (key + ".log")).open("x")
            worker = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                                      stdout=log, stderr=subprocess.STDOUT)
            log.close()
            processes.append((key, worker, child))
            atomic_json(root / "status.json", {"status": "running", "launcher_pid": os.getpid(),
                "jobs": [{"name": k, "pid": w.pid, "gpu": c["gpu_physical_index"], "uuid": c["gpu_uuid"]}
                         for k, w, c in processes], "training_completed": False})
        for key, worker, child in processes:
            exits[key] = worker.wait()
            atomic_json(root / "exit_codes.json", exits)
        if any(exits.values()):
            raise RuntimeError("a child failed; all other jobs allowed to finish; no retries")
        if seed42_seal(historical) != p["seed42_sha256"]:
            raise RuntimeError("historical seed42 artifacts changed")
        old = json.loads((historical / "evaluation/public_safe_summary.json").read_text())
        rows = {"42": {v: old["variants"][v] for v in run.PAIRED_VARIANTS}, "10": {}, "1217": {}}
        for child in p["children"]:
            run.unchanged(child)
            out = Path(child["output"])
            suite = json.loads((out / "suite_result.json").read_text())
            if suite["status"] != "complete" or any(suite["exit_codes"].values()):
                raise RuntimeError("child completion not verified")
            report = json.loads((out / "evaluation/public_safe_summary.json").read_text())
            variant = child["variants"][0]
            if report["training"]["seed"] != child["training"]["seed"]:
                raise RuntimeError("wrong seed in aggregate")
            rows[str(child["training"]["seed"])][variant] = report["variants"][variant]
        for seed in (10, 1217):
            paired = [c for c in p["children"] if c["training"]["seed"] == seed]
            seals = [json.loads((Path(c["output"]) / "evidence_seal.json").read_text()) for c in paired]
            if any(seals[0][key] != seals[1][key] for key in
                   ("plans_sha256", "segment_orders_sha256", "anchor_orders_sha256")):
                raise RuntimeError("paired training plans differ")
            if any(rows[str(seed)][v]["training_result"]["optimizer_steps"] != 5910 or
                   rows[str(seed)][v]["training_result"]["epochs_completed"] != 30 for v in run.PAIRED_VARIANTS):
                raise RuntimeError("paired actual budget differs")
        safe = {"source_commit": p["source_commit"], "training_completed": True, "seeds": [42, 10, 1217],
                "validation_role": "val823_development_not_independent_test", "test_read": False,
                "stability_scope": p["stability_scope"], "seed42_reused_not_retrained": True,
                "rows": rows, "summary": summary(rows)}
        serialized = json.dumps(safe)
        if any(k in serialized for k in ('"clip_id"', '"recording_key"', '/root/', '/Users/', '"feature_path"')):
            raise RuntimeError("privacy check failed")
        write_json(root / "public_safe_summary.json", safe)
        write_json(root / "suite_result.json", {"status": "complete", "training_completed": True, "exit_codes": exits})
        atomic_json(root / "status.json", {"status": "completed", "training_completed": True})
    except BaseException as error:
        for _, worker, _ in processes:
            worker.wait()  # Never kill an independent normal child.
        atomic_json(root / "status.json", {"status": "failed", "error": str(error), "training_completed": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("index", "source-protocol-dir", "base-run", "output", "expected-commit", "seed42-run"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", type=lambda s: [int(x) for x in s.split(',')], required=True)
    parser.add_argument("--gpu-uuids", type=lambda s: s.split(','), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    p = plan(args)
    if args.execute:
        execute(p)
    else:
        print(json.dumps(p, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
