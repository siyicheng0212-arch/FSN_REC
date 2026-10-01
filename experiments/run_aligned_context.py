"""Plan and launch the fixed, independent four-GPU aligned-context experiment.

The default invocation performs read-only input checks and prints a plan.  A
formal run additionally needs --execute and a current, successful CUDA smoke
report.  This module never resumes a run or changes data, code, or git state.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable

from experiments.aligned_protocol import DATA_PROTOCOL, EXPECTED_COUNTS, EXPECTED_MANIFEST_SHA

VARIANTS = ("original", "context_plain", "context_aligned", "local_capacity")
QUEUES = {0: ("original",), 1: ("context_plain",), 2: ("context_aligned",), 3: ("local_capacity",)}
# Compatibility aliases use the same shared full-split protocol as the trainer.
KNOWN_MANIFEST_SHA = EXPECTED_MANIFEST_SHA
KNOWN_COUNTS = EXPECTED_COUNTS
KNOWN_CHECKPOINT_SHA = "2dea5c15ce23b3549aeab977774649f0ce8dcbc5637d5d1d1319efc019896fc3"
PROTOCOL = {
    "seed": 42, "epochs": 100, "patience": 10, "batch_size": 4,
    "accumulation_steps": 16, "effective_batch_size": 64, "workers": 4,
    "lr": .002, "weight_decay": .0005, "momentum": .9,
    "global_lr_ratio": .5, "stn_lr_ratio": .2, "temporal_lr_ratio": .2,
    "head_warmup_epochs": 5, "head_warmup_lr": .001,
    "class_weight_mode": "sqrt_inverse", "clip_grad": 20,
    "context_dim": 64, "context_grid": 3, "context_time_scale": .25,
    "context_spatial_scale": 1., "context_lr_ratio": 1.,
    "amp_dtype": "bfloat16", "input_frames": 36, "global_frames": 8,
    "local_frames": 12, "patch_size": 128,
}
NON_CLI_PROTOCOL_FIELDS = {"effective_batch_size", "momentum", "amp_dtype",
                           "input_frames", "global_frames", "local_frames", "patch_size"}


class PreflightError(RuntimeError):
    """A missing or inconsistent run prerequisite; no automatic repair."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def repository_state(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
        except subprocess.CalledProcessError as exc:
            raise PreflightError("Cannot verify the isolated git worktree") from exc
    return {"commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "clean": not git("status", "--porcelain", "--untracked-files=all")}


def runtime_source_hashes() -> dict[str, str]:
    # The trainer and its smoke report use exactly this same source helper.
    from experiments.train_aligned_context import source_hashes
    return source_hashes()


def audit_manifests(directory: Path, expected_sha: dict[str, str] = KNOWN_MANIFEST_SHA,
                    expected_counts: dict[str, int] = KNOWN_COUNTS) -> tuple[dict[str, Any], dict[str, Any]]:
    from experiments.aligned_data import load_aligned_manifest
    records, summary = {}, {}
    for split in ("train", "val"):
        path = directory / f"{split}.jsonl"
        if not path.is_file():
            raise PreflightError(f"Missing {split} manifest; do not rebuild it automatically")
        digest = sha256_file(path)
        if digest != expected_sha[split]:
            raise PreflightError(f"{split} manifest SHA differs from the restored original full train/val protocol")
        try:
            values, _raw = load_aligned_manifest(path, split)
        except (ValueError, RuntimeError, OSError) as exc:
            raise PreflightError(f"Invalid {split} manifest role or group/clip metadata: {exc}") from exc
        if len(values) != expected_counts[split]:
            raise PreflightError(f"{split} count differs from the restored original full train/val protocol")
        ids = [record.clip_id for record in values]
        if len(set(ids)) != len(ids):
            raise PreflightError(f"Duplicate clip_id within {split}")
        labels = Counter(record.label_id for record in values)
        if set(labels) != set(range(7)):
            raise PreflightError(f"{split} must contain all seven classes")
        records[split] = values
        summary[split] = {"sha256": digest, "clips": len(values),
                          "groups": len({record.group_id for record in values}),
                          "class_counts": dict(sorted(labels.items()))}
    for field in ("clip_id", "group_id"):
        overlap = {getattr(record, field) for record in records["train"]} & {
            getattr(record, field) for record in records["val"]}
        if overlap:
            raise PreflightError(f"train/val have overlapping {field}; refusing the run")
    return records, summary


def audit_cache(records: dict[str, Any], cache_root: Path, manifest_dir: Path) -> dict[str, Any]:
    from experiments.aligned_data import cache_split_hint, resolve_cache_record
    from experiments.full_data import cache_paths
    if not cache_root.is_dir():
        raise PreflightError("Missing existing cache; this launcher never decodes or rebuilds data")
    mapping, inventory, counts = [], [], Counter()
    for split in ("train", "val"):
        raw = [json.loads(line) for line in (manifest_dir / f"{split}.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        for record, row in zip(records[split], raw):
            try:
                storage = resolve_cache_record(record, cache_root, 36, 224, cache_split_hint(row))
            except (ValueError, RuntimeError, OSError) as exc:
                raise PreflightError(f"Invalid/missing cache in {split}; preserve the original inputs") from exc
            array, metadata = cache_paths(cache_root, storage)
            stat = array.stat()
            mapping.append({"split": split, "clip_id": record.clip_id, "cache_split": storage.split})
            inventory.append({"split": split, "clip_id": record.clip_id,
                              "cache_split": storage.split, "metadata_sha256": sha256_file(metadata),
                              "array_bytes": stat.st_size, "array_mtime_ns": stat.st_mtime_ns})
            counts[f"{split}->{storage.split}"] += 1
    return {"cache_mapping_sha256": object_sha256(mapping),
            "cache_inventory_sha256": object_sha256(inventory),
            "storage_counts": dict(sorted(counts.items())), "validated_clips": len(mapping),
            "inventory_scope": "all metadata SHA256 plus array sizes and mtimes; not full pixel-content hashing"}


def build_command(variant: str, args: argparse.Namespace) -> list[str]:
    if variant not in VARIANTS:
        raise ValueError("Unknown aligned-context variant")
    command = [str(args.python), "-u", "-m", "experiments.train_aligned_context",
               "--variant", variant, "--manifest-dir", str(args.manifest_dir),
               "--cache-dir", str(args.cache_dir), "--output-dir", str(args.output_dir),
               "--checkpoint", str(args.checkpoint)]
    for key, value in PROTOCOL.items():
        if key in NON_CLI_PROTOCOL_FIELDS:
            continue
        command.extend(["--" + key.replace("_", "-"), str(value)])
    return command


def prepare_plan(args: argparse.Namespace, repo: Path) -> dict[str, Any]:
    for name in ("manifest_dir", "cache_dir", "checkpoint", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir.exists():
        raise PreflightError("Output directory already exists; inspect it and choose a fresh directory")
    if args.output_dir == repo or repo in args.output_dir.parents:
        raise PreflightError("Formal output must be outside the isolated source worktree")
    if not args.checkpoint.is_file():
        raise PreflightError("Missing official SSv2 checkpoint; do not train a random formal initialization")
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != KNOWN_CHECKPOINT_SHA:
        raise PreflightError("Checkpoint SHA differs from the official shared SSv2 initialization")
    records, manifests = audit_manifests(args.manifest_dir)
    cache = audit_cache(records, args.cache_dir, args.manifest_dir)
    source = runtime_source_hashes()
    state = repository_state(repo)
    if args.execute and not state["clean"]:
        raise PreflightError("Execution requires a clean, committed isolated worktree")
    return {"schema": "fsn-aligned-context-suite-v1", "created_unix": time.time(),
            "data_protocol": DATA_PROTOCOL,
            "repository": state, "source_sha256": source,
            "paths": {name: str(getattr(args, name)) for name in
                      ("manifest_dir", "cache_dir", "checkpoint", "output_dir", "python")},
            "manifest_sha256": {split: manifests[split]["sha256"] for split in manifests},
            "manifests": manifests, "checkpoint_sha256": checkpoint_sha, **cache,
            "training": dict(PROTOCOL), "queues": {str(gpu): list(queue) for gpu, queue in QUEUES.items()},
            "commands": {variant: build_command(variant, args) for variant in VARIANTS},
            "comparison": "independent single-GPU training from the same official initialization; not DDP",
            "evaluation": "restored original train7372/val823 validation for model selection; not independent test; no test evaluation",
            "resume": False, "automatic_extra_seeds": False}


def validate_smoke_report(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PreflightError("Missing or unreadable actual CUDA three-step smoke report") from exc
    if report.get("schema") != "fsn-aligned-context-smoke-v1" or report.get("all_passed") is not True:
        raise PreflightError("CUDA smoke report does not establish successful checks")
    if plan.get("data_protocol") != DATA_PROTOCOL or report.get("data_protocol") != DATA_PROTOCOL:
        raise PreflightError("CUDA smoke data_protocol differs from the restored full train/val protocol")
    split_audit = report.get("split_audit")
    if (not isinstance(split_audit, dict) or split_audit.get("counts") != EXPECTED_COUNTS
            or split_audit.get("manifest_sha256") != EXPECTED_MANIFEST_SHA
            or split_audit.get("data_protocol") != DATA_PROTOCOL):
        raise PreflightError("CUDA smoke split_audit does not match the frozen full train/val protocol")
    for field in ("source_sha256", "checkpoint_sha256", "manifest_sha256", "cache_mapping_sha256"):
        if report.get(field) != plan.get(field):
            raise PreflightError(f"Stale CUDA smoke report: {field} differs from the formal run")
    configuration = report.get("configuration")
    if not isinstance(configuration, dict):
        raise PreflightError("CUDA smoke report does not record its actual training configuration")
    for field, expected in PROTOCOL.items():
        if field not in NON_CLI_PROTOCOL_FIELDS and configuration.get(field) != expected:
            raise PreflightError(f"CUDA smoke configuration differs from the formal run: {field}")
    for variant in VARIANTS:
        row = report.get("variants", {}).get(variant, {})
        losses = row.get("losses", [])
        difference = row.get("initial_max_abs_logit_diff")
        expected_device = {"original": 0, "context_plain": 1, "context_aligned": 2, "local_capacity": 3}[variant]
        if (row.get("passed") is not True or row.get("steps") != 3 or len(losses) != 3
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in losses)
                or not isinstance(difference, (int, float)) or not math.isfinite(difference)
                or not 0 <= difference <= 1e-5 or row.get("device_index") != expected_device):
            raise PreflightError(f"Insufficient three-step finite/initial-equality checks for {variant}")
        if variant != "original":
            gradients = row.get("gradient_checks")
            if (not isinstance(gradients, list) or len(gradients) != 3
                    or any(not isinstance(step, dict) for step in gradients)):
                raise PreflightError(f"Missing three staged module gradient checks for {variant}")
            for step in gradients:
                up = step.get("up.weight")
                if (not isinstance(up, (int, float)) or not math.isfinite(up) or up <= 0
                        or any(value is not None and (not isinstance(value, (int, float))
                               or not math.isfinite(value) or value < 0) for value in step.values())):
                    raise PreflightError(f"Invalid or zero projection gradient for {variant}")
            final_downstream = {name: value for name, value in gradients[2].items() if name != "up.weight"}
            if (not final_downstream or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                                           for value in final_downstream.values())
                    or not any(value > 0 for value in final_downstream.values())):
                raise PreflightError(f"No finite downstream module learning by step three for {variant}")
    return report


def gpu_preflight(python: str) -> dict[str, Any]:
    # This process ends before training starts.  All four selected cards must
    # exist, support bf16, and actually be RTX 3090s.
    code = """import json, torch
assert torch.cuda.device_count() == 4, 'Expected exactly four visible CUDA GPUs'
rows=[]
for i in range(4):
    torch.cuda.set_device(i)
    p=torch.cuda.get_device_properties(i)
    assert '3090' in p.name, 'Expected RTX3090'
    assert torch.cuda.is_bf16_supported(), 'GPU does not support bf16'
    rows.append({'index':i,'name':p.name,'memory_bytes':p.total_memory})
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpus':rows}))
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3")
    try:
        hardware = json.loads(subprocess.check_output([python, "-c", code], env=env, text=True))
        devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid",
                                           "--format=csv,noheader,nounits"], text=True)
        selected = {line.split(",", 1)[1].strip() for line in devices.splitlines()
                    if line.split(",", 1)[0].strip() in {"0", "1", "2", "3"}}
        if len(selected) != 4:
            raise PreflightError("Cannot identify all four physical GPU UUIDs")
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                                             "--format=csv,noheader,nounits"], text=True)
        if any(line.split(",", 1)[1].strip() in selected for line in processes.splitlines() if "," in line):
            raise PreflightError("Selected GPU has an existing compute process; inspect it before launch")
        return hardware
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as exc:
        raise PreflightError("Actual CUDA/bf16/idle-device preflight failed") from exc


def claim_output(output: Path, plan: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir()  # atomic ownership: no exist_ok, even for an empty directory
    except FileExistsError as exc:
        raise PreflightError("Output was claimed by another launcher; refusing duplicate launch") from exc
    (output / ".launch_once").write_text(json.dumps({"launcher_pid": os.getpid(),
        "created_unix": time.time(), "commit": plan["repository"]["commit"]}) + "\n")
    (output / "launcher_logs").mkdir()
    (output / "run_protocol.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    (output / "run_commit.txt").write_text(plan["repository"]["commit"] + "\n")
    (output / "launcher_logs" / "pids.tsv").write_text("variant\tgpu\tpid\tstarted_unix\n")
    (output / "launcher_logs" / "exit_codes.tsv").write_text(
        "variant\tgpu\tpid\texit_code\tstatus\tfinished_unix\n")


def result_is_complete(run_dir: Path, variant: str,
                       expected_manifest_sha: dict[str, str] = KNOWN_MANIFEST_SHA,
                       expected_counts: dict[str, int] = KNOWN_COUNTS,
                       expected_protocol: Any = DATA_PROTOCOL) -> tuple[bool, str]:
    for name in ("best.pt", "history.json", "result.json", "load_report.json", "val_predictions.jsonl"):
        if not (run_dir / name).is_file() or (run_dir / name).stat().st_size == 0:
            return False, f"missing_or_empty_{name}"
    try:
        result = json.loads((run_dir / "result.json").read_text())
        history = json.loads((run_dir / "history.json").read_text())
    except (ValueError, OSError):
        return False, "invalid_result_or_history_json"
    score = result.get("best_val_macro_f1")
    if (result.get("variant") != variant or result.get("seed") != 42
            or result.get("training_completed") is not True
            or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1
            or not isinstance(history, list) or not any(row.get("phase") == "finetune" for row in history)
            or result.get("test_metrics") is not None):
        return False, "inconsistent_or_incomplete_training_result"
    audit = result.get("split_audit", {})
    if (result.get("data_protocol") != expected_protocol
            or not isinstance(audit, dict) or audit.get("counts") != expected_counts
            or audit.get("manifest_sha256") != expected_manifest_sha
            or audit.get("data_protocol") != expected_protocol):
        return False, "result_data_protocol_mismatch"
    return True, "complete"


def run_queues(plan: dict[str, Any], repo: Path, output: Path,
               source_getter: Callable[[], dict[str, str]] = runtime_source_hashes,
               state_getter: Callable[[Path], dict[str, Any]] = repository_state,
               popen: Callable[..., Any] = subprocess.Popen) -> dict[str, Any]:
    """Run one independent job on each GPU; a failed job does not stop others."""
    cancelled = threading.Event()
    lock = threading.Lock()
    children: dict[int, Any] = {}
    outcomes: dict[str, Any] = {}
    logs = output / "launcher_logs"

    def append(name: str, line: str) -> None:
        with lock:
            with (logs / name).open("a") as handle:
                handle.write(line + "\n")

    def queue(gpu: int, variants: tuple[str, ...]) -> None:
        for variant in variants:
            if cancelled.is_set():
                outcomes[variant] = {"gpu": gpu, "status": "skipped_cancellation"}
                append("exit_codes.tsv", f"{variant}\t{gpu}\t-\t-\tskipped\t{time.time()}")
                continue
            state = state_getter(repo)
            if (not state["clean"] or state["commit"] != plan["repository"]["commit"]
                    or source_getter() != plan["source_sha256"]):
                outcomes[variant] = {"gpu": gpu, "status": "source_changed_before_launch"}
                append("exit_codes.tsv", f"{variant}\t{gpu}\t-\t-\tsource_changed\t{time.time()}")
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            try:
                with (logs / f"{variant}_seed_42.log").open("x") as handle:
                    process = popen(plan["commands"][variant], cwd=str(repo), env=env,
                                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                    with lock:
                        children[process.pid] = process
                    append("pids.tsv", f"{variant}\t{gpu}\t{process.pid}\t{time.time()}")
                    exit_code = process.wait()
                    with lock:
                        children.pop(process.pid, None)
            except (OSError, subprocess.SubprocessError) as exc:
                outcomes[variant] = {"gpu": gpu, "status": "launch_failed", "error_type": type(exc).__name__}
                append("exit_codes.tsv", f"{variant}\t{gpu}\t-\t-\tlaunch_failed\t{time.time()}")
                continue
            complete, reason = result_is_complete(
                output / variant / "seed_42", variant,
                expected_manifest_sha=plan["manifest_sha256"],
                expected_counts={split: plan["manifests"][split]["clips"] for split in ("train", "val")},
                expected_protocol=plan["data_protocol"],
            )
            completed = exit_code == 0 and complete
            outcomes[variant] = {"gpu": gpu, "pid": process.pid, "exit_code": exit_code,
                                 "status": "complete" if completed else reason if exit_code == 0 else "process_failed"}
            append("exit_codes.tsv", f"{variant}\t{gpu}\t{process.pid}\t{exit_code}\t{outcomes[variant]['status']}\t{time.time()}")

    def stop(signum: int, _frame: Any) -> None:
        cancelled.set()
        with lock:
            current = list(children.values())
        for process in current:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    saved_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            saved_handlers[signum] = signal.signal(signum, stop)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(queue, gpu, variants) for gpu, variants in QUEUES.items()]
            for future in futures:
                future.result()
    finally:
        for signum, handler in saved_handlers.items():
            signal.signal(signum, handler)
    final_state = state_getter(repo)
    source_stable = (final_state["clean"] and final_state["commit"] == plan["repository"]["commit"]
                     and source_getter() == plan["source_sha256"])
    summary = {"schema": "fsn-aligned-context-suite-result-v1", "variants": outcomes,
               "source_stable": source_stable,
               "all_complete": source_stable and len(outcomes) == 4 and all(row["status"] == "complete" for row in outcomes.values()),
               "cancelled": cancelled.is_set(), "automatic_followup": False}
    (output / "suite_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest-dir", "cache-dir", "checkpoint", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true", help="Launch only after a successful actual CUDA smoke")
    parser.add_argument("--smoke-report", type=Path)
    args = parser.parse_args(argv)
    if args.execute and args.smoke_report is None:
        parser.error("--execute requires --smoke-report from the actual CUDA smoke-only run")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    try:
        plan = prepare_plan(args, repo)
        if not args.execute:
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            for gpu, queue in QUEUES.items():
                for variant in queue:
                    print(f"GPU{gpu} {variant}: CUDA_VISIBLE_DEVICES={gpu} {shlex.join(plan['commands'][variant])}")
            return 0
        smoke = validate_smoke_report(args.smoke_report, plan)
        plan["hardware"] = gpu_preflight(str(args.python))
        plan["smoke_report_sha256"] = sha256_file(args.smoke_report)
        plan["smoke_verified"] = smoke["all_passed"]
        claim_output(args.output_dir, plan)
        summary = run_queues(plan, repo, args.output_dir)
        print(json.dumps(summary, indent=2))
        return 0 if summary["all_complete"] else 1
    except PreflightError as exc:
        print(f"Preflight refused launch: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
