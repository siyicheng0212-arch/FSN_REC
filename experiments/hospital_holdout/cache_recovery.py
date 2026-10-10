"""Explicit cache-only recovery: fresh output, bounded CPU decode subprocesses.

No training, GPU decode, resplitting or normalization/sampling changes. Old
arrays are reused only after exact decoded-pixel replay. Private jobs/sidecars
contain identities and paths and must never be committed or published.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from experiments.hospital_holdout.prepare import encoded, sha
from experiments.hospital_holdout.video_data import (
    PREPROCESS, manifest_rows, select_indices, spatial_rgb,
)

ORIGINAL_COMMIT = "af93d380fc8b9b64c34a36f7dcd2ca48de64e104"
PROTOCOL_SHA = "45797864698785a38ff484d8173b399814e97444280756153071272b782b1d84"
SCHEMA = "fsn-explicit-bounded-cache-recovery-v1"
MODULE = "experiments.hospital_holdout.cache_recovery"


def write_new(path, value):
    """Publish a fully flushed new file without replacing any existing target."""
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)  # EEXIST is intentional, never os.replace.
    temporary.unlink()


def cache_name(row):
    return hashlib.sha256(row["clip_id"].encode()).hexdigest() + ".npy"


def check_array(array):
    if array.dtype != np.uint8 or array.shape != (16, 224, 224, 3):
        raise ValueError("cache geometry/dtype differs from sealed RGB16 protocol")


def inspect_old(directory, subset):
    expected = {cache_name(row): row for row in subset}
    found = {}
    for path in Path(directory).iterdir():
        if path.suffix != ".npy":
            continue
        if path.is_symlink() or not path.is_file() or path.name not in expected:
            raise ValueError("unexpected file in old cache; cannot silently ignore")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        check_array(array)
        del array
        found[path.name] = path.resolve()
    return found


def fingerprint(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def assert_fingerprint(path, expected):
    if fingerprint(path) != expected:
        raise ValueError("source video changed since SHA verification")


def chunk_rows(subset, limit):
    if limit < 1 or limit > 16:
        raise ValueError("decode chunk must contain 1..16 clips")
    grouped = OrderedDict()
    for row in subset:
        grouped.setdefault(row["video_path"], []).append(row)
    return [items[i:i + limit] for items in grouped.values()
            for i in range(0, len(items), limit)]


def prepare_plan(args):
    failed = args.failed_run.resolve()
    protocol_dir = failed / "protocol"
    if sha(protocol_dir / "protocol.json") != PROTOCOL_SHA:
        raise ValueError("corrected data protocol SHA changed")
    run = json.loads((failed / "run_protocol.json").read_text())
    status = json.loads((failed / "status.json").read_text())
    if run.get("code_commit") != ORIGINAL_COMMIT or run.get("preprocessing") != PREPROCESS \
            or status.get("status") != "failed" or status.get("training_completed") is not False:
        raise ValueError("old cache must originate in the verified failed, untrained run")
    subsets, protocol = {}, None
    for split in ("train", "val"):
        subsets[split], protocol = manifest_rows(protocol_dir, split)
    old = inspect_old(failed / "train_cache", subsets["train"])
    if any((failed / "val_cache").glob("*.npy")):
        raise ValueError("unexpected old validation cache; audit it separately")
    output = args.output.resolve()
    checkout = Path(__file__).resolve().parents[2]
    for protected in (failed, checkout, protocol_dir):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("recovery output must be separate from old output and checkout")
    if output.exists():
        raise ValueError("recovery output already exists; no automatic retry/resume")
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                    cwd=checkout, text=True).strip()
    if current != args.expected_commit or dirty:
        raise ValueError("fixed clean recovery code required")
    fresh = [row for row in subsets["train"] if cache_name(row) not in old]
    reused = [row for row in subsets["train"] if cache_name(row) in old]
    if args.smoke:
        if not fresh or not reused:
            raise ValueError("smoke requires a missing and an existing training clip")
        # Includes the exact first missing clip where the original run stopped.
        phases = [("train", fresh[:1]), ("train", reused[-1:]),
                  ("val", subsets["val"][:1])]
    else:
        # Finish missing clips first; all old clips still require exact replay
        # before the full cache can be sealed for consumption by training.
        phases = [("train", fresh), ("val", subsets["val"]), ("train", reused)]
    jobs = [(split, chunk) for split, items in phases for chunk in chunk_rows(items, args.chunk_clips)]
    return {"schema": SCHEMA, "read_only": not args.execute, "cache_only": True,
            "training_started": False, "code_commit": current, "python": sys.executable,
            "failed_run": str(failed), "output": str(output), "protocol_sha256": PROTOCOL_SHA,
            "manifest_sha256": protocol["manifest_sha256"],
            "counts": {key: len(value) for key, value in subsets.items()},
            "existing_train_arrays": len(old), "missing_train_arrays": len(fresh),
            "old_reuse_policy": "actual_PTS_and_exact_decoded_pixel_replay_then_hardlink_or_copy",
            "preprocessing": PREPROCESS, "smoke_only": args.smoke,
            "chunk_clips": args.chunk_clips, "decode_jobs": len(jobs),
            "worker_rss_limit_bytes": args.rss_gib * (1 << 30),
            "worker_address_space_limit_bytes": 32 * (1 << 30),
            "worker_timeout_sec": args.timeout_sec, "automatic_retry": False}, \
        subsets, protocol, old, jobs


def save_decoded(row, indices, pts, rgb, array, output, old=None, old_sha=None):
    check_array(array)
    output = Path(output)
    name = cache_name(row)
    target = output / name
    replay = False
    if old is not None:
        old = Path(old)
        if sha(old) != old_sha:
            raise ValueError("old cache changed after inventory hashing")
        prior = np.load(old, mmap_mode="r", allow_pickle=False)
        check_array(prior)
        if not np.array_equal(array, prior):
            raise ValueError("old decoded pixels differ; preserve both and stop")
        del prior
        try:
            os.link(old, target)
        except OSError as exc:
            import errno
            if exc.errno != errno.EXDEV:
                raise
            with old.open("rb") as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
        if sha(target) != old_sha:
            raise ValueError("imported cache SHA mismatch")
        replay = True
    else:
        temporary = output / (name + ".partial")
        with temporary.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        temporary.unlink()
    item = {"clip_id": row["clip_id"], "file": name, "sha256": sha(target),
            "label_id": row["label_id"], "frame_indices": indices.tolist(),
            "actual_frame_pts_sec": np.asarray(pts)[indices].tolist(),
            "repeated_frames": 16 - len(set(indices.tolist())),
            "original_shape": list(rgb.shape), "cached_shape": list(array.shape),
            "imported_from_old": replay, "old_pixel_replay_exact": replay}
    write_new(output / (name + ".entry.json"), item)
    return item


def decode_job(job_file):
    # One reader per short-lived process; no reader LRU or growing long-lived
    # decoder state. CUDA intentionally remains unused by this CPU-only stage.
    import resource
    if sys.platform == "linux":
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        cap = 32 * (1 << 30)
        if hard != resource.RLIM_INFINITY:
            cap = min(cap, hard)
        resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
    from decord import VideoReader, cpu
    import cv2
    cv2.setNumThreads(1)
    job = json.loads(Path(job_file).read_text())
    if not job["rows"] or len(job["rows"]) > 16:
        raise ValueError("invalid bounded decode job")
    path = job["rows"][0]["video_path"]
    if any(row["video_path"] != path for row in job["rows"]):
        raise ValueError("one video per worker required")
    assert_fingerprint(path, job["video_fingerprint"])
    reader = VideoReader(path, num_threads=1, ctx=cpu(0))
    pts = reader.get_frame_timestamp(np.arange(len(reader)))
    for row in job["rows"]:
        indices = select_indices(pts, row["clip_start_sec"], row["clip_end_sec"])
        rgb = reader.get_batch(indices.tolist()).asnumpy()
        array = spatial_rgb(rgb)
        name = cache_name(row)
        reuse = job["reuse"].get(name)
        item = save_decoded(row, indices, pts, rgb, array, job["cache_output"],
                            reuse["path"] if reuse else None, reuse["sha256"] if reuse else None)
        assert_fingerprint(path, job["video_fingerprint"])
        print(json.dumps({"clip_committed": True, "reused_exact": item["imported_from_old"],
                          "worker_pid": os.getpid()}), flush=True)
        del rgb, array
    write_new(Path(job_file).with_suffix(".result.json"),
              {"status": "complete", "clips": len(job["rows"]), "pid": os.getpid(),
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})


def proc_rss(pid):
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        return int(next(line for line in lines if line.startswith("VmRSS:")).split()[1]) * 1024
    except (FileNotFoundError, StopIteration):
        return 0


def supervise(command, log, rss_limit, timeout_sec, on_started=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.monotonic()
    peak = 0
    with Path(log).open("xb") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        try:
            if on_started is not None:
                on_started(child.pid)
            while child.poll() is None:
                peak = max(peak, proc_rss(child.pid))
                if peak > rss_limit:
                    child.kill()
                    child.wait()
                    raise RuntimeError("own decode worker exceeded RSS cap; stopped without retry")
                if time.monotonic() - started > timeout_sec:
                    child.kill()
                    child.wait()
                    raise RuntimeError("own decode worker timed out; stopped without retry")
                time.sleep(.25)
            if child.returncode != 0:
                raise RuntimeError(f"own decode worker exit {child.returncode}; stopped without retry")
            return {"pid": child.pid, "exit_code": child.returncode,
                    "sampled_peak_rss_bytes": peak, "elapsed_sec": time.monotonic() - started}
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            write_new(Path(log).with_suffix(".worker_exit.json"),
                      {"pid": child.pid, "exit_code": child.returncode,
                       "sampled_peak_rss_bytes": peak,
                       "elapsed_sec": time.monotonic() - started})


def seal_split(protocol_dir, split, subset, output):
    inventory = []
    for row in subset:
        name = cache_name(row)
        item = json.loads((output / (name + ".entry.json")).read_text())
        if item["clip_id"] != row["clip_id"] or item["label_id"] != row["label_id"] \
                or item["file"] != name or sha(output / name) != item["sha256"]:
            raise ValueError("final durable cache inventory mismatch")
        inventory.append(item)
    if {path.name for path in output.glob("*.npy")} != {cache_name(row) for row in subset}:
        raise ValueError("final cache coverage mismatch")
    _, protocol = manifest_rows(protocol_dir, split)
    write_new(output / "private_index.json", inventory)
    seal = {"status": "complete", "split": split, "count": len(inventory),
            "manifest_sha256": protocol["manifest_sha256"][split], "preprocessing": PREPROCESS,
            "private_index_sha256": sha(output / "private_index.json"),
            "protocol_sha256": sha(Path(protocol_dir) / "protocol.json"),
            "repeated_frame_clips": sum(item["repeated_frames"] > 0 for item in inventory),
            "imported_old_exact_replays": sum(item["imported_from_old"] for item in inventory)}
    write_new(output / "cache_seal.json", seal)
    return seal


def execute(args, plan, subsets, protocol, old, jobs):
    output = args.output.resolve()
    # Conservative space allowance even if filesystem cannot hardlink the old
    # arrays. Creating a new output is the atomic, non-repeat launch guard.
    needed = sum(len(chunk) for _, chunk in jobs) * (16 * 224 * 224 * 3 + 65536)
    if shutil.disk_usage(output.parent).free < needed + (10 << 30):
        raise ValueError("insufficient disk space for full fresh cache and safety margin")
    output.mkdir(exist_ok=False)
    write_new(output / ".launch_once", {"pid": os.getpid(), "code_commit": plan["code_commit"]})
    write_new(output / "run_protocol.json", dict(plan, launcher_pid=os.getpid()))
    for name in ("jobs", "train_cache", "val_cache"):
        (output / name).mkdir(exist_ok=False)
    try:
        print(json.dumps({"stage": "verify_source_SHA", "cache_only": True}), flush=True)
        video_stats = {}
        paths = list(dict.fromkeys(row["video_path"] for _, chunk in jobs for row in chunk))
        for i, path in enumerate(paths):
            before = fingerprint(path)
            if sha(path) != protocol["video_sha256"][path]:
                raise ValueError("original video SHA changed; no decoding permitted")
            assert_fingerprint(path, before)
            video_stats[path] = before
            if (i + 1) % 25 == 0 or i + 1 == len(paths):
                print(json.dumps({"stage": "verify_source_SHA", "verified": i + 1,
                                  "total": len(paths)}), flush=True)
        old_hashes = {}
        for name, path in old.items():
            old_hashes[name] = {"path": str(path), "sha256": sha(path)}
        write_new(output / "private_old_inventory.json", old_hashes)
        done = {"train": 0, "val": 0}
        reused_count, peak = 0, 0
        started = time.monotonic()
        for number, (split, chunk) in enumerate(jobs, 1):
            job_file = output / "jobs" / f"{number:05d}.json"
            write_new(job_file, {"rows": chunk, "video_fingerprint": video_stats[chunk[0]["video_path"]],
                                "cache_output": str(output / f"{split}_cache"),
                                "reuse": {cache_name(row): old_hashes[cache_name(row)] for row in chunk
                                          if cache_name(row) in old_hashes}})
            def record(pid):
                write_new(job_file.with_suffix(".pid.json"),
                          {"pid": pid, "job": number, "split": split, "clips": len(chunk)})
            resource = supervise([sys.executable, "-m", MODULE, "--worker-job", str(job_file)],
                                 job_file.with_suffix(".log"), plan["worker_rss_limit_bytes"],
                                 args.timeout_sec, record)
            write_new(job_file.with_suffix(".exit.json"), resource)
            done[split] += len(chunk)
            reused_count += sum(cache_name(row) in old_hashes for row in chunk)
            peak = max(peak, resource["sampled_peak_rss_bytes"])
            print(json.dumps({"stage": "decode", "job": number, "jobs": len(jobs),
                              "committed": done, "reused_exact": reused_count,
                              "worker_pid": resource["pid"], "peak_worker_rss_bytes": peak,
                              "elapsed_sec": round(time.monotonic() - started, 1)}), flush=True)
        seals = {}
        if not args.smoke:
            for split in ("train", "val"):
                seals[split] = seal_split(args.failed_run / "protocol", split, subsets[split],
                                          output / f"{split}_cache")
        write_new(output / "result.json", {"status": "complete", "cache_only": True,
                  "training_started": False, "smoke_only": args.smoke, "counts": done,
                  "reused_exact": reused_count, "cache_seals": seals,
                  "sampled_peak_worker_rss_bytes": peak, "decode_elapsed_sec": time.monotonic() - started})
        write_new(output / "exit_code.json", {"exit_code": 0})
    except BaseException as exc:
        write_new(output / "failed.json", {"status": "failed", "automatic_retry": False,
                  "error_type": type(exc).__name__, "training_started": False})
        write_new(output / "exit_code.json", {"exit_code": 1})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--failed-run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--chunk-clips", type=int, default=16)
    parser.add_argument("--rss-gib", type=int, choices=range(2, 17), default=12)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.worker_job is not None:
        decode_job(args.worker_job)
        return
    if args.failed_run is None or args.output is None or not args.expected_commit \
            or args.timeout_sec < 1:
        parser.error("--failed-run, --output, --expected-commit and positive timeout required")
    plan, subsets, protocol, old, jobs = prepare_plan(args)
    print(json.dumps(plan, ensure_ascii=False), flush=True)
    if args.execute:
        execute(args, plan, subsets, protocol, old, jobs)


if __name__ == "__main__":
    main()
