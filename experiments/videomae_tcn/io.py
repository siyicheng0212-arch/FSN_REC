"""Private artifact writes and input/source checks shared by launcher workers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

from experiments.relation.data import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CODE_FILES = (
    "experiments/videomae_tcn/__init__.py", "experiments/videomae_tcn/evidence.py",
    "experiments/videomae_tcn/metrics.py", "experiments/videomae_tcn/io.py",
    "experiments/videomae_tcn/worker.py", "experiments/videomae_tcn/run.py",
    "experiments/relation/tcn_baseline.py", "experiments/relation/data.py",
    "experiments/relation/protocol.py", "experiments/relation/evaluate_suite.py",
    "experiments/relation/videomae2_followup.py", "experiments/aligned_protocol.py",
    "tests/test_videomae_tcn.py", "tests/test_relation_tcn_baseline.py")


def repository_state():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    return {"commit": commit, "clean": not dirty}


def source_hashes():
    return {name: sha256_file(ROOT / name) for name in CODE_FILES}


def write_text(path, text):
    path = Path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}")
    try:
        write_json(temporary, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_line(path, text):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def unchanged(plan):
    if repository_state() != {"commit": plan["source_commit"], "clean": True} or \
            source_hashes() != plan["source_sha256"]:
        raise RuntimeError("source changed after planning")
    index = Path(plan["index"])
    source = Path(plan["source_protocol_dir"])
    paths = {"index": index, "export_protocol": index.parent / "protocol.json",
             "source_audit": source / "audit.json"}
    paths.update({name: source / name for name in (
        "edges.jsonl", "chains_train.jsonl", "chains_val.jsonl", "manifest_metadata.jsonl")})
    if {name: sha256_file(path) for name, path in paths.items()} != plan["evidence"]["input_sha256"]:
        raise RuntimeError("sealed input changed after planning")
