"""Synthetic cache recovery tests: no private data, videos or model weights."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.hospital_holdout import cache_recovery as recovery
from experiments.hospital_holdout.prepare import sha


def row(name="synthetic-a", video="synthetic-video", label=2):
    return {"clip_id": name, "video_path": video, "label_id": label,
            "clip_start_sec": .1, "clip_end_sec": .3}


def arrays():
    rgb = np.zeros((16, 224, 224, 3), dtype=np.uint8)
    indices = np.zeros(16, dtype=np.int64)
    return indices, np.array([[.1, .2], [.2, .3]]), rgb, rgb.copy()


def test_new_file_no_overwrite(tmp_path):
    path = tmp_path / "atomic.json"
    recovery.write_new(path, {"n": 1})
    assert json.loads(path.read_text()) == {"n": 1}
    with pytest.raises(FileExistsError):
        recovery.write_new(path, {"n": 2})
    assert json.loads(path.read_text()) == {"n": 1}


@pytest.mark.parametrize("count", [1, 4, 16])
def test_chunks_bound_one_video_and_preserve_order(count):
    items = [row(f"synthetic-{i}", f"video-{i % 2}") for i in range(37)]
    chunks = recovery.chunk_rows(items, count)
    assert sum(map(len, chunks)) == 37
    assert all(1 <= len(chunk) <= count for chunk in chunks)
    assert all(len({r["video_path"] for r in chunk}) == 1 for chunk in chunks)
    for video in {r["video_path"] for r in items}:
        assert [r for chunk in chunks for r in chunk if r["video_path"] == video] == \
               [r for r in items if r["video_path"] == video]


@pytest.mark.parametrize("count", [0, 17, -1])
def test_invalid_chunk_rejected(count):
    with pytest.raises(ValueError):
        recovery.chunk_rows([row()], count)


@pytest.mark.parametrize("array", [np.zeros((16, 224, 224, 3), np.float32),
                                   np.zeros((15, 224, 224, 3), np.uint8)])
def test_invalid_geometry_rejected(array):
    with pytest.raises(ValueError):
        recovery.check_array(array)


def test_fingerprint_detects_source_mutation(tmp_path):
    path = tmp_path / "video"
    path.write_bytes(b"synthetic")
    fingerprint = recovery.fingerprint(path)
    recovery.assert_fingerprint(path, fingerprint)
    path.write_bytes(b"changed bytes")
    with pytest.raises(ValueError):
        recovery.assert_fingerprint(path, fingerprint)


def test_durable_new_clip_preserves_pts_label_and_dtype(tmp_path):
    item = recovery.save_decoded(row(), *arrays(), tmp_path)
    assert item["label_id"] == 2 and item["frame_indices"] == [0] * 16
    assert item["actual_frame_pts_sec"] == [[.1, .2]] * 16
    assert item["repeated_frames"] == 15
    assert item["imported_from_old"] is False
    assert np.load(tmp_path / item["file"]).dtype == np.uint8
    assert json.loads((tmp_path / (item["file"] + ".entry.json")).read_text()) == item
    with pytest.raises(FileExistsError):
        recovery.save_decoded(row(), *arrays(), tmp_path)


def test_exact_old_replay_hardlinks_unchanged_old(tmp_path):
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    old = old_dir / recovery.cache_name(row())
    np.save(old, arrays()[-1], allow_pickle=False)
    before = sha(old)
    item = recovery.save_decoded(row(), *arrays(), new_dir, old, before)
    assert item["old_pixel_replay_exact"] and item["sha256"] == before == sha(old)
    assert old.stat().st_ino == (new_dir / old.name).stat().st_ino


def test_old_replay_changed_pixels_stop_before_import(tmp_path):
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    old = old_dir / recovery.cache_name(row())
    np.save(old, arrays()[-1], allow_pickle=False)
    indices, pts, rgb, different = arrays()
    different[0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="pixels differ"):
        recovery.save_decoded(row(), indices, pts, rgb, different, new_dir, old, sha(old))
    assert not list(new_dir.iterdir())


def test_old_cache_sha_mutation_rejected(tmp_path):
    old = tmp_path / "old.npy"
    np.save(old, arrays()[-1], allow_pickle=False)
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    with pytest.raises(ValueError, match="changed"):
        recovery.save_decoded(row(), *arrays(), new_dir, old, "0" * 64)
    assert not list(new_dir.iterdir())


def test_inventory_unknown_or_symlink_rejected(tmp_path):
    np.save(tmp_path / "unknown.npy", arrays()[-1], allow_pickle=False)
    with pytest.raises(ValueError):
        recovery.inspect_old(tmp_path, [row()])


def test_inventory_shape_checks_without_faking_metadata(tmp_path):
    path = tmp_path / recovery.cache_name(row())
    np.save(path, arrays()[-1], allow_pickle=False)
    assert recovery.inspect_old(tmp_path, [row()]) == {path.name: path}
    assert not (tmp_path / "private_index.json").exists()


def test_worker_success_cpu_only_exit_record(tmp_path):
    log = tmp_path / "job.log"
    result = recovery.supervise([sys.executable, "-c",
        "import os; assert os.environ['CUDA_VISIBLE_DEVICES']==''; print('ok')"],
        log, 12 << 30, 10)
    assert result["exit_code"] == 0 and b"ok" in log.read_bytes()
    assert json.loads(log.with_suffix(".worker_exit.json").read_text())["exit_code"] == 0


def test_worker_failure_not_retried_and_exit_preserved(tmp_path):
    log = tmp_path / "job.log"
    with pytest.raises(RuntimeError, match="exit 3"):
        recovery.supervise([sys.executable, "-c", "raise SystemExit(3)"], log, 12 << 30, 10)
    assert json.loads(log.with_suffix(".worker_exit.json").read_text())["exit_code"] == 3


def test_own_worker_timeout_and_rss_cap(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="timed out"):
        recovery.supervise([sys.executable, "-c", "import time; time.sleep(10)"],
                           tmp_path / "timeout.log", 12 << 30, .01)
    monkeypatch.setattr(recovery, "proc_rss", lambda pid: 1000)
    with pytest.raises(RuntimeError, match="RSS cap"):
        recovery.supervise([sys.executable, "-c", "import time; time.sleep(10)"],
                           tmp_path / "memory.log", 999, 10)
    assert json.loads((tmp_path / "memory.worker_exit.json").read_text())["exit_code"] != 0


def test_full_seal_exact_coverage_and_original_contract(tmp_path, monkeypatch):
    item = recovery.save_decoded(row(), *arrays(), tmp_path)
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    (protocol / "protocol.json").write_text("synthetic")
    monkeypatch.setattr(recovery, "manifest_rows", lambda directory, split:
                        ([row()], {"manifest_sha256": {"train": "synthetic-sha"}}))
    seal = recovery.seal_split(protocol, "train", [row()], tmp_path)
    assert seal["status"] == "complete" and seal["count"] == 1
    assert seal["preprocessing"] == recovery.PREPROCESS
    assert seal["private_index_sha256"] == sha(tmp_path / "private_index.json")
    assert json.loads((tmp_path / "private_index.json").read_text())[0] == item


def test_partial_cannot_seal(tmp_path):
    with pytest.raises(FileNotFoundError):
        recovery.seal_split(tmp_path, "train", [row()], tmp_path)
    assert not (tmp_path / "cache_seal.json").exists()


def plan_fixture(tmp_path, monkeypatch):
    failed = tmp_path / "failed"
    failed.mkdir()
    protocol_dir = failed / "protocol"
    protocol_dir.mkdir()
    protocol_file = protocol_dir / "protocol.json"
    protocol_file.write_text("synthetic protocol")
    monkeypatch.setattr(recovery, "PROTOCOL_SHA", sha(protocol_file))
    (failed / "run_protocol.json").write_text(json.dumps({
        "code_commit": recovery.ORIGINAL_COMMIT, "preprocessing": recovery.PREPROCESS}))
    (failed / "status.json").write_text(json.dumps({"status": "failed", "training_completed": False}))
    (failed / "train_cache").mkdir()
    np.save(failed / "train_cache" / recovery.cache_name(row()), arrays()[-1])
    (failed / "val_cache").mkdir()
    train = [row(), row("missing")]
    val = [row("hospital-development")]
    monkeypatch.setattr(recovery, "manifest_rows", lambda directory, split:
                        (train if split == "train" else val,
                         {"manifest_sha256": {"train": "x", "val": "y"}}))
    monkeypatch.setattr(recovery.subprocess, "check_output", lambda command, **kwargs:
                        "synthetic-commit\n" if "rev-parse" in command else "")
    return SimpleNamespace(failed_run=failed, output=tmp_path / "new", expected_commit="synthetic-commit",
                           execute=False, smoke=False, chunk_clips=16, rss_gib=12, timeout_sec=900)


def test_default_plan_no_create_and_new_first(tmp_path, monkeypatch):
    args = plan_fixture(tmp_path, monkeypatch)
    plan, _, _, _, jobs = recovery.prepare_plan(args)
    assert not args.output.exists() and plan["read_only"] and plan["cache_only"]
    assert plan["existing_train_arrays"] == 1 and plan["missing_train_arrays"] == 1
    assert jobs[0][1][0]["clip_id"] == "missing"
    assert jobs[-1][1][0]["clip_id"] == "synthetic-a"


@pytest.mark.parametrize("reason", ["commit", "output", "not_failed", "protocol"])
def test_plan_mismatch_fail_closed(tmp_path, monkeypatch, reason):
    args = plan_fixture(tmp_path, monkeypatch)
    if reason == "commit":
        args.expected_commit = "wrong"
    elif reason == "output":
        args.output.mkdir()
    elif reason == "not_failed":
        (args.failed_run / "status.json").write_text(json.dumps({"status": "running"}))
    else:
        (args.failed_run / "protocol/protocol.json").write_text("modified")
    with pytest.raises(ValueError):
        recovery.prepare_plan(args)


def test_real_decode_worker_bounded_reader_and_sidecars(tmp_path, monkeypatch):
    source = tmp_path / "synthetic-video"
    source.write_bytes(b"synthetic")
    output = tmp_path / "cache"
    output.mkdir()
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"rows": [row(video=str(source))], "reuse": {},
                              "cache_output": str(output),
                              "video_fingerprint": recovery.fingerprint(source)}))
    created = []
    class FakeReader:
        def __init__(self, path, num_threads, ctx):
            created.append((path, num_threads, ctx))
        def __len__(self):
            return 2
        def get_frame_timestamp(self, indices):
            return arrays()[1]
        def get_batch(self, indices):
            return SimpleNamespace(asnumpy=lambda: arrays()[2])
    monkeypatch.setitem(sys.modules, "decord", SimpleNamespace(VideoReader=FakeReader, cpu=lambda n: n))
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(setNumThreads=lambda n: None))
    monkeypatch.setattr(recovery, "spatial_rgb", lambda rgb: rgb.copy())
    recovery.decode_job(job)
    assert len(created) == 1 and created[0][1:] == (1, 0)
    assert len(list(output.glob("*.entry.json"))) == 1
    assert json.loads(job.with_suffix(".result.json").read_text())["clips"] == 1
