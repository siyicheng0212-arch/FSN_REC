"""Synthetic-only CPU tests; no private manifests, recordings or weights."""
import json

import pytest

from experiments.hospital_temporal import prepare as temporal
from experiments.relation.source_edges import build_source_relations


def row(clip, start=0, end=1, **kwargs):
    return {"clip_id": clip, "split": "train", "group_id": "synthetic-group",
            "record_id": "synthetic-record", "video_path": "synthetic-video",
            "source_collection": "lishui", "label_id": 0,
            "clip_start_sec": start, "clip_end_sec": end,
            "time_unit_verified": True, **kwargs}


def test_candidates_match_old_masks_and_keep_all_clips():
    records = [row("a", 0, 10), row("b", 1, 2), row("c", 3, 5),
               row("d", 10, 11), row("e", 16, 17), row("f", 23, 24)]
    _, chains, segments, excluded = temporal.build_chains(records)
    old = build_source_relations(records, {"lishui": "clinical"})
    assert chains[0]["eligible"] == old["chains"]["train"][0]["eligible"]
    assert chains[0]["eligible"] == [False, False, True, True, False]
    assert segments["train"] == [["a"], ["b"], ["c", "d", "e"], ["f"]]
    assert excluded[("train", "overlap_or_nested_interval")] == 2
    assert excluded[("train", "gap_exceeds_cutoff")] == 1
    assert "status" not in chains[0]  # No C/D target.


def test_duplicate_start_cuts_without_dropping_or_bridging():
    _, chains, segments, _ = temporal.build_chains([row("a"), row("b"), row("c", 1, 2)])
    assert chains[0]["eligible"] == [False, False]
    assert segments["train"] == [["a"], ["b"], ["c"]]


@pytest.mark.parametrize("field", ["record_id", "group_id", "video_path", "timebase_id"])
def test_different_structural_identity_cannot_join(field):
    _, chains, segments, _ = temporal.build_chains([
        row("a"), row("b", 1, 2, **{field: "different-synthetic-identity"})])
    assert len(chains) == 2
    assert sorted(segments["train"]) == [["a"], ["b"]]


@pytest.mark.parametrize("source,kind", [("lishui", "clinical"), ("FSN", "network")])
def test_sources_and_actions_do_not_gate_connections(source, kind):
    records = [row("a", source_collection=source, label_id=6),
               row("b", 1, 2, source_collection=source, label_id=2)]
    _, chains, segments, _ = temporal.build_chains(records)
    old = build_source_relations(records, {source: kind})
    assert chains[0]["eligible"] == old["chains"]["train"][0]["eligible"] == [True]
    assert segments["train"] == [["a", "b"]]


@pytest.mark.parametrize("field", ["clip_id", "record_id", "group_id", "video_path"])
def test_cross_split_leakage_rejected(field):
    a = row("a")
    b = row("b", split="val", source_collection="menzhen", group_id="dev-group",
            record_id="dev-record", video_path="dev-video")
    b[field] = a[field]
    with pytest.raises(ValueError):
        temporal.build_chains([a, b])


@pytest.mark.parametrize("gap", [True, -1, 6, float("nan"), "5"])
def test_cutoff_cannot_be_silently_changed(gap):
    with pytest.raises(ValueError):
        temporal.build_chains([row("a")], gap)


def test_unverified_units_rejected():
    with pytest.raises(ValueError):
        temporal.build_chains([row("a", time_unit_verified=False)])


def test_safe_aggregate_counts_no_clip_identifiers():
    records = [row("private-synthetic-a"), row("private-synthetic-b", 1, 2),
               row("private-synthetic-c", 10, 11),
               row("private-synthetic-dev", split="val", source_collection="menzhen",
                   record_id="dev-record", group_id="dev-group", video_path="dev-video")]
    records, chains, segments, cuts = temporal.build_chains(records)
    summary = temporal.aggregate(records, chains, segments, cuts)
    assert summary["train"]["clips"] == 3
    assert summary["train"]["candidate_edges"] == 1
    assert summary["train"]["neighbor_clips"] == 2
    assert summary["train"]["isolated_clips"] == 1
    assert summary["val"]["isolated_clips"] == 1
    text = json.dumps(summary)
    assert not any(value in text for value in ["private-synthetic", "dev-record", "dev-video", "clip_id"])


def test_read_only_audit_and_original_rows_unchanged(monkeypatch, tmp_path):
    records = [row("train"), row("dev", split="val", source_collection="menzhen",
                             record_id="dev-record", group_id="dev-group", video_path="dev-video")]
    before = json.dumps(records, sort_keys=True)
    protocol = {"video_byte_overlap": 0, "R_D_enabled": False, "manifest_sha256": {}}
    monkeypatch.setattr(temporal, "manifest_rows", lambda directory, split:
                        ([r for r in records if r["split"] == split], protocol))
    monkeypatch.setattr(temporal, "validate_partition", lambda *args: None)
    monkeypatch.setattr(temporal, "sha", lambda path: "0" * 64)
    summary, private = temporal.audit(tmp_path)
    assert summary["temporal_training_started"] is False
    assert summary["requires_new_A_trained_without_Menzhen"] is True
    assert private["contains_private_clip_identifiers"] is True
    assert not list(tmp_path.iterdir())
    assert json.dumps(records, sort_keys=True) == before


def test_existing_output_is_not_overwritten(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(ValueError):
        temporal.write_audit(target, tmp_path / "input", {}, {})
    assert not list(target.iterdir())
