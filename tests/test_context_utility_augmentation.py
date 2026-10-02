"""Context-utility sampling invariants: labels do not determine any donors."""
import copy
import hashlib
import random
from types import SimpleNamespace

import pytest
import torch

from experiments.context_utility.augmentation import (
    apply_view, build_epoch_plan, plan_hash, validate_plan,
)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def record(prefix, *, split="train", collection="lishui", kind="clinical", size=5, masks=None):
    metadata = {}
    for i in range(size):
        clip = f"{prefix}-{i}"
        metadata[clip] = {
            "clip_id": clip, "split": split, "group_id": "group-" + prefix,
            "recording_key": sha("record-" + prefix), "source_video_key": sha("video-" + prefix),
            "timebase_key": sha("time-" + prefix), "source_collection": collection, "source_kind": kind,
            "clip_start_sec": float(i * 2), "clip_end_sec": float(i * 2 + 1), "label_id": i % 7,
        }
    ids = list(metadata)
    first = metadata[ids[0]]
    chain = {key: first[key] for key in (
        "split", "group_id", "recording_key", "source_video_key", "timebase_key", "source_kind")}
    chain.update(chain_id="chain-" + prefix, ordered_clip_ids=ids,
                 eligible=[True] * (size - 1) if masks is None else masks)
    return metadata, chain


def fixture():
    metadata, chains = {}, {"train": [], "val": []}
    for options in (
        dict(prefix="a", masks=[True, False, True, True]), dict(prefix="b"),
        dict(prefix="only", collection="menzhen"), dict(prefix="singleton", size=1),
        dict(prefix="web", kind="network", collection="FSN"),
        dict(prefix="val", split="val"),
    ):
        rows, chain = record(**options)
        metadata.update(rows)
        chains[chain["split"]].append(chain)
    return SimpleNamespace(metadata=metadata, chains=chains)


def test_exact_train_coverage_and_structural_cuts():
    bundle = fixture()
    original = copy.deepcopy(bundle)
    plan, audit = build_epoch_plan(bundle, seed=42, epoch=0)
    ids = [clip for entry in plan for clip in entry["ordered_clip_ids"]]
    assert set(ids) == {clip for clip, row in bundle.metadata.items() if row["split"] == "train"}
    assert len(ids) == len(set(ids)) == audit["original_train_clip_coverage"]
    assert ["a-0", "a-1"] in [entry["ordered_clip_ids"] for entry in plan]
    assert ["a-2", "a-3", "a-4"] in [entry["ordered_clip_ids"] for entry in plan]
    assert audit["segments"] == 6
    assert audit["singleton_segments"] == 1
    assert audit["actual_replacements"] > 0
    assert audit["unfilled_replacements"] > 0
    assert audit["plan_sha256"] == plan_hash(plan)
    assert bundle.metadata == original.metadata and bundle.chains == original.chains
    assert validate_plan(bundle, plan, max_neighbors=2)


def test_donor_contract_and_clean_web_singletons():
    bundle = fixture()
    plan, _ = build_epoch_plan(bundle, seed=17, epoch=3, max_neighbors=1)
    for entry in plan:
        ids = entry["ordered_clip_ids"]
        if len(ids) == 1 or bundle.metadata[ids[0]]["source_kind"] == "network":
            assert entry["replacements"] == []
        for replacement in entry["replacements"]:
            assert replacement["position"] not in entry["anchor_positions"]
            assert min(abs(replacement["position"] - a) for a in entry["anchor_positions"]) <= 1
            donor = bundle.metadata[replacement["donor_clip_id"]]
            for anchor_pos in entry["anchor_positions"]:
                anchor = bundle.metadata[ids[anchor_pos]]
                assert donor["split"] == anchor["split"] == "train"
                assert donor["source_kind"] == anchor["source_kind"] == "clinical"
                assert donor["source_collection"] == anchor["source_collection"]
                assert all(donor[key] != anchor[key] for key in (
                    "group_id", "recording_key", "source_video_key"))


def test_label_blind_input_order_independent_and_model_rng_independent():
    bundle = fixture()
    result = build_epoch_plan(bundle, seed=12, epoch=7)
    changed = copy.deepcopy(bundle)
    for row in changed.metadata.values():
        row.pop("label_id")
        row["unused_label_string"] = "different"
    changed.metadata = dict(reversed(list(changed.metadata.items())))
    changed.chains["train"].reverse()
    random.seed(923)
    random.random()
    torch.manual_seed(901)
    torch.rand(98)
    assert result == build_epoch_plan(changed, seed=12, epoch=7)
    assert result[1]["action_labels_used"] is False
    assert result[1]["train_only"] is True
    assert plan_hash(result[0]) != plan_hash(build_epoch_plan(bundle, seed=12, epoch=8)[0])


def test_multiple_anchors_are_never_replaced_and_zero_plan_is_explicit():
    bundle = fixture()
    plan, audit = build_epoch_plan(bundle, seed=42, epoch=0, anchors_per_segment=2)
    for entry in plan:
        assert len(entry["anchor_positions"]) == min(2, len(entry["ordered_clip_ids"]))
        assert not set(entry["anchor_positions"]) & {r["position"] for r in entry["replacements"]}
    rows, chain = record("alone")
    small = SimpleNamespace(metadata=rows, chains={"train": [chain]})
    clean, audit = build_epoch_plan(small, seed=42, epoch=0)
    assert audit["actual_replacements"] == 0
    assert audit["has_constructed_replacements"] is False
    assert audit["unfilled_replacements"] > 0
    assert not clean[0]["replacements"]


@pytest.mark.parametrize("kwargs", [dict(seed=True), dict(epoch=-1),
                                  dict(anchors_per_segment=0), dict(max_neighbors=0)])
def test_bad_sampling_parameters(kwargs):
    options = dict(seed=42, epoch=0)
    options.update(kwargs)
    with pytest.raises(ValueError):
        build_epoch_plan(fixture(), **options)


def replacement_entry(plan):
    return next(entry for entry in plan if entry["replacements"])


@pytest.mark.parametrize("donor", ["val-0", "web-0", "only-0", "missing"])
def test_invalid_donors_rejected(donor):
    bundle = fixture()
    plan, _ = build_epoch_plan(bundle, seed=42, epoch=0)
    replacement_entry(plan)["replacements"][0]["donor_clip_id"] = donor
    with pytest.raises(ValueError, match="donor"):
        validate_plan(bundle, plan)


def test_same_record_and_anchor_replacements_rejected():
    bundle = fixture()
    plan, _ = build_epoch_plan(bundle, seed=42, epoch=0)
    entry = replacement_entry(plan)
    entry["replacements"][0]["donor_clip_id"] = entry["ordered_clip_ids"][0]
    with pytest.raises(ValueError, match="donor"):
        validate_plan(bundle, plan)
    plan, _ = build_epoch_plan(bundle, seed=42, epoch=0)
    entry = replacement_entry(plan)
    entry["replacements"][0]["position"] = entry["anchor_positions"][0]
    with pytest.raises(ValueError, match="anchor"):
        validate_plan(bundle, plan)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "reorder", "val"])
def test_missing_duplicate_structural_bridge_and_cross_split_rejected(mode):
    bundle = fixture()
    plan, _ = build_epoch_plan(bundle, seed=42, epoch=0)
    if mode == "missing":
        plan.pop()
    elif mode == "duplicate":
        plan.append(copy.deepcopy(plan[0]))
    elif mode == "reorder":
        entry = next(entry for entry in plan if len(entry["ordered_clip_ids"]) > 1)
        entry["ordered_clip_ids"].reverse()
    else:
        plan[0]["ordered_clip_ids"][0] = "val-0"
    with pytest.raises(ValueError):
        validate_plan(bundle, plan)


def test_identity_leakage_and_malformed_metadata_rejected():
    bundle = fixture()
    for key in ("group_id", "recording_key", "source_video_key"):
        changed = copy.deepcopy(bundle)
        changed.metadata["val-0"][key] = changed.metadata["a-0"][key]
        with pytest.raises(ValueError, match="leakage"):
            build_epoch_plan(changed, seed=42, epoch=0)
    changed = copy.deepcopy(bundle)
    changed.metadata["val-0"]["split"] = "test"
    with pytest.raises(ValueError, match="test"):
        build_epoch_plan(changed, seed=42, epoch=0)


def test_apply_view_paired_donor_replacement_and_anchor_preservation():
    bundle = fixture()
    plan, _ = build_epoch_plan(bundle, seed=42, epoch=0)
    entry = replacement_entry(plan)
    n = len(entry["ordered_clip_ids"])
    features = torch.arange(n * 4, dtype=torch.float32).reshape(n, 4)
    logits = torch.arange(n * 7, dtype=torch.float32).reshape(n, 7)
    before_f, before_l = features.clone(), logits.clone()
    reads = []
    def read_clip(clip):
        reads.append(clip)
        return torch.full((4,), 100.0), torch.full((7,), 200.0)
    output_f, output_l, anchors = apply_view(features, logits, entry, read_clip)
    assert reads == [row["donor_clip_id"] for row in entry["replacements"]]
    assert torch.equal(features, before_f) and torch.equal(logits, before_l)
    assert output_f.data_ptr() != features.data_ptr()
    assert output_l.data_ptr() != logits.data_ptr()
    for pos in anchors:
        assert torch.equal(output_f[pos], features[pos])
        assert torch.equal(output_l[pos], logits[pos])
    for row in entry["replacements"]:
        assert torch.equal(output_f[row["position"]], torch.full((4,), 100.0))
        assert torch.equal(output_l[row["position"]], torch.full((7,), 200.0))


@pytest.mark.parametrize("pair", [
    (torch.ones(5), torch.ones(7)), (torch.ones(4), torch.ones(6)),
    (torch.ones(4, dtype=torch.long), torch.ones(7)),
    (torch.full((4,), float("nan")), torch.ones(7)),
    (torch.ones(4), torch.full((7,), float("inf"))),
    (torch.full((4,), 1e100, dtype=torch.float64), torch.ones(7)),
    (torch.ones(4),),
])
def test_misaligned_or_nonfinite_donor_pair_rejected(pair):
    bundle = fixture()
    plan, _ = build_epoch_plan(bundle, seed=42, epoch=0)
    entry = replacement_entry(plan)
    n = len(entry["ordered_clip_ids"])
    with pytest.raises(ValueError):
        apply_view(torch.ones(n, 4), torch.ones(n, 7), entry, lambda _: pair)


def test_clean_view_uses_same_anchors_and_does_not_read_donors():
    plan, _ = build_epoch_plan(fixture(), seed=42, epoch=0)
    entry = copy.deepcopy(replacement_entry(plan))
    entry["replacements"] = []
    n = len(entry["ordered_clip_ids"])
    features, logits = torch.ones(n, 4), torch.ones(n, 7)
    def never_read(_):
        raise AssertionError("clean view must not read donor")
    output_f, output_l, anchors = apply_view(features, logits, entry, never_read)
    assert torch.equal(output_f, features) and torch.equal(output_l, logits)
    assert anchors == entry["anchor_positions"]
