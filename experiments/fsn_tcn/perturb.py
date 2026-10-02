"""Deterministic, label-free validation challenges kept apart from real chains."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
import random

from .hard_negatives import _different_record, _same_record, _seed, _text, validate_metadata


CONDITIONS = ("clinical_cross_record", "clinical_shuffle")


def _chain_id(condition, clips):
    digest = hashlib.sha256(json.dumps([condition, clips], separators=(",", ":")).encode()).hexdigest()
    return "challenge-" + digest[:24]


def _runs(chain):
    clips, masks = chain["ordered_clip_ids"], chain["eligible"]
    start = 0
    for index, enabled in enumerate(masks):
        if not enabled:
            yield clips[start:index + 1]
            start = index + 1
    yield clips[start:]


def _validate_chains(chains, metadata):
    if not isinstance(chains, list) or not chains:
        raise ValueError("chains must be a nonempty list of original val chains")
    copied, seen, identifiers, original_pairs, original_cuts = [], set(), set(), set(), set()
    for raw in chains:
        if not isinstance(raw, dict):
            raise ValueError("chain rows must be objects")
        chain = copy.deepcopy(raw)
        if chain.get("split") != "val" or chain.get("synthetic_chain", False):
            raise ValueError("challenges require original val chains only, never train/test or synthetic input")
        identifier = _text(chain, "chain_id")
        if identifier in identifiers:
            raise ValueError("duplicate chain_id")
        identifiers.add(identifier)
        clips, masks = chain.get("ordered_clip_ids"), chain.get("eligible")
        if not isinstance(clips, list) or not clips or any(not isinstance(clip, str) for clip in clips):
            raise ValueError("ordered_clip_ids must be a nonempty string list")
        if not isinstance(masks, list) or len(masks) != len(clips) - 1 or any(type(mask) is not bool for mask in masks):
            raise ValueError("eligible must be an exact-length boolean list")
        for clip in clips:
            if clip in seen or clip not in metadata or metadata[clip]["split"] != "val":
                raise ValueError("unknown, repeated or non-val clip in challenge chains")
            seen.add(clip)
        first = metadata[clips[0]]
        for clip in clips:
            row = metadata[clip]
            if not _same_record(first, row):
                raise ValueError("original chain crosses a declared record, group, source video or timebase")
            for field in ("source_kind", "group_id", "recording_key", "source_video_key", "timebase_key"):
                if field in chain and chain[field] != row.get(field):
                    raise ValueError(f"original chain {field} differs from clip metadata")
        for (left_id, right_id), enabled in zip(zip(clips, clips[1:]), masks):
            left, right = metadata[left_id], metadata[right_id]
            if right["clip_start_sec"] < left["clip_start_sec"]:
                raise ValueError("original chains must retain chronological order")
            if enabled:
                if right["clip_start_sec"] < left["clip_end_sec"]:
                    raise ValueError("eligible original edge is overlapping")
                original_pairs.add((left_id, right_id))
            else:
                original_cuts.add((left_id, right_id))
        copied.append(chain)
    expected = {clip for clip, row in metadata.items() if row["split"] == "val"}
    if seen != expected:
        raise ValueError("challenge chains must cover every val clip exactly once")
    return copied, original_pairs, original_cuts


def _make_chain(clips, condition, metadata, original_pairs, original_cuts):
    synthetic_breaks = [(left, right) not in original_pairs for left, right in zip(clips, clips[1:])]
    if any((left, right) in original_cuts for left, right in zip(clips, clips[1:])):
        raise ValueError("challenge attempted to bridge an original structural cut")
    first = metadata[clips[0]]
    collections = {metadata[clip]["source_collection"] for clip in clips}
    kinds = {metadata[clip]["source_kind"] for clip in clips}
    if len(collections) != 1 or len(kinds) != 1:
        raise ValueError("challenge source type or collection mismatch")
    # Deliberately no source_video_key/recording_key/group_id at chain level:
    # synthetic chains can include multiple records and must not masquerade as
    # a legal original chain. Individual metadata remains the authoritative ID.
    return {
        "chain_id": _chain_id(condition, clips), "split": "val", "source_kind": first["source_kind"],
        "source_collection": first["source_collection"], "ordered_clip_ids": list(clips),
        "eligible": [True] * (len(clips) - 1), "synthetic_chain": True,
        "synthetic_origin": "fsn_tcn_" + condition + "_v1", "synthetic_breaks": synthetic_breaks,
        "candidate_reasons": ["synthetic_disconnection" if cut else "retained_original_candidate"
                              for cut in synthetic_breaks],
    }


def _shuffle_run(clips, rng, original_pairs):
    """Find a permutation containing no original forward adjacent pair.

    Retrying is bounded. Reverse order is a deterministic guaranteed fallback
    for a chronological, unique-ID run: all resulting neighbors are reversed.
    """
    if len(clips) <= 1:
        return list(clips)
    for _ in range(64):
        shuffled = list(clips)
        rng.shuffle(shuffled)
        if all(pair not in original_pairs for pair in zip(shuffled, shuffled[1:])):
            return shuffled
    return list(reversed(clips))


def build_challenge(chains, metadata, *, condition, seed):
    """Return synthetic-only val chains with each original clip exactly once.

    Structural cuts are first split into separate runs and never bridged.
    Cross-record challenges exchange run tails between distinct records within
    the same clinical source collection; singleton runs are concatenated. Runs
    lacking a compatible partner stay unchanged and are counted in the audit.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    checked = validate_metadata(metadata)
    originals, original_pairs, original_cuts = _validate_chains(chains, checked)
    rng = random.Random(_seed(seed))
    clinical, untouched = defaultdict(list), []
    for chain in sorted(originals, key=lambda row: row["chain_id"]):
        for run in _runs(chain):
            first = checked[run[0]]
            if first["source_kind"] == "clinical":
                clinical[first["source_collection"]].append(list(run))
            else:
                untouched.append(list(run))
    outputs, exchanged_pairs, unmatched, shuffled_runs = [], 0, 0, 0
    if condition == "clinical_shuffle":
        for collection in sorted(clinical):
            for run in clinical[collection]:
                outputs.append(_shuffle_run(run, rng, original_pairs))
                shuffled_runs += len(run) > 1
    else:
        for collection in sorted(clinical):
            pending = list(clinical[collection])
            rng.shuffle(pending)
            while pending:
                left = pending.pop(0)
                compatible = next((index for index, right in enumerate(pending)
                                   if _different_record(checked[left[0]], checked[right[0]])), None)
                if compatible is None:
                    outputs.append(left)
                    unmatched += 1
                    continue
                right = pending.pop(compatible)
                if min(len(left), len(right)) < 2:
                    outputs.append(left + right)
                else:
                    left_cut, right_cut = len(left) // 2, len(right) // 2
                    outputs.extend((left[:left_cut] + right[right_cut:], right[:right_cut] + left[left_cut:]))
                exchanged_pairs += 1
    outputs.extend(untouched)
    result = [_make_chain(run, condition, checked, original_pairs, original_cuts) for run in outputs]
    before = Counter(clip for chain in originals for clip in chain["ordered_clip_ids"])
    after = Counter(clip for chain in result for clip in chain["ordered_clip_ids"])
    if before != after or any(count != 1 for count in after.values()):
        raise AssertionError("challenge construction did not preserve clips exactly once")
    changed = sum(sum(chain["synthetic_breaks"]) for chain in result)
    if changed == 0:
        raise ValueError("no synthetic disconnections could be constructed for this challenge")
    audit = {
        "condition": condition, "seed": seed, "split": "val", "synthetic_only": True,
        "input_chains": len(originals), "output_chains": len(result), "clips": sum(before.values()),
        "clinical_clips": sum(row["source_kind"] == "clinical" and row["split"] == "val" for row in checked.values()),
        "retained_original_edges": sum(not cut for chain in result for cut in chain["synthetic_breaks"]),
        "synthetic_breaks": changed, "original_structural_cuts_preserved_by_separation": len(original_cuts),
        "cross_record_run_pairs": exchanged_pairs, "unmatched_clinical_runs": unmatched,
        "shuffled_clinical_runs": shuffled_runs, "clips_exactly_once": True,
        "actions_used_to_construct_challenge": False, "test_read": False,
        "limitation": "Constructed breaks test synthetic robustness, not detection of human-verified real editing boundaries.",
    }
    return result, audit
