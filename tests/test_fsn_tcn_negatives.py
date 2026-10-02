"""Train-only synthetic supervision and isolated val challenge invariants."""
import copy
import hashlib
import json
import math
import unittest

from experiments.fsn_tcn.hard_negatives import build_hard_negatives, validate_metadata
from experiments.fsn_tcn.perturb import build_challenge


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def record(prefix, *, split="train", collection="lishui", kind="clinical", size=4, masks=None):
    rows = {}
    for index in range(size):
        clip = f"{prefix}-{index}"
        rows[clip] = {
            "clip_id": clip, "split": split, "group_id": "group-" + prefix,
            "source_collection": collection, "source_kind": kind,
            "recording_key": sha("record-" + prefix), "source_video_key": sha("video-" + prefix),
            "timebase_key": sha("source_video_seconds"),
            "clip_start_sec": float(index * 2), "clip_end_sec": float(index * 2 + 1),
            "label_id": index % 7,
        }
    ids = list(rows)
    masks = [True] * (size - 1) if masks is None else masks
    chain = {
        "chain_id": "chain-" + prefix, "split": split, "source_kind": kind,
        "group_id": "group-" + prefix, "recording_key": sha("record-" + prefix),
        "source_video_key": sha("video-" + prefix), "timebase_key": sha("source_video_seconds"),
        "ordered_clip_ids": ids, "eligible": masks,
    }
    edges = [{
        "edge_id": f"edge-{prefix}-{index}", "split": split, "status": "C" if kind == "clinical" else "D",
        "source_kind": kind, "label_origin": "source_rule_v1",
        "left_clip_id": left, "right_clip_id": right,
    } for index, ((left, right), enabled) in enumerate(zip(zip(ids, ids[1:]), masks)) if enabled]
    return rows, chain, edges


def fixtures():
    rows, chains, edges = {}, [], []
    for kwargs in (dict(prefix="train-a"), dict(prefix="train-b"),
                   dict(prefix="train-other", collection="menzhen"),
                   dict(prefix="train-web", collection="youtube", kind="network"),
                   dict(prefix="val-a", split="val"), dict(prefix="val-b", split="val"),
                   dict(prefix="val-other", split="val", collection="menzhen"),
                   dict(prefix="val-web", split="val", collection="youtube", kind="network")):
        new_rows, chain, new_edges = record(**kwargs)
        rows.update(new_rows)
        chains.append(chain)
        edges.extend(new_edges)
    return rows, chains, edges


class HardNegativesTest(unittest.TestCase):
    def test_only_train_same_collection_distinct_all_three_identities(self):
        metadata, _, edges = fixtures()
        negatives, audit = build_hard_negatives(metadata, edges)
        self.assertTrue(negatives)
        for edge in negatives:
            left, right = (metadata[edge[key]] for key in ("left_clip_id", "right_clip_id"))
            self.assertEqual((left["split"], right["split"]), ("train", "train"))
            self.assertEqual((left["source_kind"], right["source_kind"]), ("clinical", "clinical"))
            self.assertEqual(left["source_collection"], right["source_collection"])
            for key in ("group_id", "recording_key", "source_video_key"):
                self.assertNotEqual(left[key], right[key])
            self.assertEqual(edge["status"], "D")
            self.assertEqual(edge["label_origin"], "synthetic_cross_record_v1")
            self.assertFalse(edge["eligible_for_formal_chain"])
        self.assertEqual(audit["skipped_requests"], 3)  # one unmatched menzhen record
        self.assertEqual(audit["generated_negatives"], 6)

    def test_deterministic_input_order_independent_without_action_selection(self):
        metadata, _, edges = fixtures()
        actual = build_hard_negatives(metadata, edges, seed=17, ratio=1.5)
        changed = copy.deepcopy(metadata)
        for row in changed.values():
            row.pop("label_id")
            row["unused_action_text"] = "unavailable"
        self.assertEqual(actual, build_hard_negatives(dict(reversed(list(changed.items()))),
                                                   list(reversed(edges)), seed=17, ratio=1.5))
        self.assertEqual(actual[1]["requested_negatives"], math.ceil(9 * 1.5))

    def test_no_mutation_unique_pairs_and_shortfall(self):
        metadata, _, edges = fixtures()
        before = copy.deepcopy((metadata, edges))
        negatives, audit = build_hard_negatives(metadata, edges, ratio=10)
        self.assertEqual(before, (metadata, edges))
        pairs = [(edge["left_clip_id"], edge["right_clip_id"]) for edge in negatives]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(len(negatives), 24)
        self.assertEqual(audit["generated_negatives"] + audit["skipped_requests"], audit["requested_negatives"])
        audit_text = json.dumps(audit)
        self.assertNotIn("train-a", audit_text)
        self.assertNotIn("group-", audit_text)
        self.assertNotIn("clip_id", audit_text)

    def test_no_hard_negative_or_no_positive_is_an_error(self):
        metadata, _, edges = record("only")
        with self.assertRaisesRegex(ValueError, "no hard negatives"):
            build_hard_negatives(metadata, edges)
        with self.assertRaisesRegex(ValueError, "no clinical train C"):
            build_hard_negatives(metadata, [])

    def test_each_identity_and_split_leakage_rejected(self):
        metadata, _, edges = fixtures()
        for key in ("group_id", "recording_key", "source_video_key"):
            changed = copy.deepcopy(metadata)
            changed["val-a-0"][key] = changed["train-a-0"][key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "leakage"):
                build_hard_negatives(changed, edges)
        changed = copy.deepcopy(metadata)
        changed["val-a-0"]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "test"):
            build_hard_negatives(changed, edges)

    def test_distinct_group_is_not_enough_to_mark_other_record(self):
        metadata, _, edges = fixtures()
        # Equal recording hash or equal source video hash must both disqualify.
        for field in ("recording_key", "source_video_key"):
            changed = {clip: row for clip, row in metadata.items() if clip.startswith("train-a") or clip.startswith("train-b")}
            changed = copy.deepcopy(changed)
            for clip, row in changed.items():
                if clip.startswith("train-b"):
                    row[field] = changed["train-a-0"][field]
            selected_edges = [edge for edge in edges if edge["left_clip_id"] in changed]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "no hard negatives"):
                build_hard_negatives(changed, selected_edges)

    def test_malformed_and_synthetic_input_rejected(self):
        metadata, _, edges = fixtures()
        for override in ({"source_kind": "unknown"}, {"clip_start_sec": float("nan")},
                         {"recording_key": "not-a-hash"}, {"clip_end_sec": 0},
                         {"clip_id": "different"}, {"duration": 8}):
            changed = copy.deepcopy(metadata)
            changed["train-a-0"].update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_metadata(changed)
        for ratio in (0, -1, True, float("nan")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                build_hard_negatives(metadata, edges, ratio=ratio)
        changed_edges = copy.deepcopy(edges)
        changed_edges[0]["label_origin"] = "synthetic_cross_record_v1"
        with self.assertRaisesRegex(ValueError, "synthetic"):
            build_hard_negatives(metadata, changed_edges)


class PerturbTest(unittest.TestCase):
    def val_fixture(self):
        metadata, chains, _ = fixtures()
        return metadata, [chain for chain in chains if chain["split"] == "val"]

    def assert_contract(self, result, metadata, expected_clips):
        actual = [clip for chain in result for clip in chain["ordered_clip_ids"]]
        self.assertCountEqual(actual, expected_clips)
        self.assertEqual(len(actual), len(set(actual)))
        for chain in result:
            self.assertIs(chain["synthetic_chain"], True)
            self.assertEqual(chain["split"], "val")
            self.assertEqual(len(chain["synthetic_breaks"]), len(chain["ordered_clip_ids"]) - 1)
            self.assertEqual(chain["eligible"], [True] * len(chain["synthetic_breaks"]))
            self.assertTrue(all(type(value) is bool for value in chain["synthetic_breaks"]))
            self.assertTrue(all(metadata[clip]["split"] == "val" for clip in chain["ordered_clip_ids"]))
            self.assertNotIn("recording_key", chain)

    def test_cross_record_has_known_breaks_but_unchanged_all_clip_count(self):
        metadata, chains = self.val_fixture()
        before = copy.deepcopy((metadata, chains))
        output, audit = build_challenge(chains, metadata, condition="clinical_cross_record", seed=42)
        self.assertEqual(before, (metadata, chains))
        self.assert_contract(output, metadata, [clip for chain in chains for clip in chain["ordered_clip_ids"]])
        self.assertEqual(audit["synthetic_breaks"], 2)
        for chain in output:
            for (left_id, right_id), broken in zip(zip(chain["ordered_clip_ids"], chain["ordered_clip_ids"][1:]), chain["synthetic_breaks"]):
                if broken:
                    left, right = metadata[left_id], metadata[right_id]
                    self.assertEqual(left["source_collection"], right["source_collection"])
                    self.assertEqual((left["source_kind"], right["source_kind"]), ("clinical", "clinical"))
                    for key in ("group_id", "recording_key", "source_video_key"):
                        self.assertNotEqual(left[key], right[key])
        self.assertEqual(audit["unmatched_clinical_runs"], 1)

    def test_shuffle_no_retained_original_clinical_neighbors_and_web_unchanged(self):
        metadata, chains = self.val_fixture()
        output, audit = build_challenge(chains, metadata, condition="clinical_shuffle", seed=42)
        originals = {(left, right) for chain in chains for left, right in zip(chain["ordered_clip_ids"], chain["ordered_clip_ids"][1:])}
        for chain in output:
            if chain["source_kind"] == "clinical":
                self.assertTrue(all(chain["synthetic_breaks"]))
                self.assertTrue(all(pair not in originals for pair in zip(chain["ordered_clip_ids"], chain["ordered_clip_ids"][1:])))
            else:
                self.assertEqual(chain["ordered_clip_ids"], [f"val-web-{i}" for i in range(4)])
                self.assertFalse(any(chain["synthetic_breaks"]))
        self.assertEqual(audit["synthetic_breaks"], 9)

    def test_original_structural_cuts_stay_separated(self):
        metadata, chains = self.val_fixture()
        chains[0]["eligible"] = [True, False, True]
        forbidden = (chains[0]["ordered_clip_ids"][1], chains[0]["ordered_clip_ids"][2])
        for condition in ("clinical_shuffle", "clinical_cross_record"):
            output, audit = build_challenge(chains, metadata, condition=condition, seed=42)
            for chain in output:
                self.assertNotIn(forbidden, list(zip(chain["ordered_clip_ids"], chain["ordered_clip_ids"][1:])))
            self.assertEqual(audit["original_structural_cuts_preserved_by_separation"], 1)
            self.assert_contract(output, metadata, [clip for chain in chains for clip in chain["ordered_clip_ids"]])

    def test_singleton_cross_record_concatenation_keeps_each_clip(self):
        rows_a, chain_a, _ = record("v-a", split="val", size=1)
        rows_b, chain_b, _ = record("v-b", split="val", size=1)
        output, audit = build_challenge([chain_a, chain_b], rows_a | rows_b,
                                       condition="clinical_cross_record", seed=42)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["synthetic_breaks"], [True])
        self.assertEqual(audit["clips"], 2)

    def test_label_independence_reproducibility_and_aggregate_audit(self):
        metadata, chains = self.val_fixture()
        for condition in ("clinical_cross_record", "clinical_shuffle"):
            first = build_challenge(chains, metadata, condition=condition, seed=23)
            changed = copy.deepcopy(metadata)
            for row in changed.values():
                row.pop("label_id")
            self.assertEqual(first, build_challenge(list(reversed(chains)), dict(reversed(list(changed.items()))),
                                                  condition=condition, seed=23))
            text = json.dumps(first[1])
            self.assertNotIn("val-a", text)
            self.assertNotIn("group-", text)

    def test_train_test_subset_duplicate_and_synthetic_chains_are_rejected(self):
        metadata, chains = self.val_fixture()
        for override in ({"split": "train"}, {"split": "test"}, {"synthetic_chain": True},
                         {"eligible": [1, True, True]}, {"recording_key": sha("different")}):
            changed = copy.deepcopy(chains)
            changed[0].update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                build_challenge(changed, metadata, condition="clinical_shuffle", seed=42)
        with self.assertRaisesRegex(ValueError, "cover every val"):
            build_challenge(chains[:-1], metadata, condition="clinical_shuffle", seed=42)
        changed = copy.deepcopy(chains)
        changed[0]["ordered_clip_ids"][1] = changed[0]["ordered_clip_ids"][0]
        with self.assertRaisesRegex(ValueError, "repeated"):
            build_challenge(changed, metadata, condition="clinical_shuffle", seed=42)

    def test_no_challenge_is_an_error_rather_than_fake_breaks(self):
        rows, chain, _ = record("single-record", split="val", size=1)
        for condition in ("clinical_shuffle", "clinical_cross_record"):
            with self.subTest(condition=condition), self.assertRaisesRegex(ValueError, "no synthetic"):
                build_challenge([chain], rows, condition=condition, seed=42)


if __name__ == "__main__":
    unittest.main()
