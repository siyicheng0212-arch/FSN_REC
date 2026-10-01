import copy
import json
import unittest

import numpy as np
import torch
from torch import nn

from experiments.relation.evaluate_suite import classification_metrics, evaluate_suite


class MemoryIndex:
    def __init__(self, rows, features):
        self.clips = {row["clip_id"]: row for row in rows}
        self.features = features

    def read(self, clip_id):
        return copy.deepcopy(self.features[clip_id])


class FixedRelation(nn.Module):
    def __init__(self, logit):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(float(logit)))

    def forward(self, left, right):
        return self.logit.expand(1)


def fixture():
    definitions = [("private_clinical_first", 3, "clinical", "clinical_record", 5.),
                   ("private_clinical_second", 4, "clinical", "clinical_record", 5.),
                   ("private_network_first", 3, "network", "network_record", .5),
                   ("private_network_second", 3, "network", "network_record", 2.)]
    definitions += [(f"private_single_{i}", i, "network", f"single_{i}", 1.) for i in (0, 1, 2, 5, 6)]
    metadata, features = [], {}
    for clip, label, source, group, duration in definitions:
        metadata.append({"clip_id": clip, "label_id": label, "source_kind": source,
                         "group_id": group, "split": "val", "duration": duration})
        logits = np.full(7, -10., dtype=np.float32)
        logits[label] = 10.
        if "second" in clip:
            logits[3], logits[4] = 1., .9
        features[clip] = {"logits": logits, "global_tokens": np.zeros((2, 3), dtype=np.float32),
                          "local_tokens": np.zeros((3, 4), dtype=np.float32),
                          "global_positions": np.array([0., 1.], dtype=np.float32),
                          "local_positions": np.array([0., .5, 1.], dtype=np.float32)}
    chains = [{"chain_id": "private_clinical_chain", "ordered_clip_ids": [definitions[0][0], definitions[1][0]], "eligible": [True]},
              {"chain_id": "private_network_chain", "ordered_clip_ids": [definitions[2][0], definitions[3][0]], "eligible": [True]}]
    chains += [{"chain_id": f"private_chain_{row[0]}", "ordered_clip_ids": [row[0]], "eligible": []}
               for row in definitions[4:]]
    edges = [{"edge_id": "clinical_edge", "left_clip_id": definitions[0][0], "right_clip_id": definitions[1][0],
              "split": "val", "status": "C"},
             {"edge_id": "network_edge", "left_clip_id": definitions[2][0], "right_clip_id": definitions[3][0],
              "split": "val", "status": "D"}]
    transition = np.full((7, 7), .001)
    transition[:, 4] = .994
    return MemoryIndex(metadata, features), metadata, chains, edges, transition


class RelationEvaluateSuiteTest(unittest.TestCase):
    def test_paired_source_rule_protects_network_and_fixes_clinical(self):
        index, metadata, chains, edges, transition = fixture()
        summary, private = evaluate_suite(index, metadata, chains, edges, transition,
                                          {"learned_mlp": FixedRelation(-20), "learned_dual": FixedRelation(20)})
        variants = summary["variants"]
        self.assertEqual(variants["source_rule"]["paired_vs_A"]["improved"], 1)
        self.assertEqual(variants["source_rule"]["paired_vs_A"]["worsened"], 0)
        self.assertEqual(variants["all_candidate"]["paired_vs_A"]["worsened"], 1)
        self.assertEqual(variants["all_candidate"]["network_candidate_gate"]["activation_fraction"], 1.)
        self.assertEqual(variants["source_rule"]["network_candidate_gate"]["activation_fraction"], 0.)
        self.assertEqual(variants["learned_mlp"]["paired_vs_A"]["changed_count"], 0)
        self.assertEqual(variants["learned_dual"]["confusion"], variants["all_candidate"]["confusion"])
        self.assertEqual(variants["learned_dual"]["source_policy_edge_metrics"]["fp"], 1)
        self.assertEqual(variants["A_only"]["sweep_reperfusion"]["reperfusion_to_sweep"]["count"], 1)
        self.assertEqual(variants["source_rule"]["sweep_reperfusion"]["reperfusion_to_sweep"]["count"], 0)
        self.assertEqual(len(private), 9)

    def test_action_labels_are_not_inference_inputs(self):
        index, metadata, chains, edges, transition = fixture()
        _, before = evaluate_suite(index, metadata, chains, edges, transition,
                                   {"learned_mlp": FixedRelation(20)})
        for row in metadata:
            row["label_id"] = (row["label_id"] + 1) % 7
        _, after = evaluate_suite(index, metadata, chains, edges, transition,
                                  {"learned_mlp": FixedRelation(20)})
        self.assertEqual([row["predictions"] for row in before], [row["predictions"] for row in after])

    def test_cut_and_zero_strength_reproduce_same_A_for_all_variants(self):
        index, metadata, chains, edges, transition = fixture()
        summary, private = evaluate_suite(index, metadata, chains, edges, transition,
                                          {"learned_mlp": FixedRelation(20)}, strength=0)
        self.assertTrue(all(len(set(row["predictions"].values())) == 1 for row in private))
        for chain in chains:
            chain["eligible"] = [False] * (len(chain["ordered_clip_ids"]) - 1)
        summary, private = evaluate_suite(index, metadata, chains, [], transition,
                                          {"learned_mlp": FixedRelation(20)})
        self.assertTrue(all(len(set(row["predictions"].values())) == 1 for row in private))
        self.assertEqual(summary["variants"]["source_rule"]["network_candidate_gate"]["count"], 0)

    def test_singleton_and_candidate_edge_seal_consistency(self):
        index, metadata, chains, edges, transition = fixture()
        row = metadata[-1]
        single = MemoryIndex([row], {row["clip_id"]: index.read(row["clip_id"])})
        summary, private = evaluate_suite(single, [row], [chains[-1]], [], transition,
                                          {"learned_mlp": FixedRelation(20)})
        self.assertEqual(len(private), 1)
        self.assertEqual(summary["variants"]["learned_mlp"]["network_candidate_gate"]["count"], 0)
        for change in ("origin", "status", "missing", "source_video", "timebase", "recording"):
            index, metadata, chains, edges, transition = fixture()
            if change == "origin":
                edges[0]["label_origin"] = "manual_supplied"
            elif change == "status":
                edges[0]["status"] = "D"
            elif change == "missing":
                edges.pop()
            else:
                key = {"source_video": "source_video_key", "timebase": "timebase_key", "recording": "recording_key"}[change]
                metadata[0][key], metadata[1][key] = "left_record", "right_record"
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate_suite(index, metadata, chains, edges, transition)

    def test_rejects_leakage_incomplete_coverage_and_metadata_mismatch(self):
        for change in ("missing", "duplicate", "train", "group", "label"):
            index, metadata, chains, edges, transition = fixture()
            if change == "missing":
                chains.pop()
            elif change == "duplicate":
                chains.append({"chain_id": "other", "ordered_clip_ids": [metadata[0]["clip_id"]]})
            elif change == "train":
                metadata[0]["split"] = "train"
            elif change == "group":
                metadata[1]["group_id"] = "different_record"
            else:
                metadata = copy.deepcopy(metadata)
                metadata[0]["label_id"] = 2
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate_suite(index, metadata, chains, edges, transition)

    def test_safe_aggregate_contains_no_identifiers_and_empty_slices_are_finite(self):
        index, metadata, chains, edges, transition = fixture()
        summary, _ = evaluate_suite(index, metadata, chains, edges, transition)
        serialized = json.dumps(summary, allow_nan=False)
        self.assertNotIn("private_", serialized)
        self.assertNotIn("group_id", serialized)
        self.assertNotIn("clip_id", serialized)
        empty = classification_metrics([], [])
        self.assertEqual(empty["macro_f1"], 0.)
        self.assertEqual([row["support"] for row in empty["per_class"]], [0] * 7)
        self.assertTrue(np.isfinite(empty["accuracy"]))


if __name__ == "__main__":
    unittest.main()
