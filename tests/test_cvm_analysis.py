"""Statistical/diagnostic tests with hand-checkable synthetic outcomes."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cvm.analysis import (AMBIGUOUS_CLASSES, CLINICAL_GROUPS, compare_predictions,
                          derive_visual_groups, main, metric_report,
                          summarize_seed_differences, validate_predictions)


def metadata(mode="flat", groups=CLINICAL_GROUPS):
    heads = {"flat": ["flat"], "hierarchy": ["group", "conditional"],
             "aux_flat": ["flat", "group", "conditional"]}[mode]
    return {"mode": mode, "trained_heads": heads, "taxonomy": {"name": "test", "groups": [list(group) for group in groups]},
            "protocol_sha256": "a" * 64, "split": "val", "selection_exposure": "development_validation",
            "eval_metadata": {"sampling": "uniform", "num_frames": 36}}


def flat_row(index, target, prediction=None, group=None, source="privatePatientAlice", scores=None):
    prediction = target if prediction is None else prediction
    values = [-4.0] * 7 if scores is None else list(scores)
    if scores is None:
        values[prediction] = 4.0
    return {"clip_id": f"private-clip-{index}", "group_id": f"private-record-{index // 2 if group is None else group}",
            "source": source, "duration": 2.0, "target": target,
            "flat_logits": values, "repeated_frame_fraction": 0.5}


def hierarchy_row(index, target, route=3, fine=None, groups=CLINICAL_GROUPS):
    row = flat_row(index, target)
    row["flat_logits"] = None
    row["group_logits"] = [-4.0] * len(groups)
    row["group_logits"][route] = 4.0
    row["conditional_logits"] = []
    for group in groups:
        scores = [0.0] if len(group) == 1 else [-2.0] * len(group)
        if len(group) > 1:
            best = target if target in group else group[0]
            if fine is not None and fine in group:
                best = fine
            scores[group.index(best)] = 2.0
        row["conditional_logits"].append(scores)
    return row


class MetricTests(unittest.TestCase):
    def test_confusion_metrics_exact_and_absent_classes_fixed_zero(self):
        rows = [flat_row(0, 3, 3), flat_row(1, 4, 3), flat_row(2, 3, 4), flat_row(3, 4, 4)]
        report = metric_report(rows, metadata=metadata())
        metrics = report["methods"]["main"]
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["micro_f1"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0 / 7)
        self.assertAlmostEqual(metrics["macro_precision"], 1.0 / 7)
        self.assertAlmostEqual(metrics["macro_recall"], 1.0 / 7)
        self.assertEqual(metrics["micro_precision"], 0.5)
        self.assertEqual(metrics["micro_recall"], 0.5)
        self.assertEqual(metrics["weighted_f1"], 0.5)
        self.assertEqual(metrics["confusion_matrix"][3][4], 1)
        self.assertEqual(metrics["confusion_matrix"][4][3], 1)

    def test_five_primary_preserves_singleton_to_five_false_positives(self):
        rows = [flat_row(0, 1, 1), flat_row(1, 0, 1), flat_row(2, 6, 1)]
        report = metric_report(rows, metadata=metadata())
        primary = report["ambiguous_five"]["methods"]["main"]
        restricted = report["ambiguous_five"]["ground_truth_restricted_sensitivity"]["main"]
        self.assertAlmostEqual(primary["per_class"][0]["precision"], 1.0 / 3)
        self.assertAlmostEqual(primary["per_class"][0]["f1"], 0.5)
        self.assertEqual(restricted["per_class"][0]["f1"], 1.0)
        self.assertAlmostEqual(primary["macro_f1"], 0.1)
        self.assertEqual(primary["confusion_matrix"][0][1], 1)

    def test_five_to_singleton_false_negative_and_renormalized_diagnostic(self):
        rows = [flat_row(0, 1, scores=[8, 7, -1, -1, -1, -1, -1]), flat_row(1, 1, 1), flat_row(2, 6, 6)]
        report = metric_report(rows, metadata=metadata())
        primary = report["ambiguous_five"]["methods"]["main"]
        diagnostic = report["ambiguous_five"]["subset_renormalized_main_diagnostic"]
        self.assertEqual(primary["per_class"][0]["recall"], 0.5)
        self.assertEqual(primary["confusion_matrix"][1][0], 1)
        self.assertEqual(diagnostic["per_class"][0]["recall"], 1.0)

    def test_untrained_heads_are_never_reported(self):
        rows = [flat_row(0, 0), flat_row(1, 6), flat_row(2, 3)]
        for row in rows:
            row["group_logits"] = [0.0] * 4
            row["conditional_logits"] = [[0], [0], [0, 0, 0], [0, 0]]
        report = metric_report(rows, metadata=metadata())
        self.assertIsNone(report["routing"]["actual_router"])
        self.assertEqual(report["routing"]["actual_router_unavailable_reason"], "group_head_not_trained")
        self.assertIsNotNone(report["routing"]["flat_mapped_group"])
        self.assertNotIn("hard", report["methods"])

    def test_routing_conditionals_error_decomposition_and_group_f1(self):
        rows = [hierarchy_row(0, 4, route=2), hierarchy_row(1, 4, route=3, fine=3),
                hierarchy_row(2, 4, route=3, fine=4), hierarchy_row(3, 0, route=0)]
        report = metric_report(rows, metadata=metadata("hierarchy"))
        router = report["routing"]["actual_router"]
        self.assertEqual(router["accuracy"], 0.75)
        self.assertAlmostEqual(router["conditional_fine_recall_given_correct_route"], 2.0 / 3)
        self.assertEqual(router["per_class_conditional_recall"][4]["route_correct_support"], 2)
        self.assertEqual(router["per_class_conditional_recall"][4]["recall"], 0.5)
        self.assertEqual(len(router["per_class"]), 4)
        self.assertIn("macro_f1", router)
        self.assertAlmostEqual(router["macro_precision"], 0.5)
        self.assertAlmostEqual(router["macro_recall"], 5.0 / 12)
        self.assertAlmostEqual(router["ground_truth_ambiguous_five"]["route_accuracy"], 2.0 / 3)
        self.assertEqual(router["ground_truth_ambiguous_five"]["conditional_fine_recall_given_correct_route"], 0.5)
        errors = report["hard_error_decomposition"]["all"]
        self.assertEqual(errors["routing_errors"], 1)
        self.assertEqual(errors["within_group_errors_given_correct_route"], 1)
        self.assertEqual(errors["correct"], 2)
        self.assertIsNone(report["routing"]["flat_mapped_group"])
        self.assertNotIn("oracle_flat", report["methods"])
        self.assertIn("oracle_hierarchy", report["methods"])

    def test_flat_mapped_and_probability_aggregated_groups_differ(self):
        row = flat_row(0, 4)
        row.pop("flat_logits")
        row["flat_probs"] = [0.3, 0.10, 0.10, 0.22, 0.23, 0.03, 0.02]
        report = metric_report([row, dict(row, clip_id="another", group_id="another-group")], metadata=metadata())
        routing = report["routing"]
        self.assertEqual(routing["flat_mapped_group"]["accuracy"], 0)
        self.assertEqual(routing["flat_aggregated_probability_group"]["accuracy"], 1)

    def test_same_checkpoint_hard_soft_difference_and_singleton_contribution(self):
        row = hierarchy_row(0, 0, route=2)
        row["group_logits"] = [1.8, -8.0, 2.0, -8.0]
        row["conditional_logits"] = [[0], [0], [0, 0, 0], [0, 0]]
        report = metric_report([row, hierarchy_row(2, 6, route=1)], metadata=metadata("hierarchy"))
        self.assertEqual(report["same_checkpoint_hard_soft"]["corrected"], 1)
        self.assertEqual(report["same_checkpoint_hard_soft"]["changed"], 1)
        self.assertEqual(report["singleton_contribution"]["num_samples"], 2)
        self.assertGreater(report["singleton_contribution"]["contribution_to_seven_class_macro_f1"], 0)

    def test_private_source_aliases_and_small_slice_suppression(self):
        raw_source = "/private/patientAlice/video.mp4"
        rows = [flat_row(0, 0, source=raw_source), flat_row(1, 3, source=raw_source), flat_row(2, 6, source="other")]
        report = metric_report(rows, metadata=metadata())
        serialized = json.dumps(report)
        for private in ("patientAlice", raw_source, "private-clip", "private-record"):
            self.assertNotIn(private, serialized)
        alias = "source:source-" + hashlib.sha256(raw_source.encode()).hexdigest()[:12]
        self.assertTrue(report["slices"]["results"][alias]["suppressed"])
        self.assertNotIn("methods", report["slices"]["results"][alias])
        self.assertEqual(report["slices"]["repeated_frame_metadata_coverage"], 3)


class ValidationTests(unittest.TestCase):
    def test_invalid_vectors_probabilities_and_metadata_fail(self):
        cases = []
        for value in ([1.0] * 6, [float("nan")] + [0.0] * 6, [float("inf")] + [0.0] * 6):
            row = flat_row(0, 0)
            row["flat_logits"] = value
            cases.append([row])
        for probabilities in ([0.1] * 7, [-0.1, 1.1, 0, 0, 0, 0, 0], [True, 0, 0, 0, 0, 0, 0]):
            row = flat_row(0, 0)
            row["leaf_probs"] = probabilities
            cases.append([row])
        cases.append([flat_row(0, 0), flat_row(0, 1)])
        row = flat_row(0, 0)
        row["target"] = 1.0
        cases.append([row])
        row = flat_row(0, 0)
        row["duration"] = -1
        cases.append([row])
        for rows in cases:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                metric_report(rows, metadata=metadata())
        with self.assertRaises(ValueError):
            metric_report([flat_row(0, 0)], metadata={})

    def test_conditional_group_probability_normalization(self):
        row = hierarchy_row(0, 4)
        row.pop("conditional_logits")
        row["conditional_probs"] = [[1], [1], [.3, .3, .3], [.5, .5]]
        with self.assertRaises(ValueError):
            metric_report([row], metadata=metadata("hierarchy"))

    def test_prediction_label_alias_and_taxonomy_mismatches(self):
        row = flat_row(0, 0)
        row["label"] = 1
        with self.assertRaises(ValueError):
            metric_report([row], metadata=metadata())
        row = flat_row(0, 0)
        row["prediction"] = 1
        with self.assertRaises(ValueError):
            metric_report([row], metadata=metadata())
        with self.assertRaises(ValueError):
            metric_report([flat_row(0, 0)], metadata=metadata(), groups=((0,), (1, 2, 3, 4, 5, 6)))


class PairedTests(unittest.TestCase):
    def setUp(self):
        self.baseline = [flat_row(i, i % 7, (i + 1) % 7) for i in range(14)]
        self.candidate = [flat_row(i, i % 7) for i in range(14)]

    def compare(self, left=None, right=None, **kwargs):
        return compare_predictions(self.baseline if left is None else left, self.candidate if right is None else right,
                                   baseline_metadata=kwargs.pop("baseline_metadata", metadata()),
                                   candidate_metadata=kwargs.pop("candidate_metadata", metadata()),
                                   bootstrap_replicates=100, **kwargs)

    def test_paired_group_bootstrap_reproducible_order_matching_and_fixed_classes(self):
        first = self.compare()
        second = self.compare(right=list(reversed(self.candidate)))
        self.assertEqual(first["paired_uncertainty"], second["paired_uncertainty"])
        uncertainty = first["paired_uncertainty"]
        self.assertEqual(uncertainty["num_recording_groups"], 7)
        self.assertEqual(uncertainty["num_clips"], 14)
        self.assertEqual(uncertainty["metrics"]["accuracy"]["paired_delta"], 1.0)
        self.assertEqual(uncertainty["metrics"]["five_macro_f1"]["paired_delta"], 1.0)
        self.assertGreater(uncertainty["replicates_missing_at_least_one_class"], 0)
        self.assertIsNone(uncertainty["p_values"])
        self.assertEqual(first["paired_prediction_changes"]["corrected"], 14)

    def test_identical_checkpoints_have_zero_paired_interval(self):
        report = self.compare(right=self.baseline)
        for metrics in report["paired_uncertainty"]["metrics"].values():
            self.assertEqual(metrics["paired_delta"], 0)
            self.assertEqual(metrics["percentile_95_interval"], [0.0, 0.0])

    def test_strict_pairing_rejects_intersection_labels_groups_treatment(self):
        invalid = [self.candidate[:-1]]
        for key, value in (("target", 6), ("group_id", "changed"), ("source", "changed"), ("duration", 5), ("repeated_frame_fraction", 0)):
            changed = copy.deepcopy(self.candidate)
            changed[0][key] = value
            invalid.append(changed)
        for rows in invalid:
            with self.assertRaises(ValueError):
                self.compare(right=rows)
        for key, value in (("protocol_sha256", "b" * 64), ("split", "test"), ("eval_metadata", {"sampling": "other"})):
            altered = metadata()
            altered[key] = value
            with self.assertRaises(ValueError):
                self.compare(candidate_metadata=altered)

    def test_bootstrap_refuses_single_group_and_untrained_methods(self):
        for row in self.baseline + self.candidate:
            row["group_id"] = "one-recording"
        with self.assertRaises(ValueError):
            self.compare()
        with self.assertRaises(ValueError):
            self.compare(baseline_method="hard")

    def test_fair_flat_oracle_recomputed_with_candidate_taxonomy(self):
        changed_groups = ((0,), (6,), (1, 3, 4), (2, 5))
        baseline = [flat_row(0, 1, scores=[-4, 5, -4, -4, -4, 8, -4]), flat_row(2, 1, scores=[-4, 5, -4, -4, -4, 8, -4])]
        candidate = [hierarchy_row(0, 1, route=2, groups=changed_groups), hierarchy_row(2, 1, route=2, groups=changed_groups)]
        result = self.compare(left=baseline, right=candidate, candidate_metadata=metadata("hierarchy", changed_groups))
        oracle = result["fair_oracle_comparison"]
        self.assertTrue(oracle["baseline_grouping_recomputed"])
        self.assertEqual(oracle["shared_truth_group_taxonomy"], [list(group) for group in changed_groups])
        self.assertEqual(result["baseline"]["methods"]["oracle_flat"]["accuracy"], 0)
        self.assertEqual(oracle["baseline_trained_flat"]["accuracy"], 1)
        self.assertEqual(oracle["candidate_trained_hierarchy"]["accuracy"], 1)
        paired_oracle = self.compare(left=baseline, right=candidate, candidate_metadata=metadata("hierarchy", changed_groups),
                                    baseline_method="oracle_flat", candidate_method="oracle_hierarchy")
        self.assertEqual(paired_oracle["paired_uncertainty"]["metrics"]["accuracy"]["paired_delta"], 0)
        self.assertEqual(paired_oracle["paired_prediction_changes"]["corrected"], 0)
        self.assertIn("oracle_shared_truth_group_taxonomy", paired_oracle["paired_uncertainty"])

    def test_seed_summary_separate_and_duplicate_seeds_rejected(self):
        report = summarize_seed_differences([{"seed": 1, "baseline": .7, "candidate": .8}, {"seed": 2, "baseline": .8, "candidate": .7}])
        self.assertEqual(report["num_training_seeds"], 2)
        self.assertAlmostEqual(report["paired_delta_mean"], 0)
        self.assertAlmostEqual(report["paired_delta_sample_sd"], 2 ** .5 * .1)
        with self.assertRaises(ValueError):
            summarize_seed_differences([{"seed": 1, "baseline": .7, "candidate": .8}] * 2)


class VisualGroupingAndCLITests(unittest.TestCase):
    def test_visual_grouping_rejects_eval_confusion_and_is_deterministic(self):
        confusion = [[int(a == b) * 10 for b in range(7)] for a in range(7)]
        confusion[1][2] = confusion[2][1] = confusion[1][5] = confusion[5][1] = 8
        provenance = {"split": "train", "out_of_fold": True, "held_out_evaluation_used": False}
        self.assertEqual(derive_visual_groups(confusion, provenance=provenance), derive_visual_groups(confusion, provenance=provenance))
        for key, value in (("split", "val"), ("split", "test"), ("out_of_fold", False), ("held_out_evaluation_used", True)):
            with self.assertRaises(ValueError):
                derive_visual_groups(confusion, provenance={**provenance, key: value})

    def test_cli_writes_only_safe_aggregate_and_protects_private_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            predictions = path / "val_private.jsonl"
            sidecar = path / "prediction_metadata.json"
            output = path / "safe_report.json"
            rows = [flat_row(i, i % 7) for i in range(14)]
            predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
            sidecar.write_text(json.dumps(metadata()))
            self.assertEqual(main(["aggregate", "--predictions", str(predictions), "--output", str(output)]), 0)
            serialized = output.read_text()
            self.assertNotIn("privatePatientAlice", serialized)
            self.assertNotIn("private-clip", serialized)
            with self.assertRaises(SystemExit):
                main(["aggregate", "--predictions", str(predictions), "--output", str(sidecar)])
            self.assertEqual(json.loads(sidecar.read_text()), metadata())


if __name__ == "__main__":
    unittest.main()
