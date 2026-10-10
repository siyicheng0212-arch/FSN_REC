"""Synthetic CPU contracts; these are not real-feature CUDA or training results."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.anchor_reference.data import anchor_batch, anchor_view, sampling_plan, train_replacements
from experiments.anchor_reference.model import AnchorReferenceTCN
from experiments.input_evidence import run_suite, worker
from experiments.input_evidence.model import AUGMENTED, MASKED, VARIANTS, InputEvidenceTCN, input_policy
from experiments.relation.tcn_baseline import TCNConfig, collate_segments
from experiments.relation.data import sha256_file
from experiments.paper_validation import perturb
from tests.test_anchor_reference import fixtures, inputs


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, variant="visual", dropout=0):
        return InputEvidenceTCN(TCNConfig(5, 7, 8, 3, dropout), variant)

    def test_all_four_have_identical_initial_weights_and_parameter_counts(self):
        first = self.model("logits")
        for variant in VARIANTS:
            m = self.model(variant)
            self.assertEqual(m.parameter_counts()["parameters"], first.parameter_counts()["parameters"])
            self.assertEqual(m.parameter_counts()["reference_parameters"], 0)
            for name, weight in first.state_dict().items():
                self.assertTrue(torch.equal(weight, m.state_dict()[name]), (variant, name))

    def test_constructor_and_mask_do_not_consume_rng(self):
        torch.manual_seed(22)
        before = torch.get_rng_state().clone()
        m = self.model()
        f, a, valid = inputs()
        m.effective_inputs(f, a)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_logits_cell_exactly_reuses_previous_plain_model(self):
        f, a, valid = inputs()
        for variant in ("logits", "logits_aug"):
            m = self.model(variant).eval()
            old = AnchorReferenceTCN(m.config, "plain").eval()
            torch.testing.assert_close(m(f, a, valid), old(f, a, valid), atol=0, rtol=0)

    def test_masked_cells_match_same_weight_plain_on_zeroed_slots(self):
        f, a, valid = inputs()
        for variant in MASKED:
            m = self.model(variant).eval()
            old = AnchorReferenceTCN(m.config, "plain").eval()
            zeroed = torch.cat((f[..., :-7], torch.zeros_like(a)), -1)
            torch.testing.assert_close(m(f, a, valid), old(zeroed, torch.zeros_like(a), valid), atol=0, rtol=0)

    def test_raw_score_changes_never_change_masked_outputs(self):
        f, a, valid = inputs()
        other = f.clone()
        other[..., -7:] = other[..., -7:] * 20 + 7
        for variant in MASKED:
            m = self.model(variant).eval()
            torch.testing.assert_close(m(f, a, valid), m(other, other[..., -7:], valid), atol=0, rtol=0)

    def test_logits_control_is_sensitive_to_seven_input_scores(self):
        f, a, valid = inputs()
        other = f.clone()
        other[..., -7:] += 3
        m = self.model("logits").eval()
        self.assertGreater(float((m(f, a, valid) - m(other, other[..., -7:], valid)).abs().max()), 0)

    def test_visual_input_still_affects_masked_model(self):
        f, a, valid = inputs()
        other = f.clone()
        other[..., :3] += 3
        m = self.model().eval()
        self.assertGreater(float((m(f, a, valid) - m(other, a, valid)).abs().max()), 0)

    def test_masking_does_not_mutate_shared_evidence(self):
        f, a, valid = inputs()
        before_f, before_a = f.clone(), a.clone()
        self.model()(f, a, valid)
        self.assertTrue(torch.equal(before_f, f))
        self.assertTrue(torch.equal(before_a, a))

    def test_mask_works_in_training_and_dropout_rng_is_unchanged(self):
        f, a, valid = inputs()
        other = f.clone()
        other[..., -7:] += 100
        m = self.model(dropout=.1).train()
        torch.manual_seed(123)
        first = m(f, a, valid)
        after = torch.get_rng_state().clone()
        torch.manual_seed(123)
        second = m(other, other[..., -7:], valid)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertTrue(torch.equal(after, torch.get_rng_state()))

    def test_three_cpu_updates_have_finite_gradients_and_no_mask_leak(self):
        f, a, valid = inputs((7, 7))
        targets = torch.arange(14).reshape(2, 7) % 7
        for variant in VARIANTS:
            m = self.model(variant)
            optimizer = torch.optim.AdamW(m.parameters(), lr=.001)
            before = m.base.classifier.weight.detach().clone()
            for _ in range(3):
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(m(f, a, valid)[valid], targets[valid])
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
                self.assertGreater(m.base.classifier.weight.grad.abs().max().item(), 0)
                if variant in MASKED:
                    self.assertEqual(m.base.stem[0].weight.grad[:, -7:].abs().max().item(), 0)
                optimizer.step()
            self.assertFalse(torch.equal(before, m.base.classifier.weight))
            self.assertIsNone(f.grad)
            self.assertIsNone(a.grad)

    def test_A_score_autograd_has_no_masked_path(self):
        f, _, valid = inputs()
        f.requires_grad_()
        a = f[..., -7:]
        self.model()(f, a, valid).sum().backward()
        self.assertEqual(f.grad[..., -7:].abs().max().item(), 0)
        self.assertGreater(f.grad[..., :-7].abs().max().item(), 0)

    def test_singleton_uses_native_classifier_not_A_fallback(self):
        f, a, valid = inputs((1,))
        for variant in VARIANTS:
            m = self.model(variant).eval()
            z = m(f, a, valid)
            torch.testing.assert_close(z, worker.native_oracle(m, (f, a, None, valid, None), "cpu"))
            self.assertFalse(torch.equal(z, a))

    def test_zero_neighbor_audit_equals_same_weight_single_clip(self):
        f, a, valid = inputs((7, 3))
        for variant in VARIANTS:
            m = self.model(variant).eval()
            z = m(f, a, valid, mode="zero")
            for b, t in zip(*torch.where(valid)):
                single = m(f[b:b + 1, t:t + 1], a[b:b + 1, t:t + 1], torch.ones(1, 1, dtype=torch.bool))
                torch.testing.assert_close(z[b, t], single[0, 0], atol=2e-6, rtol=2e-6)

    def test_padding_cannot_modify_valid_predictions(self):
        f, a, valid = inputs((7, 2))
        other = f.clone()
        other[~valid] *= 100
        for variant in VARIANTS:
            m = self.model(variant).eval()
            torch.testing.assert_close(m(f, a, valid)[valid], m(other, other[..., -7:], valid)[valid], atol=1e-6, rtol=1e-6)

    def test_corrupt_pair_rejected_even_when_masked(self):
        f, a, valid = inputs()
        with self.assertRaises(ValueError):
            self.model()(f, a + 1, valid)

    def test_nonfinite_raw_scores_cannot_be_hidden_by_mask(self):
        f, a, valid = inputs()
        f[0, 0, -1] = float("nan")
        with self.assertRaises(ValueError):
            self.model()(f, f[..., -7:], valid)

    def test_invalid_variant_and_overrides_rejected(self):
        with self.assertRaises(ValueError):
            self.model("reference")
        f, a, valid = inputs()
        with self.assertRaises(ValueError):
            self.model()(f, a, valid, overrides={})

    def test_strict_checkpoint_missing_weight_rejected(self):
        state = dict(self.model().state_dict())
        state.pop("base.classifier.bias")
        with self.assertRaises(RuntimeError):
            self.model().load_state_dict(state, strict=True)


class DataAndMetricTests(unittest.TestCase):
    def test_shared_replacement_schedule_keeps_anchor_and_pairs(self):
        segments, metadata = fixtures()
        schedule = train_replacements(segments, metadata)
        for row in schedule["rows"]:
            view = anchor_view(segments, row, True)
            j = row["anchor"]
            np.testing.assert_array_equal(view.features[j], segments[row["segment"]].features[j])
            np.testing.assert_array_equal(view.a_logits[j], segments[row["segment"]].a_logits[j])
            self.assertEqual(sum(view.labels >= 0), 1)
            for position, donor_segment, donor_index in row["swaps"]:
                np.testing.assert_array_equal(view.features[position], segments[donor_segment].features[donor_index])
                np.testing.assert_array_equal(view.a_logits[position], segments[donor_segment].a_logits[donor_index])

    def test_no_augmentation_has_same_number_of_anchor_views_and_loss_targets(self):
        segments, metadata = fixtures()
        schedule = train_replacements(segments, metadata)
        normal, normal_positions = anchor_batch(segments, schedule, range(4), False)
        augmented, augmented_positions = anchor_batch(segments, schedule, range(4), True)
        self.assertEqual(normal_positions, augmented_positions)
        self.assertTrue(torch.equal(normal[2], augmented[2]))
        self.assertTrue(torch.equal(normal[3], augmented[3]))

    def test_masked_augmented_donors_have_no_score_path(self):
        segments, metadata = fixtures()
        batch, _ = anchor_batch(segments, train_replacements(segments, metadata), range(3), True)
        changed = batch[0].clone()
        changed[..., -7:] += 10
        m = InputEvidenceTCN(TCNConfig(5, 7, 8, 3, 0), "visual_aug").eval()
        torch.testing.assert_close(m(batch[0], batch[1], batch[3]), m(changed, changed[..., -7:], batch[3]), atol=0, rtol=0)

    def test_sampler_independent_of_model_initialization(self):
        first = sampling_plan(10, 20)
        InputEvidenceTCN(TCNConfig(5, 7, 8), "visual_aug")
        self.assertEqual(first, sampling_plan(10, 20))

    def test_anchor_loss_never_supervises_donor_labels(self):
        segments, metadata = fixtures()
        batch, positions = anchor_batch(segments, train_replacements(segments, metadata), range(3), True)
        m = InputEvidenceTCN(TCNConfig(5, 7, 8, 3, 0), "visual_aug")
        self.assertTrue(torch.isfinite(worker.loss_on_batch(m, batch, "cpu", positions)))
        with self.assertRaises(ValueError):
            worker.loss_on_batch(m, batch, "cpu")

    def test_train_plan_rejects_val_donors(self):
        segments, metadata = fixtures()
        metadata[-1]["split"] = "val"
        with self.assertRaises(ValueError):
            train_replacements(segments, metadata)

    def test_eligible_native_and_challenge_use_identical_subset(self):
        labels = torch.tensor([0, 1, 2, 3])
        native = torch.eye(7)[labels] * 3
        changed = native.clone()
        changed[1] = torch.eye(7)[2] * 3
        data = {"labels": labels, "a": native}
        report = worker.eligible_audit(data, native, changed, [1, 3])
        self.assertEqual(report["native_eligible_metrics"]["count"], 2)
        self.assertEqual(report["eligible_metrics"]["count"], 2)
        self.assertEqual(report["paired_eligible_vs_native"]["worsened"], 1)
        self.assertEqual(report["paired_eligible_vs_native"]["improved"], 0)

    def test_empty_or_duplicate_intervention_cohort_not_misreported(self):
        scores = torch.ones(3, 7)
        data = {"labels": torch.tensor([0, 1, 2]), "a": scores}
        self.assertIsNone(worker.eligible_audit(data, scores, scores, [])["eligible_metrics"])
        with self.assertRaises(ValueError):
            worker.eligible_audit(data, scores, scores, [1, 1])

    def test_score_entropy_finite_for_confident_scores(self):
        x = torch.zeros(4, 7)
        x[:, 0] = 1000
        result = worker.score_statistics(x)
        self.assertEqual(result["mean_max_probability"], 1)
        self.assertEqual(result["mean_entropy"], 0)

    def test_anchor_signal_counts_only_preserved_targets(self):
        segments, metadata = fixtures()
        schedule = train_replacements(segments, metadata)
        m = InputEvidenceTCN(TCNConfig(5, 7, 8, 3, 0), "visual_aug")
        result = worker.anchor_signal(m, segments, schedule, True, "cpu")
        self.assertEqual(result["count"], len(schedule["rows"]))

    def test_public_report_rejects_private_paths_or_ids(self):
        for value in ({"clip_id": "example"}, {"clip_ids": []}, {"patient_id": "example"},
                      {"recording_key": "example"}, {"text": "/root/private/example"}):
            with self.assertRaises(RuntimeError):
                worker.public_safe(value)
        self.assertEqual(worker.public_safe({"macro_f1": .9}), {"macro_f1": .9})

    def test_factorial_contrasts_have_correct_direction_and_require_four_cells(self):
        f1 = {"logits": .7, "logits_aug": .75, "visual": .8, "visual_aug": .9}
        result = worker.factorial_contrasts(f1)
        self.assertAlmostEqual(result["mask_without_augmentation"], .1)
        self.assertAlmostEqual(result["augmentation_without_logits"], .1)
        self.assertAlmostEqual(result["mask_by_augmentation_interaction"], .05)
        with self.assertRaises(ValueError):
            worker.factorial_contrasts({"logits": .7})


class ArtifactAndEvaluationTests(unittest.TestCase):
    """Real CPU checkpoint replay/evaluation using explicitly synthetic assets."""

    def setup_synthetic_assets(self, root):
        segments, metadata = fixtures()
        for row in metadata:
            row["split"] = "val"
            row["duration"] = 3.0
        config = {"global_dim": 5, "local_dim": 7, "hidden_dim": 8, "blocks": 3, "dropout": 0}
        p = {"source_commit": "1" * 40, "schema": run_suite.CONFIG["schema"],
             "training": copy.deepcopy(run_suite.CONFIG), "backbone": "videomae2",
             "variants": list(VARIANTS), "validation_role": "synthetic_CPU_fixture_not_real_val823",
             "evidence": {"counts": {"train": 12, "val": 12}, "model_config": config,
                          "input_sha256": {"synthetic_only": "0" * 64}}}
        seal = {"sampling_hash": "a" * 64, "train_replacement_hash": "b" * 64}
        all_segments = {"train": segments, "val": segments}
        a = torch.from_numpy(np.concatenate([s.a_logits for s in segments]))
        labels = torch.from_numpy(np.concatenate([s.labels for s in segments]))
        val = {"a": a, "labels": labels, "t": a.clone(), "clip_ids": [c for s in segments for c in s.clip_ids]}
        fixed = {"train_replacements": {"rows": [{"synthetic_only": True}]},
                 "validation": perturb.plans(segments, metadata)}
        for variant in VARIANTS:
            m = worker.model(p, variant, "cpu").eval()
            z, _, _ = worker.infer(m, segments, "cpu")
            metric = worker.simple_metrics(val, z)
            directory = root / variant
            directory.mkdir()
            checkpoint = {"schema": run_suite.CONFIG["schema"], "source_commit": p["source_commit"],
                          "variant": variant, "backbone": p["backbone"], "input_policy": input_policy(variant),
                          "input_sha256": p["evidence"]["input_sha256"], "training": p["training"], **seal,
                          "model": m.state_dict(), "epoch": 1, "macro_f1": metric["macro_f1"], "accuracy": metric["accuracy"]}
            worker._torch_save(directory / "best.pt", checkpoint)
            (directory / "result.json").write_text(json.dumps({"training_completed": True, "variant": variant,
                "input_policy": input_policy(variant), "epochs_completed": 60, "optimizer_steps": 60,
                "best_epoch": 1, "best_macro_f1": metric["macro_f1"],
                "best_accuracy_same_F1_checkpoint": metric["accuracy"], "best_sha256": sha256_file(directory / "best.pt")}))
            (directory / "history.json").write_text(json.dumps([{"epoch": i + 1, "cumulative_steps": i + 1}
                                                               for i in range(60)]))
        return p, all_segments, metadata, fixed, {"val": val}, seal

    def test_complete_CPU_evaluation_replays_all_four_and_exports_no_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, segments, metadata, fixed, data, seal = self.setup_synthetic_assets(root)
            worker.evaluate(p, root, segments, metadata, fixed, data, seal, torch.device("cpu"))
            public = json.loads((root / "evaluation/public_safe_summary.json").read_text())
            self.assertEqual(set(public["models"]), set(VARIANTS))
            self.assertTrue(public["test_read"] is False)
            worker.public_safe(public)
            for variant, result in public["models"].items():
                self.assertEqual(len(result["metrics"]["per_class"]), 7)
                self.assertEqual(result["training_result"]["input_policy"], input_policy(variant))
                for name in ("shuffle", "cross_clinical"):
                    entry = result["interventions"][name]
                    self.assertEqual(entry["native_eligible_metrics"]["count"], entry["eligible_metrics"]["count"])
            with self.assertRaises(FileExistsError):
                worker.evaluate(p, root, segments, metadata, fixed, data, seal, torch.device("cpu"))

    def test_wrong_input_policy_or_missing_checkpoint_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, segments, _, _, _, seal = self.setup_synthetic_assets(root)
            ck_path = root / "visual/best.pt"
            ck = torch.load(ck_path, weights_only=True)
            ck["input_policy"] = "seven_raw_A_logits"
            worker._torch_save(ck_path, ck)
            r_path = root / "visual/result.json"
            r = json.loads(r_path.read_text())
            r["best_sha256"] = sha256_file(ck_path)
            r_path.write_text(json.dumps(r))
            with self.assertRaises(RuntimeError):
                worker.load_completed(p, root, segments, seal, "visual", "cpu")
            ck["input_policy"] = input_policy("visual")
            ck["model"].pop("base.classifier.bias")
            worker._torch_save(ck_path, ck)
            r["best_sha256"] = sha256_file(ck_path)
            r_path.write_text(json.dumps(r))
            with self.assertRaises(RuntimeError):
                worker.load_completed(p, root, segments, seal, "visual", "cpu")

    def test_partial_history_is_not_a_completed_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, segments, _, _, _, seal = self.setup_synthetic_assets(root)
            (root / "visual/history.json").write_text(json.dumps([{"epoch": 1, "cumulative_steps": 1}]))
            with self.assertRaises(RuntimeError):
                worker.load_completed(p, root, segments, seal, "visual", "cpu")


class LauncherTests(unittest.TestCase):
    def test_one_two_three_four_gpu_schedules(self):
        for count, lengths in ((1, [1, 1, 1, 1]), (2, [2, 2]), (3, [3, 1]), (4, [4])):
            w = run_suite.waves(list(range(count)), ["GPU-" + str(i) * 16 for i in range(count)])
            self.assertEqual([len(x) for x in w], lengths)
            self.assertEqual([s["variant"] for wave in w for s in wave], list(VARIANTS))
            self.assertEqual([s["gpu"] for s in w[-1]], list(range(len(w[-1]))))

    def test_invalid_or_duplicate_gpu_mappings_rejected(self):
        for g, u in (([], []), ([0, 0], ["a", "b"]), ([0, 1], ["a", "a"]),
                     ([-1], ["a"]), ([0, 1], ["a"]), ([0, 1, 2, 3, 4], list("abcde"))):
            with self.assertRaises(ValueError):
                run_suite.waves(g, u)

    def test_plan_has_four_cells_and_never_queries_GPU_or_creates_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "private_new"
            args = SimpleNamespace(gpus="0,1,2", gpu_uuids="GPU-00000000,GPU-11111111,GPU-22222222")
            base = {"config_sha256": "base", "evidence": {"model_config": {"global_dim": 5, "local_dim": 7}},
                    "output": str(output), "plan_only_does_not_probe_GPU_or_create_output": True}
            with patch.object(run_suite.shared, "plan", return_value=base), \
                    patch.object(run_suite, "wait_released_gpu", side_effect=AssertionError("GPU probed")):
                p = run_suite.plan(args)
            self.assertFalse(output.exists())
            self.assertEqual(p["variants"], list(VARIANTS))
            self.assertTrue(p["no_new_gate"])
            self.assertEqual(len(set(v["parameters"] for v in p["parameters"].values())), 1)

    def fake_execute(self, root, failure=None, smoke_bad_policy=False, incomplete_training=False):
        feature_dir = root / "features"
        feature_dir.mkdir()
        index = feature_dir / "index.jsonl"
        index.write_text("synthetic-test-only")
        source = root / "source"
        source.mkdir()
        p = {"output": str(root / "new"), "index": str(index), "source_protocol_dir": str(source),
             "source_commit": "1" * 40, "python": sys.executable, "variants": list(VARIANTS),
             "waves": run_suite.waves([0, 1, 2], ["GPU-" + str(i) * 16 for i in range(3)])}
        started, waited = [], []

        class Child:
            def __init__(self, command, **kwargs):
                self.pid = 100 + len(started)
                self.stage = "unit_tests" if "unittest" in command else command[command.index("--stage") + 1]
                self.variant = command[command.index("--variant") + 1] if "--variant" in command else "all"
                self.command = command
                started.append((self.stage, self.variant))
                expected_uuid = "" if self.variant == "all" else next(
                    s["gpu_uuid"] for w in p["waves"] for s in w if s["variant"] == self.variant)
                if self.stage == "evaluate":
                    expected_uuid = p["waves"][0][0]["gpu_uuid"]
                assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == expected_uuid

            def wait(self):
                waited.append((self.stage, self.variant))
                if (self.stage, self.variant) == failure:
                    return 1
                out = Path(p["output"])
                if self.stage == "smoke":
                    d = out / "smoke"
                    d.mkdir(exist_ok=True)
                    policy = "wrong" if smoke_bad_policy else input_policy(self.variant)
                    (d / (self.variant + ".json")).write_text(json.dumps({"all_passed": True, "input_policy": policy}))
                if self.stage == "train":
                    d = out / self.variant
                    d.mkdir()
                    (d / "best.pt").write_bytes(b"synthetic-test-only-not-a-checkpoint")
                    (d / "history.json").write_text(json.dumps([{}] * 60))
                    (d / "result.json").write_text(json.dumps({"training_completed": not incomplete_training,
                        "variant": self.variant, "input_policy": input_policy(self.variant),
                        "epochs_completed": 60, "best_sha256": sha256_file(d / "best.pt")}))
                if self.stage == "evaluate":
                    d = out / "evaluation"
                    d.mkdir()
                    (d / "public_safe_summary.json").write_text(json.dumps({"training_completed": True,
                                                                          "models": dict.fromkeys(VARIANTS, {})}))
                return 0

        with patch.dict("os.environ", {}, clear=True), patch.object(run_suite, "unchanged"), \
                patch.object(run_suite, "wait_released_gpu"), patch.object(run_suite.subprocess, "Popen", Child):
            if failure or smoke_bad_policy or incomplete_training:
                with self.assertRaises(RuntimeError):
                    run_suite.execute(p)
            else:
                run_suite.execute(p)
        return p, started, waited

    def test_all_four_smokes_precede_training_and_second_wave_waits(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, started, waited = self.fake_execute(Path(tmp))
            self.assertLess(max(i for i, x in enumerate(started) if x[0] == "smoke"),
                            min(i for i, x in enumerate(started) if x[0] == "train"))
            self.assertEqual([x[1] for x in waited if x[0] == "train"], list(VARIANTS))
            self.assertIn(("evaluate", "all"), waited)
            self.assertEqual(json.loads((Path(p["output"]) / "suite_result.json").read_text())["status"], "complete")

    def test_failed_first_wave_waits_for_peers_and_stops_visual_aug(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, started, waited = self.fake_execute(Path(tmp), failure=("train", "logits_aug"))
            self.assertEqual([x[1] for x in waited if x[0] == "train"], list(VARIANTS[:3]))
            self.assertNotIn(("train", "visual_aug"), started)
            self.assertNotIn(("evaluate", "all"), started)

    def test_failed_smoke_or_wrong_input_policy_prevents_training(self):
        for failure, bad in ((("smoke", "visual"), False), (None, True)):
            with tempfile.TemporaryDirectory() as tmp:
                _, started, _ = self.fake_execute(Path(tmp), failure=failure, smoke_bad_policy=bad)
                self.assertFalse(any(x[0] == "train" for x in started))

    def test_exit_zero_incomplete_training_stops_second_wave(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, started, waited = self.fake_execute(Path(tmp), incomplete_training=True)
            self.assertEqual([x[1] for x in waited if x[0] == "train"], list(VARIANTS[:3]))
            self.assertNotIn(("train", "visual_aug"), started)

    def test_existing_output_and_outer_CUDA_mapping_rejected(self):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}), \
                patch.object(run_suite, "wait_released_gpu") as guard:
            with self.assertRaises(RuntimeError):
                run_suite.execute({})
            guard.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "old"
            output.mkdir()
            marker = output / "keep"
            marker.write_text("existing-original")
            p = {"output": str(output), "index": str(root / "index"), "source_protocol_dir": str(root / "source"),
                 "waves": run_suite.waves([0], ["GPU-00000000"])}
            with patch.dict("os.environ", {}, clear=True), patch.object(run_suite, "unchanged"), \
                    patch.object(run_suite, "wait_released_gpu"):
                with self.assertRaises(FileExistsError):
                    run_suite.execute(p)
            self.assertEqual(marker.read_text(), "existing-original")

    def test_changed_config_or_missing_cell_rejected(self):
        p = {"schema": run_suite.CONFIG["schema"], "training": copy.deepcopy(run_suite.CONFIG),
             "variants": list(VARIANTS), "config_sha256": sha256_file(run_suite.CONFIG_PATH),
             "shared_config_sha256": "base"}
        with patch.object(run_suite.shared, "unchanged"):
            run_suite.unchanged(p)
            p["training"]["epochs"] = 30
            with self.assertRaises(RuntimeError):
                run_suite.unchanged(p)


if __name__ == "__main__":
    unittest.main()
