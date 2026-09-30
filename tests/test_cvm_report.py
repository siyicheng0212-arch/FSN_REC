import json
from pathlib import Path
import tempfile
import unittest

from cvm.report import collect_runs


class ReportTests(unittest.TestCase):
    def make_run(self, root, seed, status="completed", epoch=1, lr=.1):
        path = Path(root) / f"run{seed}_{epoch}"
        path.mkdir()
        config = {"seed": seed, "backbone": "r2plus1d_18", "mode": "flat", "protocol_sha256": "a" * 64,
                  "taxonomy_definition": {"groups": [[0], [6], [1, 2, 5], [3, 4]]},
                  "code_hash": "b" * 64, "head_lr": lr, "output": "/private/patient-alice"}
        result = {"status": status, "training_completed": status == "completed", "protocol_sha256": "a" * 64}
        report = {"metadata": {"protocol_sha256": "a" * 64}, "num_recording_groups": 7,
                  "methods": {"main": {"macro_f1": .5, "accuracy": .6, "per_class": [], "confusion_matrix": []}},
                  "ambiguous_five": {"methods": {"main": {"macro_f1": .4}}}}
        for name, data in (("config", config), ("result", result), ("val_report", report)):
            (path / f"{name}.json").write_text(json.dumps(data))
        return path

    def test_incomplete_runs_are_visible_and_private_path_not_emitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make_run(tmp, 42)
            b = self.make_run(tmp, 2026, "interrupted")
            r = collect_runs([a, b])
            self.assertFalse(r["methods"][0]["all_declared_seeds_completed"])
            self.assertEqual(r["methods"][0]["missing_seeds"], [2027])
            self.assertIsNone(r["methods"][0]["seed_summary"]["val_macro_f1"]["sd"])
            self.assertNotIn("patient-alice", json.dumps(r))
            self.assertEqual(len(r["methods"][0]["runs"]), 2)

    def test_favorable_duplicate_seed_and_changed_settings_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make_run(tmp, 42)
            b = self.make_run(tmp, 42, epoch=2)
            with self.assertRaises(ValueError):
                collect_runs([a, b])
            c = self.make_run(tmp, 2026, lr=.2)
            with self.assertRaises(ValueError):
                collect_runs([a, c])


from copy import deepcopy
from cvm.analysis import metric_report
from cvm.report import collect_plans, main, render_markdown
from cvm.taxonomy import get_taxonomy


class PlannedReportTests(unittest.TestCase):
    PROTOCOL = "a" * 64
    CODE = "b" * 64
    WEIGHTS = "c" * 64
    CHECKPOINT = "d" * 64

    def specification(self, mode="flat", *, backbone="r2plus1d_18", taxonomy="clinical", frames=16,
                      weighting="none", training="finetune"):
        identifier = f"{backbone}__{mode}__{taxonomy}__f{frames}__w{weighting}__bt{training}"
        return {"configuration_id": identifier, "backbone": backbone, "mode": mode, "taxonomy": taxonomy,
                "overrides": {"frames": frames, "class_weights": weighting, "backbone_training": training},
                "purpose": "synthetic_fixture", "purposes": ["synthetic_fixture"], "task_type": "train"}

    def plan(self, root, specs, *, seeds=(42,), stage="pilot", contrasts=(), filename="plan.json"):
        jobs = [{**deepcopy(spec), "seed": seed, "main_decoder": "soft" if spec["mode"] == "hierarchy" else "flat",
                 "checkpoint_sha256": self.WEIGHTS,
                 "output": str(Path(root) / "runs" / (spec["configuration_id"] + f"__seed{seed}"))}
                for spec in specs for seed in seeds]
        plan = {"schema_version": "fsn-cvm-plan-1", "stage": stage, "protocol_sha256": self.PROTOCOL,
                "code_hash": self.CODE, "settings": {"head_lr": .1}, "configurations": deepcopy(specs),
                "jobs": jobs, "contrasts": list(contrasts), "private_cache": "/private/patient-alice/cache"}
        path = Path(root) / filename
        path.write_text(json.dumps(plan))
        return path, plan

    def write(self, path, name, value):
        path.mkdir(parents=True, exist_ok=True)
        (path / (name + ".json")).write_text(json.dumps(value))

    def complete(self, job, *, error=False, crop=112, pretraining=None):
        path = Path(job["output"])
        spec = job["overrides"]
        taxonomy = get_taxonomy(job["taxonomy"])
        groups = taxonomy.to_dict()["groups"]
        heads = {"flat": ["flat"], "capacity_control": ["flat", "capacity"],
                 "aux_flat": ["flat", "group", "conditional"], "hierarchy": ["group", "conditional"]}[job["mode"]]
        preprocessing = {"frames": spec["frames"], "crop_size": [crop, crop], "resize_size": [128, 171], "mean": [.4, .4, .4], "std": [.2, .2, .2]}
        config = {"seed": job["seed"], "backbone": job["backbone"], "mode": job["mode"], "head_lr": .1,
                  "protocol_sha256": self.PROTOCOL, "code_hash": self.CODE, "checkpoint_sha256": job["checkpoint_sha256"],
                  "taxonomy_definition": taxonomy.to_dict(), "trained_heads": heads, "preprocessing": preprocessing,
                  "frames": spec["frames"], "class_weights": spec["class_weights"], "backbone_training": spec["backbone_training"],
                  "aux_weight": 1., "group_weight": 1., "conditional_weight": 1.,
                  "main_decoder": "soft" if job["mode"] == "hierarchy" else "flat", "parameters_trainable_finetune": 123,
                  "output": "/private/patient-alice", "private_note": "PRIVATE_NOTE"}
        metadata = {"mode": job["mode"], "trained_heads": heads, "taxonomy": taxonomy.to_dict(),
                    "protocol_sha256": self.PROTOCOL, "split": "val", "selection_exposure": "used_each_epoch_for_checkpoint_selection",
                    "checkpoint_sha256": self.CHECKPOINT,
                    "eval_metadata": {"backbone": job["backbone"], "preprocessing": preprocessing, "cache_frames": 36,
                                      "cache_size": 224, "probe": "none", "probe_seed": job["seed"],
                                      "repeated_frame_fraction_basis": f"selected {spec['frames']} frames", "temporal_positions": "uniform"},
                    "private_note": "PRIVATE_SIDECAR"}
        rows = []
        for index in range(14):
            target = index % 7
            predicted = 1 if error and index == 0 else target
            flat = [8. if c == predicted else 0. for c in range(7)]
            router = [8. if predicted in group else 0. for group in groups]
            conditional = [[8. if c == predicted else 0. for c in group] for group in groups]
            rows.append({"clip_id": f"PRIVATE_CLIP_{index}", "group_id": f"PRIVATE_RECORDING_{index % 4}",
                         "source": "/private/patient-alice/source.mp4", "duration": 12., "target": target,
                         "flat_logits": flat if "flat" in heads else None,
                         "group_logits": router if "group" in heads else None,
                         "conditional_logits": conditional if "group" in heads else None,
                         "repeated_frame_fraction": .1})
        report = metric_report(rows, metadata=metadata)
        result = {"status": "completed", "training_completed": True, "protocol_sha256": self.PROTOCOL,
                  "best_epoch": 1, "seconds": 3., "peak_cuda_allocated_bytes": 456}
        for name, value in (("config", config), ("result", result), ("prediction_metadata", metadata), ("val_report", report),
                            ("load_report", pretraining or {"pretrained": True, "source": "/private/patient-alice/weights.pt"})):
            self.write(path, name, value)
        return path, config, report, metadata

    def contrast(self, baseline, candidate, *, kind="paired_training", factors=()):
        return {"contrast_id": "synthetic_contrast", "baseline_configuration_id": baseline["configuration_id"],
                "candidate_configuration_id": candidate["configuration_id"], "comparison_kind": kind, "varying_factors": list(factors)}

    def test_unstarted_jobs_and_missing_review_controls_never_fabricate_scores(self):
        with tempfile.TemporaryDirectory() as root:
            specs = [self.specification(), self.specification("hierarchy")]
            path, _ = self.plan(root, specs, seeds=(42, 2026, 2027), stage="formal", contrasts=[self.contrast(*specs)])
            result = collect_plans([path])
            self.assertFalse(result["all_declared_jobs_completed"])
            self.assertTrue(all(row["seed_summary"] is None for row in result["experiment_rows"]))
            self.assertEqual(result["comparison_rows"][0]["paired_seed_summary"], None)
            coverage = result["review_evidence"]["modern_backbone_comparison"]
            self.assertEqual(len(coverage["missing_required_controls"]), 3)
            self.assertEqual(coverage["status"], "declared_incomplete")
            self.assertIn("pending", render_markdown(result))
            self.assertNotIn("patient-alice", json.dumps(result))

    def test_complete_primary_metrics_diagnostics_and_privacy(self):
        with tempfile.TemporaryDirectory() as root:
            specs = [self.specification(), self.specification("hierarchy")]
            path, plan = self.plan(root, specs, seeds=(42, 2026, 2027), stage="formal", contrasts=[self.contrast(*specs)])
            for job in plan["jobs"]:
                self.complete(job, error=job["mode"] == "flat")
            result = collect_plans([path])
            self.assertTrue(result["all_declared_jobs_completed"])
            self.assertFalse(result["formal_claim_ready"])
            contrast = result["comparison_rows"][0]
            self.assertEqual(contrast["completed_paired_seeds"], [42, 2026, 2027])
            self.assertGreater(contrast["paired_seed_summary"]["five_macro_f1_delta"]["mean"], 0.)
            self.assertIsNone(contrast["cluster_bootstrap_interval"])
            hierarchy = next(row for row in result["experiment_rows"] if row["mode"] == "hierarchy")
            diagnostics = hierarchy["runs"][0]["diagnostics"]
            self.assertEqual(diagnostics["routing"]["actual_router"]["macro_f1"], 1.)
            self.assertEqual(len(diagnostics["routing"]["actual_router"]["per_group"]), 4)
            self.assertEqual(diagnostics["same_checkpoint_hard_soft"]["hard_five"]["macro_f1"], 1.)
            self.assertIsNotNone(diagnostics["oracle_hierarchy"])
            self.assertEqual(diagnostics["singleton_contribution"]["num_samples"], 4)
            self.assertTrue(any(key.startswith("source:source-") for key in diagnostics["slices"]))
            for text in (json.dumps(result), render_markdown(result)):
                for private in ("patient-alice", "PRIVATE_NOTE", "PRIVATE_SIDECAR", "PRIVATE_CLIP", "PRIVATE_RECORDING"):
                    self.assertNotIn(private, text)
            self.assertEqual(result["review_evidence"]["modern_backbone_comparison"]["status"], "partial_requirement_coverage")

    def test_five_f1_must_preserve_singleton_false_positives(self):
        with tempfile.TemporaryDirectory() as root:
            path, plan = self.plan(root, [self.specification()])
            directory, _, report, _ = self.complete(plan["jobs"][0], error=True)
            report["ambiguous_five"]["methods"]["main"]["macro_f1"] = 1.
            self.write(directory, "val_report", report)
            with self.assertRaisesRegex(ValueError, "confusion matrix|false positives"):
                collect_plans([path])

    def test_protocol_settings_and_taxonomy_mislabeling_rejected(self):
        for mutation in ("protocol", "head_lr", "taxonomy", "frames", "backbone_training"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                path, plan = self.plan(root, [self.specification()])
                directory, config, _, _ = self.complete(plan["jobs"][0])
                if mutation == "protocol":
                    config["protocol_sha256"] = "e" * 64
                elif mutation == "taxonomy":
                    config["taxonomy_definition"] = get_taxonomy("random_17").to_dict()
                elif mutation == "frames":
                    config["frames"] = config["preprocessing"]["frames"] = 32
                elif mutation == "backbone_training":
                    config["backbone_training"] = "frozen"
                else:
                    config["head_lr"] = .2
                self.write(directory, "config", config)
                with self.assertRaises(ValueError):
                    collect_plans([path])

    def test_pairing_rejects_pretraining_resolution_and_undeclared_treatments(self):
        for mutation in ("checkpoint", "crop", "cache_size"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                specs = [self.specification(), self.specification("hierarchy")]
                path, plan = self.plan(root, specs, contrasts=[self.contrast(*specs)])
                self.complete(plan["jobs"][0])
                job = plan["jobs"][1]
                if mutation == "checkpoint":
                    job["checkpoint_sha256"] = "e" * 64
                    path.write_text(json.dumps(plan))
                directory, config, _, metadata = self.complete(job, crop=224 if mutation == "crop" else 112)
                if mutation == "cache_size":
                    metadata["eval_metadata"]["cache_size"] = 256
                    self.write(directory, "prediction_metadata", metadata)
                with self.assertRaises(ValueError):
                    collect_plans([path])

    def test_single_factor_frames_and_representation_are_explicit(self):
        for factor, candidate in (("frames", self.specification(frames=32)), ("backbone_training", self.specification(training="frozen")), ("class_weights", self.specification(weighting="sqrt_inverse"))):
            with self.subTest(factor=factor), tempfile.TemporaryDirectory() as root:
                baseline = self.specification()
                stage = "representation" if factor == "backbone_training" else "sensitivity"
                path, plan = self.plan(root, [baseline, candidate], stage=stage, contrasts=[self.contrast(baseline, candidate, kind="single_factor", factors=(factor,))])
                for job in plan["jobs"]:
                    self.complete(job)
                result = collect_plans([path])
                self.assertEqual(result["comparison_rows"][0]["status"], "complete_development_comparison")
                self.assertEqual(len(result["experiment_rows"]), 2)
                if factor == "frames":
                    job = plan["jobs"][1]
                    directory = Path(job["output"])
                    metadata = json.loads((directory / "prediction_metadata.json").read_text())
                    metadata["eval_metadata"]["cache_size"] = 256
                    self.write(directory, "prediction_metadata", metadata)
                    with self.assertRaisesRegex(ValueError, "evaluation treatment"):
                        collect_plans([path])

    def test_cross_backbone_table_discloses_native_differences_but_no_paired_claim(self):
        with tempfile.TemporaryDirectory() as root:
            specs = [self.specification(), self.specification(backbone="videomae_base16")]
            path, plan = self.plan(root, specs, stage="benchmark")
            self.complete(plan["jobs"][0])
            self.complete(plan["jobs"][1], crop=224, pretraining={"declared_checkpoint_route": "K400 self-supervised -> K400 supervised", "source": "PRIVATE_SOURCE"})
            result = collect_plans([path])
            self.assertFalse(result["cross_backbone_disclosure"]["equal_compute_claim"])
            mae = next(row for row in result["experiment_rows"] if row["backbone"] == "videomae_base16")
            self.assertEqual(mae["runs"][0]["pretraining"]["datasets"], ["Kinetics-400"])
            plan["contrasts"] = [self.contrast(*specs)]
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "mix backbones"):
                collect_plans([path])

    def test_same_output_stage_reuse_allowed_but_favorable_reruns_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path, plan = self.plan(root, [self.specification()])
            self.complete(plan["jobs"][0])
            alternative = deepcopy(plan)
            alternative["stage"] = "benchmark"
            path2 = Path(root) / "second_plan.json"
            path2.write_text(json.dumps(alternative))
            result = collect_plans([path, path2])
            self.assertEqual(result["experiment_rows"][0]["stages"], ["benchmark", "pilot"])
            self.assertEqual(result["experiment_rows"][0]["completed_seeds"], [42])
            alternative["jobs"][0]["output"] += "_favorable"
            path2.write_text(json.dumps(alternative))
            with self.assertRaisesRegex(ValueError, "duplicate configuration/seed"):
                collect_plans([path, path2])

    def test_robustness_is_same_checkpoint_probe_only_and_never_training(self):
        with tempfile.TemporaryDirectory() as root:
            _, training_plan = self.plan(root, [self.specification()])
            source_job = training_plan["jobs"][0]
            training_dir, _, report, metadata = self.complete(source_job)
            specs = []
            jobs = []
            for probe in ("none", "static", "shuffle"):
                spec = {**self.specification(), "configuration_id": source_job["configuration_id"] + "__probe_" + probe,
                        "probe": probe, "task_type": "evaluate", "train_configuration_id": source_job["configuration_id"]}
                specs.append(spec)
                job = {**spec, "seed": 42, "checkpoint_sha256": self.CHECKPOINT,
                       "output": str(Path(root) / spec["configuration_id"]), "trained_run_output": str(training_dir)}
                jobs.append(job)
                md = deepcopy(metadata)
                md["eval_metadata"]["probe"] = probe
                self.write(Path(job["output"]), "prediction_metadata", md)
                self.write(Path(job["output"]), "report", report)
                self.write(Path(job["output"]), "result", {"status": "evaluation_complete", "split": "val", "probe": probe, "training_started": False})
            path = Path(root) / "probes.json"
            plan = {"schema_version": "fsn-cvm-plan-1", "stage": "robustness", "protocol_sha256": self.PROTOCOL,
                    "code_hash": self.CODE, "jobs": jobs, "configurations": specs,
                    "contrasts": [self.contrast(specs[0], specs[1], kind="same_checkpoint_probe", factors=("probe",))]}
            path.write_text(json.dumps(plan))
            result = collect_plans([path])
            self.assertEqual(result["comparison_rows"][0]["status"], "complete_development_comparison")
            self.assertFalse(result["training_or_inference_started"])
            self.assertEqual(result["review_evidence"]["robustness_probes"]["missing_required_controls"], [])
            md = deepcopy(metadata)
            md["eval_metadata"]["probe"] = "static"
            md["eval_metadata"]["cache_size"] = 256
            self.write(Path(jobs[1]["output"]), "prediction_metadata", md)
            with self.assertRaisesRegex(ValueError, "changes more than"):
                collect_plans([path])
            md["checkpoint_sha256"] = "e" * 64
            self.write(Path(jobs[1]["output"]), "prediction_metadata", md)
            with self.assertRaisesRegex(ValueError, "trained checkpoint"):
                collect_plans([path])

    def test_failed_and_interrupted_never_promote_saved_validation_scores(self):
        for status in ("failed", "interrupted"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as root:
                path, plan = self.plan(root, [self.specification()])
                directory, _, _, _ = self.complete(plan["jobs"][0])
                self.write(directory, "result", {"status": status, "training_completed": False})
                result = collect_plans([path])
                self.assertEqual(result["experiment_rows"][0]["runs"][0]["status"], status)
                self.assertIsNone(result["experiment_rows"][0]["seed_summary"])

    def test_small_source_slices_suppressed_even_if_upstream_flag_missing(self):
        with tempfile.TemporaryDirectory() as root:
            path, plan = self.plan(root, [self.specification()])
            directory, _, report, _ = self.complete(plan["jobs"][0])
            report["slices"]["results"]["source:PRIVATE_CATEGORY"] = {"num_samples": 1, "num_recording_groups": 1, "methods": {"main": report["methods"]["main"]}}
            self.write(directory, "val_report", report)
            result = collect_plans([path])
            slices = result["experiment_rows"][0]["runs"][0]["diagnostics"]["slices"]
            self.assertTrue(any(block["suppressed"] and block["main"] is None for block in slices.values()))
            self.assertNotIn("PRIVATE_CATEGORY", json.dumps(result))

    def test_cli_writes_safe_json_and_markdown_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            plan_path, _ = self.plan(root, [self.specification()])
            output = Path(root) / "summary.json"
            markdown = Path(root) / "tables.md"
            argv = ["--plan", str(plan_path), "--output", str(output), "--markdown-output", str(markdown)]
            main(argv)
            self.assertIn("pending", markdown.read_text())
            self.assertFalse(json.loads(output.read_text())["training_or_inference_started"])
            with self.assertRaises(SystemExit) as caught:
                main(argv)
            self.assertEqual(caught.exception.code, 2)


    def test_visual_alias_uses_frozen_actual_taxonomy_name_and_train_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            spec = self.specification("hierarchy")
            path, plan = self.plan(root, [spec], stage="ablation")
            job = plan["jobs"][0]
            directory, config, report, metadata = self.complete(job)
            visual = deepcopy(config["taxonomy_definition"])
            visual["name"] = "train_feature_prototypes_v1"
            visual["provenance"] = {"split": "train"}
            job["taxonomy"] = "visual_train_only"
            job["configuration_id"] = job["configuration_id"].replace("__clinical__", "__visual_train_only__")
            job["taxonomy_file_sha256"] = "e" * 64
            plan["configurations"] = [{key: value for key, value in job.items() if key not in ("seed", "output")}]
            path.write_text(json.dumps(plan))
            config["taxonomy_definition"] = visual
            config["taxonomy_file_metadata"] = visual
            config["taxonomy_file_sha256"] = "e" * 64
            metadata["taxonomy"] = visual
            self.write(directory, "config", config)
            self.write(directory, "prediction_metadata", metadata)
            result = collect_plans([path])
            self.assertEqual(result["experiment_rows"][0]["taxonomy"], "visual_train_only")
            config["taxonomy_file_metadata"]["provenance"]["split"] = "val"
            self.write(directory, "config", config)
            with self.assertRaisesRegex(ValueError, "training-only"):
                collect_plans([path])

    def test_short_clip_sensitivity_alias_preserved_in_safe_tables(self):
        with tempfile.TemporaryDirectory() as root:
            path, plan = self.plan(root, [self.specification()])
            directory, _, report, _ = self.complete(plan["jobs"][0])
            alias = "sensitivity:exclude_duration_lt_0_1s"
            report["slices"]["results"][alias] = {"num_samples": 14, "num_recording_groups": 4, "suppressed": False,
                                                     "methods": {"main": report["methods"]["main"]}}
            self.write(directory, "val_report", report)
            result = collect_plans([path])
            self.assertIn(alias, result["experiment_rows"][0]["runs"][0]["diagnostics"]["slices"])
            self.assertIn(alias, render_markdown(result))


if __name__ == "__main__":
    unittest.main()
