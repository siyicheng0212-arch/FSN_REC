"""Collect complete, predeclared training seeds without publishing private rows."""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Sequence

from cvm.analysis import CLASS_NAMES, _metrics_from_confusion
from cvm.taxonomy import Taxonomy, get_taxonomy

METRICS = ("macro_f1", "weighted_f1", "micro_f1", "accuracy", "macro_precision", "macro_recall")
TRAINING_SETTINGS = ("batch_size", "accum_steps", "effective_batch_size", "epochs", "warmup_epochs", "patience",
                    "optimizer", "backbone_lr", "head_lr", "warmup_lr", "weight_decay", "class_weights", "clip_grad", "augmentation", "amp", "backbone_training")
OBJECTIVE_SETTINGS = ("aux_weight", "group_weight", "conditional_weight", "main_decoder")
BACKBONES = ("r2plus1d_18", "mvit_v2_s", "videomamba_tiny16", "videomae_base16", "r3d_18")
MODES = ("flat", "capacity_control", "aux_flat", "hierarchy")
STAGES = ("smoke", "pilot", "benchmark", "ablation", "sensitivity", "representation", "formal", "extended", "robustness")


def _load(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        raise ValueError("required aggregate artifact is missing or invalid JSON") from None
    if not isinstance(result, dict):
        raise ValueError("expected JSON object")
    return result


def _stat(values: list[float]) -> dict:
    if not all(math.isfinite(v) for v in values):
        raise ValueError("non-finite summary metric")
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else None,
            "n_seeds": len(values)}


def collect_runs(run_directories: Sequence[Path], expected_seeds: Sequence[int] = (42, 2026, 2027)) -> dict:
    expected = tuple(int(seed) for seed in expected_seeds)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected seeds must be distinct and nonempty")
    grouped: dict[tuple, list[dict]] = {}
    protocol_shas = set()
    for directory in run_directories:
        config = _load(directory / "config.json")
        result = _load(directory / "result.json")
        protocol_sha = config["protocol_sha256"]
        if result.get("protocol_sha256") != protocol_sha:
            raise ValueError("run result and configuration protocol differ")
        protocol_shas.add(protocol_sha)
        taxonomy = config["taxonomy_definition"]
        key = (config["backbone"], config["mode"], tuple(tuple(g) for g in taxonomy["groups"]),
               config.get("frames", config.get("preprocessing", {}).get("frames", 16)),
               config.get("class_weights", "none"), config.get("backbone_training", "finetune"))
        record = {"seed": int(config["seed"]), "status": result["status"],
                  "training_completed": bool(result.get("training_completed")),
                  "best_epoch": result.get("best_epoch"), "seconds": result.get("seconds"),
                  "parameters_trainable_finetune": result.get("parameters_trainable_finetune"),
                  "peak_cuda_allocated_bytes": result.get("peak_cuda_allocated_bytes"),
                  "git_commit": config.get("git_commit"), "code_hash": config.get("code_hash"),
                  "pretrained_sha256": config.get("checkpoint_sha256"),
                  "hyperparameters": {k: config.get(k) for k in ("batch_size", "accum_steps", "epochs", "warmup_epochs", "patience",
                  "optimizer", "backbone_lr", "head_lr", "warmup_lr", "weight_decay", "class_weights", "clip_grad", "augmentation",
                  "main_decoder", "aux_weight", "group_weight", "conditional_weight", "preprocessing")}}
        if record["seed"] not in expected:
            raise ValueError("undeclared training seed; declare it before summarizing")
        if record["training_completed"]:
            if result["status"] not in ("completed", "early_stopped"):
                raise ValueError("completion flag conflicts with run status")
            report = _load(directory / "val_report.json")
            if report["metadata"]["protocol_sha256"] != protocol_sha:
                raise ValueError("prediction report protocol mismatch")
            main = report["methods"]["main"]
            five = report["ambiguous_five"]["methods"]["main"]
            record.update({"val_macro_f1": main["macro_f1"], "val_accuracy": main["accuracy"],
                           "val_five_macro_f1": five["macro_f1"], "per_class": main["per_class"],
                           "confusion_matrix": main["confusion_matrix"],
                           "evaluation_metadata": {"protocol_sha256": protocol_sha, "split": "validation"},
                           "num_recording_groups": report["num_recording_groups"]})
        grouped.setdefault(key, []).append(record)
    if not grouped or len(protocol_shas) != 1:
        raise ValueError("all collected runs must use one frozen protocol")
    output = []
    for (backbone, mode, groups, frames, class_weights, backbone_training), records in sorted(grouped.items()):
        if len({r["seed"] for r in records}) != len(records):
            raise ValueError("duplicate seed for one method; cannot select a favorable rerun")
        if len({json.dumps(r["hyperparameters"], sort_keys=True) for r in records}) != 1:
            raise ValueError("different hyperparameters cannot be pooled as seed replication")
        if len({r["code_hash"] for r in records}) != 1:
            raise ValueError("different implementations cannot be pooled as seed replication")
        if len({r["pretrained_sha256"] for r in records}) != 1:
            raise ValueError("different pretrained weights cannot be pooled as seed replication")
        completed = [r for r in records if r["training_completed"]]
        metrics = {key: _stat([r[key] for r in completed]) for key in
                   ("val_macro_f1", "val_accuracy", "val_five_macro_f1")} if completed else None
        output.append({"backbone": backbone, "mode": mode, "taxonomy_groups": [list(g) for g in groups],
                       "frames": frames, "class_weights": class_weights, "backbone_training": backbone_training,
                       "expected_seeds": list(expected), "missing_seeds": sorted(set(expected) - {r["seed"] for r in records}),
                       "all_declared_seeds_completed": len(completed) == len(expected),
                       "seed_summary": metrics, "runs": sorted(records, key=lambda r: r["seed"])})
    return {"schema_version": "fsn-cvm-seed-report-1", "protocol_sha256": next(iter(protocol_shas)),
            "split": "validation", "methods": output,
            "interpretation": "Validation selected the checkpoint. These are development estimates, not an independent test or a significance claim. Report every declared seed.",
            "privacy": {"aggregate_only": True, "raw_paths": False, "clip_ids": False, "recording_ids": False}}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,180}", value):
        raise ValueError("declared public configuration or contrast identifier is invalid")
    return value


def _sha(value: Any, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("aggregate provenance requires valid SHA-256 digests")
    return value.lower()


def _finite(value: Any, *, score: bool = False) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or (score and not 0 <= value <= 1):
        raise ValueError("aggregate metric must be finite and on the declared scale")
    return float(value)


def _input_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config.get("preprocessing", {})
    if not isinstance(spec, Mapping):
        raise ValueError("preprocessing provenance must be a mapping")
    public = {}
    for key in ("frames", "crop_size", "resize_size", "mean", "std"):
        value = spec.get(key)
        if value is not None:
            values = value if isinstance(value, (list, tuple)) else [value]
            if any(_finite(x) is None for x in values):
                raise ValueError("preprocessing numeric provenance is invalid")
            public[key] = list(value) if isinstance(value, (list, tuple)) else value
    frames = config.get("frames", public.get("frames", 16))
    if not isinstance(frames, int) or isinstance(frames, bool) or frames <= 0:
        raise ValueError("input frame count must be a positive integer")
    if "frames" in public and public["frames"] != frames:
        raise ValueError("configured input frames disagree with preprocessing")
    public["frames"] = frames
    return public


def _pretraining(config: Mapping[str, Any], load_report: Mapping[str, Any]) -> dict[str, Any]:
    # Copy dataset names, never the checkpoint's path, URL, or an arbitrary
    # load_report string. Unknown provenance stays unknown.
    text = " ".join(str(load_report.get(key, "")) for key in ("pretraining", "weights_enum", "declared_checkpoint_route"))
    patterns = (("ImageNet-1K", r"ImageNet[- _]?(?:1K|1000)"), ("Kinetics-400", r"(?:Kinetics[- _]?400|K400)"),
                ("Kinetics-600", r"(?:Kinetics[- _]?600|K600)"), ("Something-Something-V2", r"(?:Something[- _]Something[- _]V2|SSV2)"))
    datasets = [name for name, pattern in patterns if re.search(pattern, text, re.I)]
    if not datasets and config.get("backbone") in ("r2plus1d_18", "r3d_18", "mvit_v2_s") and load_report.get("pretrained") is True:
        datasets = ["Kinetics-400"]
    return {"checkpoint_sha256": _sha(config.get("checkpoint_sha256")), "datasets": datasets,
            "dataset_declaration_available": bool(datasets)}


def _safe_block(block: Any, *, validate_confusion: bool = False, classes: Sequence[int] = tuple(range(7))) -> dict[str, Any] | None:
    if not isinstance(block, Mapping):
        return None
    public = {key: _finite(block.get(key), score=True) for key in METRICS}
    count = block.get("num_samples", block.get("support"))
    if count is not None:
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("aggregate sample counts must be nonnegative integers")
        public["num_samples"] = count
    matrix = block.get("confusion_matrix")
    if matrix:
        if not isinstance(matrix, list) or len(matrix) != 7 or any(not isinstance(row, list) or len(row) != 7 for row in matrix) or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for row in matrix for x in row):
            raise ValueError("seven-class aggregate confusion matrix is invalid")
        computed = _metrics_from_confusion(matrix, classes)
        if validate_confusion:
            for metric in METRICS:
                if public[metric] is not None and not math.isclose(public[metric], computed[metric], abs_tol=1e-7):
                    raise ValueError("reported seven-class metrics disagree with their confusion matrix")
            if count is not None and count != computed["num_samples"]:
                raise ValueError("reported sample count disagrees with confusion matrix")
        public["confusion_matrix"] = matrix
        public["averaged_class_ids"] = list(classes)
        public["per_class"] = [{"class_id": item["class_id"], "class_name": CLASS_NAMES[item["class_id"]],
                                "support": item["support"], "precision": item["precision"], "recall": item["recall"], "f1": item["f1"]}
                               for item in computed["per_class"]]
    return public


def _safe_diagnostics(report: Mapping[str, Any]) -> dict[str, Any]:
    routing = report.get("routing", {})
    safe_routes = {}
    for name in ("actual_router", "flat_mapped_group", "flat_aggregated_probability_group"):
        block = routing.get(name) if isinstance(routing, Mapping) else None
        if isinstance(block, Mapping):
            safe_routes[name] = {key: _finite(block.get(key), score=True) for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "conditional_fine_recall_given_correct_route")}
            matrix = block.get("confusion_matrix")
            if matrix is not None:
                if not isinstance(matrix, list) or not matrix or len(matrix) > 7 or any(not isinstance(row, list) or len(row) != len(matrix) for row in matrix) or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for row in matrix for x in row):
                    raise ValueError("router aggregate confusion matrix is invalid")
                computed = _metrics_from_confusion(matrix, tuple(range(len(matrix))))
                safe_routes[name]["confusion_matrix"] = matrix
                safe_routes[name]["per_group"] = [{key: item[key] for key in ("class_id", "support", "precision", "recall", "f1")} for item in computed["per_class"]]
                for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall"):
                    if safe_routes[name][key] is not None and not math.isclose(safe_routes[name][key], computed[key], abs_tol=1e-7):
                        raise ValueError("router metrics disagree with their confusion matrix")
            conditional = block.get("per_class_conditional_recall", [])
            if not isinstance(conditional, list):
                raise ValueError("per-class conditional route diagnostics are invalid")
            safe_routes[name]["per_class_conditional_recall"] = []
            for item in conditional:
                if not isinstance(item, Mapping) or item.get("class_id") not in range(7) or not isinstance(item.get("route_correct_support"), int) or isinstance(item["route_correct_support"], bool) or item["route_correct_support"] < 0:
                    raise ValueError("per-class conditional route diagnostics are invalid")
                safe_routes[name]["per_class_conditional_recall"].append({"class_id": item["class_id"], "route_correct_support": item["route_correct_support"], "recall": _finite(item.get("recall"), score=True)})
            for key in ("num_route_correct", "num_route_errors"):
                value = block.get(key)
                if value is not None:
                    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                        raise ValueError("routing counts are invalid")
                    safe_routes[name][key] = value
        else:
            safe_routes[name] = None
    methods = report.get("methods", {})
    hard, soft = _safe_block(methods.get("hard"), validate_confusion=True), _safe_block(methods.get("soft"), validate_confusion=True)
    five_methods = report.get("ambiguous_five", {}).get("methods", {})
    hard_five = _safe_block(five_methods.get("hard"), validate_confusion=True, classes=(1, 2, 3, 4, 5))
    soft_five = _safe_block(five_methods.get("soft"), validate_confusion=True, classes=(1, 2, 3, 4, 5))
    same_checkpoint = {"hard": hard, "soft": soft,
                       "hard_five": hard_five, "soft_five": soft_five,
                       "soft_minus_hard_macro_f1": soft["macro_f1"] - hard["macro_f1"] if hard and soft and hard["macro_f1"] is not None and soft["macro_f1"] is not None else None,
                       "soft_minus_hard_five_macro_f1": soft_five["macro_f1"] - hard_five["macro_f1"] if hard_five and soft_five and hard_five["macro_f1"] is not None and soft_five["macro_f1"] is not None else None}
    changes = report.get("same_checkpoint_hard_soft")
    if isinstance(changes, Mapping):
        same_checkpoint["prediction_changes"] = {key: changes[key] for key in ("num_samples", "changed", "corrected", "harmed")
                                                  if isinstance(changes.get(key), int) and not isinstance(changes.get(key), bool) and changes[key] >= 0}
    slices = {}
    for key, value in report.get("slices", {}).get("results", {}).items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            raise ValueError("aggregate slice structure is invalid")
        if key.startswith("source:"):
            tail = key[len("source:"):]
            alias = tail if re.fullmatch(r"source-[0-9a-f]{12}", tail) else "source-" + hashlib.sha256(tail.encode()).hexdigest()[:12]
            public_key = "source:" + alias
        elif re.fullmatch(r"(?:duration|repeated_frames):[a-z0-9_]+", key) or key == "sensitivity:exclude_duration_lt_0_1s":
            public_key = key
        else:
            continue
        counts = {key_: value.get(key_) for key_ in ("num_samples", "num_recording_groups")}
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in counts.values()):
            raise ValueError("aggregate slice counts are invalid")
        suppressed = value.get("suppressed") is True or counts["num_samples"] < 10 or counts["num_recording_groups"] < 2
        slices[public_key] = {**counts, "suppressed": suppressed,
                              "main": None if suppressed else _safe_block(value.get("methods", {}).get("main"))}
    decomposition = {}
    for name, block in report.get("hard_error_decomposition", {}).items():
        if name in ("all", "ground_truth_ambiguous_five") and isinstance(block, Mapping):
            decomposition[name] = {key: block[key] for key in ("num_samples", "routing_errors", "within_group_errors_given_correct_route", "correct", "total_hard_errors")
                                   if isinstance(block.get(key), int) and not isinstance(block.get(key), bool) and block[key] >= 0}
    singleton = report.get("singleton_contribution", {})
    safe_singleton = {key: _finite(singleton.get(key), score=True) for key in ("accuracy_on_singletons", "contribution_to_seven_class_macro_f1", "main_seven_class_macro_f1_minus_singleton_terms")}
    for key in ("num_samples", "num_correct_main"):
        value = singleton.get(key)
        if value is not None:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("singleton aggregate counts are invalid")
            safe_singleton[key] = value
    return {"routing": safe_routes, "same_checkpoint_hard_soft": same_checkpoint, "slices": slices,
            "hard_error_decomposition": decomposition,
            "singleton_contribution": safe_singleton,
            "oracle_flat": _safe_block(methods.get("oracle_flat"), validate_confusion=True),
            "oracle_hierarchy": _safe_block(methods.get("oracle_hierarchy"), validate_confusion=True),
            "oracle_interpretation": "True-group explanatory diagnostic; not deployable performance or a fair separately trained flat-oracle comparison."}


def _read_plan_record(plan: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    backbone, mode = job.get("backbone"), job.get("mode")
    if backbone not in BACKBONES or mode not in MODES:
        raise ValueError("plan uses an undeclared backbone or classifier mode")
    identifier = _token(job.get("configuration_id", f"{backbone}__{mode}__{job.get('taxonomy', 'clinical')}"))
    seed = job.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("planned training seeds must be integers")
    task = job.get("task_type", "train")
    if task not in ("train", "evaluate"):
        raise ValueError("unknown planned task type")
    directory = Path(job["output"])
    declared_settings = {**plan.get("settings", {}), **job.get("overrides", {})}
    training = declared_settings.get("backbone_training", "finetune")
    if training not in ("finetune", "frozen") or declared_settings.get("class_weights", "none") not in ("none", "sqrt_inverse"):
        raise ValueError("declared representation/weight settings are invalid")
    frames = declared_settings.get("frames", 16)
    if not isinstance(frames, int) or isinstance(frames, bool) or frames not in (16, 32):
        raise ValueError("declared input frame count is invalid")
    probe = job.get("probe", "none")
    if probe not in ("none", "static", "shuffle") or (task == "train" and probe != "none"):
        raise ValueError("declared evaluation probe is invalid")
    record = {"configuration_id": identifier, "seed": seed, "backbone": backbone, "mode": mode,
              "taxonomy": _token(job.get("taxonomy", "clinical")), "stages": [plan["stage"]], "task_type": task,
              "status": "not_started" if not directory.exists() else "unfinished", "completed": False,
              "metrics": None, "diagnostics": None, "num_recording_groups": None,
              "pretraining": {"checkpoint_sha256": _sha(job.get("checkpoint_sha256")), "datasets": [], "dataset_declaration_available": False},
              "frames": frames, "class_weights": declared_settings.get("class_weights", "none"),
              "backbone_training": training, "probe": probe,
              "purposes": sorted({_token(value) for value in job.get("purposes", [job.get("purpose", "unspecified")])}),
              "_declaration": {key: job.get(key) for key in ("backbone", "mode", "taxonomy", "task_type", "overrides", "train_configuration_id", "probe")}}
    if not (directory / "result.json").is_file():
        return record
    result = _load(directory / "result.json")
    record["status"] = result.get("status") if result.get("status") in ("completed", "early_stopped", "interrupted", "failed", "evaluation_complete") else "invalid_status"
    completed = result.get("training_completed") is True if task == "train" else result.get("status") == "evaluation_complete"
    if not completed:
        return record
    if task == "evaluate":
        trained_directory = Path(job["trained_run_output"])
        config = _load(trained_directory / "config.json")
        trained_result = _load(trained_directory / "result.json")
        if trained_result.get("training_completed") is not True or trained_result.get("status") not in ("completed", "early_stopped"):
            raise ValueError("probe evaluation cannot promote an incomplete training run")
        report = _load(directory / "report.json")
    else:
        config = _load(directory / "config.json")
        report = _load(directory / "val_report.json")
        if record["status"] not in ("completed", "early_stopped") or result.get("protocol_sha256") != plan["protocol_sha256"]:
            raise ValueError("completed training result has inconsistent status or protocol")
    if config.get("protocol_sha256") != plan["protocol_sha256"] or report.get("metadata", {}).get("protocol_sha256") != plan["protocol_sha256"]:
        raise ValueError("plan/configuration/report frozen protocols disagree")
    if config.get("backbone") != backbone or config.get("mode") != mode or config.get("seed") != seed:
        raise ValueError("actual run does not match the declared configuration and seed")
    if config.get("code_hash") != plan.get("code_hash"):
        raise ValueError("actual run implementation differs from its frozen plan")
    metadata = _load(directory / "prediction_metadata.json") if (directory / "prediction_metadata.json").is_file() else dict(report.get("metadata", {}))
    if metadata.get("split") not in ("val", "validation") or report.get("metadata", {}).get("split") not in ("val", "validation"):
        raise ValueError("development tables require explicit validation reports; test results must remain separate")
    if metadata.get("protocol_sha256") != plan["protocol_sha256"]:
        raise ValueError("prediction sidecar protocol differs from its declaration")
    actual_input = _input_settings(config)
    if actual_input["frames"] != record["frames"] or config.get("class_weights", "none") != record["class_weights"] or config.get("backbone_training", "finetune") != training:
        raise ValueError("actual input/weight settings differ from the declared configuration")
    if task == "train":
        for key in (*TRAINING_SETTINGS, *OBJECTIVE_SETTINGS):
            if key in declared_settings and config.get(key) != declared_settings[key]:
                raise ValueError("actual training settings differ from the frozen plan")
        if job.get("main_decoder") is not None and config.get("main_decoder") != job["main_decoder"]:
            raise ValueError("actual prediction decoder differs from the frozen plan")
    pretrained = _pretraining(config, _load((trained_directory if task == "evaluate" else directory) / "load_report.json")
                              if ((trained_directory if task == "evaluate" else directory) / "load_report.json").is_file() else {})
    if task == "train" and job.get("checkpoint_sha256") is not None and pretrained["checkpoint_sha256"] != _sha(job["checkpoint_sha256"]):
        raise ValueError("actual pretrained weights differ from the frozen plan")
    main = _safe_block(report.get("methods", {}).get("main"), validate_confusion=True)
    five = _safe_block(report.get("ambiguous_five", {}).get("methods", {}).get("main"), validate_confusion=True, classes=(1, 2, 3, 4, 5))
    if main is None or main["macro_f1"] is None or five is None or five["macro_f1"] is None:
        raise ValueError("completed run lacks primary seven/five-class metrics")
    if not main.get("confusion_matrix") or five.get("confusion_matrix") != main["confusion_matrix"]:
        raise ValueError("primary seven/five diagnostics require the same full seven-class confusion matrix")
    decoder = config.get("main_decoder", "soft" if mode == "hierarchy" else "flat")
    declared_primary = _safe_block(report.get("methods", {}).get(decoder), validate_confusion=True)
    if declared_primary is None or declared_primary.get("confusion_matrix") != main.get("confusion_matrix"):
        raise ValueError("primary aggregate prediction differs from the frozen main decoder")
    if main.get("confusion_matrix"):
        expected_five = _metrics_from_confusion(main["confusion_matrix"], (1, 2, 3, 4, 5))["macro_f1"]
        if not math.isclose(five["macro_f1"], expected_five, abs_tol=1e-7):
            raise ValueError("primary five-class F1 must preserve full seven-class false positives")
    group_count = report.get("num_recording_groups")
    if not isinstance(group_count, int) or isinstance(group_count, bool) or group_count <= 0:
        raise ValueError("completed reports require recording-group counts")
    definition = config.get("taxonomy_definition", {})
    groups = definition.get("groups")
    Taxonomy.from_dict(definition)
    if record["taxonomy"] == "visual_train_only":
        declared_taxonomy_sha = _sha(job.get("taxonomy_file_sha256"), required=True)
        actual_taxonomy = config.get("taxonomy_file_metadata", {})
        if config.get("taxonomy_file_sha256") != declared_taxonomy_sha or definition.get("name") != actual_taxonomy.get("name") or groups != actual_taxonomy.get("groups"):
            raise ValueError("training-derived taxonomy differs from its frozen declaration")
        provenance = config.get("taxonomy_file_metadata", {}).get("provenance", {})
        if provenance.get("split") != "train":
            raise ValueError("visual taxonomy must have training-only provenance")
    elif groups != get_taxonomy(record["taxonomy"]).to_dict()["groups"] or definition.get("name") != record["taxonomy"]:
        raise ValueError("actual taxonomy does not match its declared clinical/random control")
    if report.get("metadata", {}).get("taxonomy_groups") != groups:
        raise ValueError("trained and reported taxonomies disagree")
    if metadata.get("mode") != mode or metadata.get("taxonomy", {}).get("groups") != groups:
        raise ValueError("private prediction sidecar classifier/taxonomy differs from its aggregate report")
    expected_heads = {"flat": ["flat"], "capacity_control": ["flat", "capacity"], "aux_flat": ["flat", "group", "conditional"], "hierarchy": ["group", "conditional"]}[mode]
    if metadata.get("trained_heads") != expected_heads or report.get("metadata", {}).get("trained_heads") != expected_heads:
        raise ValueError("reported diagnostics must correspond to the classifier's trained heads")
    if not isinstance(metadata.get("eval_metadata"), Mapping) or metadata["eval_metadata"].get("probe") != probe:
        raise ValueError("evaluation treatment metadata differs from its frozen declaration")
    if task == "evaluate" and job.get("checkpoint_sha256") is not None and _sha(metadata.get("checkpoint_sha256"), required=True) != _sha(job["checkpoint_sha256"]):
        raise ValueError("probe uses a different trained checkpoint from its frozen plan")
    for key in ("best_epoch", "parameters_trainable_finetune", "peak_cuda_allocated_bytes"):
        value = config.get(key) if key == "parameters_trainable_finetune" else result.get(key)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise ValueError("aggregate runtime/parameter counts must be nonnegative integers")
    record.update({"completed": True, "metrics": {"seven": main, "five": five}, "diagnostics": _safe_diagnostics(report),
                   "num_recording_groups": group_count, "pretraining": pretrained, "input_settings": actual_input,
                   "best_epoch": result.get("best_epoch", metadata.get("checkpoint_epoch")),
                   "seconds": _finite(result.get("seconds")), "parameters_trainable_finetune": config.get("parameters_trainable_finetune"),
                   "peak_cuda_allocated_bytes": result.get("peak_cuda_allocated_bytes"),
                   "taxonomy_groups": groups, "code_hash": _sha(config.get("code_hash"), required=True),
                   "training_settings": {key: config.get(key) for key in TRAINING_SETTINGS},
                   "objective_settings": {key: config.get(key) for key in OBJECTIVE_SETTINGS},
                   "_eval_metadata": metadata.get("eval_metadata"), "_checkpoint_sha256": _sha(metadata.get("checkpoint_sha256")),
                   "_selection_exposure": metadata.get("selection_exposure"),
                   "_configuration": config})
    return record


def _paired_guard(baseline: Mapping[str, Any], candidate: Mapping[str, Any], contrast: Mapping[str, Any]) -> None:
    if baseline["backbone"] != candidate["backbone"]:
        raise ValueError("paired controlled contrasts cannot mix backbones; cross-backbone tables are descriptive")
    if baseline.get("code_hash") != candidate.get("code_hash"):
        raise ValueError("paired contrasts cannot mix implementations")
    if baseline.get("_selection_exposure") != candidate.get("_selection_exposure"):
        raise ValueError("paired contrasts cannot mix checkpoint-selection exposure")
    if not baseline["pretraining"]["checkpoint_sha256"] or baseline["pretraining"]["checkpoint_sha256"] != candidate["pretraining"]["checkpoint_sha256"]:
        raise ValueError("paired contrasts require identical known pretrained weights")
    allowed = contrast.get("varying_factors", [])
    kind = contrast.get("comparison_kind", "paired_training")
    if kind == "single_factor":
        if len(allowed) != 1 or allowed[0] not in ("frames", "class_weights", "backbone_training", "aux_weight", "group_weight", "conditional_weight"):
            raise ValueError("sensitivity contrasts must declare exactly one supported changing factor")
        if baseline["mode"] != candidate["mode"] or baseline["taxonomy_groups"] != candidate["taxonomy_groups"]:
            raise ValueError("single-factor sensitivity cannot also change mode or taxonomy")
    elif kind == "same_checkpoint_probe":
        if allowed != ["probe"] or baseline.get("probe") != "none" or candidate.get("probe") not in ("static", "shuffle"):
            raise ValueError("probe contrasts must declare only none-to-static/shuffle treatment")
        if baseline["task_type"] != "evaluate" or candidate["task_type"] != "evaluate" or baseline["mode"] != candidate["mode"] or baseline["taxonomy_groups"] != candidate["taxonomy_groups"]:
            raise ValueError("probe contrasts cannot change trained mode/taxonomy or introduce training")
    elif kind != "paired_training" or allowed:
        raise ValueError("only declared single-factor sensitivity may change input/training settings")
    input_left, input_right = dict(baseline["input_settings"]), dict(candidate["input_settings"])
    if "frames" in allowed:
        input_left.pop("frames", None)
        input_right.pop("frames", None)
    if input_left != input_right:
        raise ValueError("paired contrast mixes undeclared input resolution, normalization, or frame settings")
    for key in TRAINING_SETTINGS:
        if key not in allowed and baseline["training_settings"].get(key) != candidate["training_settings"].get(key):
            raise ValueError("paired contrast mixes undeclared training settings")
    for key in ("aux_weight", "group_weight", "conditional_weight"):
        if key not in allowed and baseline["_configuration"].get(key) != candidate["_configuration"].get(key):
            raise ValueError("paired contrast mixes undeclared objective weighting")
    if baseline["metrics"]["seven"].get("num_samples") != candidate["metrics"]["seven"].get("num_samples") or baseline["num_recording_groups"] != candidate["num_recording_groups"]:
        raise ValueError("paired aggregate cohorts have different sample or recording-group counts")
    eval_left, eval_right = baseline.get("_eval_metadata"), candidate.get("_eval_metadata")
    if kind == "same_checkpoint_probe":
        if not baseline.get("_checkpoint_sha256") or baseline["_checkpoint_sha256"] != candidate.get("_checkpoint_sha256"):
            raise ValueError("robustness probes must use the identical known trained checkpoint")
        if not isinstance(eval_left, Mapping) or not isinstance(eval_right, Mapping):
            raise ValueError("probe contrasts require explicit evaluation metadata")
        eval_left, eval_right = dict(eval_left), dict(eval_right)
        eval_left.pop("probe", None)
        eval_right.pop("probe", None)
        if eval_left != eval_right:
            raise ValueError("probe contrast changes more than the declared probe")
    elif "frames" in allowed:
        if baseline["backbone"] not in ("r2plus1d_18", "r3d_18") or baseline["frames"] != 16 or candidate["frames"] != 32:
            raise ValueError("frame sensitivity requires the declared 16-to-32 convolutional-backbone contrast")
        if not isinstance(eval_left, Mapping) or not isinstance(eval_right, Mapping):
            raise ValueError("frame sensitivity requires explicit evaluation treatment metadata")
        eval_left, eval_right = dict(eval_left), dict(eval_right)
        if eval_left.get("probe") != "none" or eval_right.get("probe") != "none":
            raise ValueError("frame sensitivity cannot also change the probe")
        preprocess_left, preprocess_right = dict(eval_left.pop("preprocessing", {})), dict(eval_right.pop("preprocessing", {}))
        if preprocess_left.pop("frames", None) != 16 or preprocess_right.pop("frames", None) != 32 or preprocess_left != preprocess_right:
            raise ValueError("frame sensitivity changed undeclared spatial preprocessing")
        eval_left.pop("repeated_frame_fraction_basis", None)
        eval_right.pop("repeated_frame_fraction_basis", None)
        if eval_left != eval_right:
            raise ValueError("frame sensitivity changed undeclared evaluation treatment")
    elif eval_left != eval_right:
        raise ValueError("paired contrast mixes undeclared evaluation treatment")


def _public_record(record: Mapping[str, Any]) -> dict[str, Any]:
    allowed = ("configuration_id", "seed", "backbone", "mode", "taxonomy", "stages", "task_type", "status", "completed",
               "metrics", "diagnostics", "num_recording_groups", "pretraining", "frames", "class_weights", "backbone_training", "probe", "purposes", "input_settings",
               "best_epoch", "seconds", "parameters_trainable_finetune", "peak_cuda_allocated_bytes", "taxonomy_groups", "code_hash")
    return {key: record[key] for key in allowed if key in record}


def _review_coverage(rows: list[dict[str, Any]], contrasts: list[dict[str, Any]]) -> dict[str, Any]:
    def cover(selected: list[dict[str, Any]], requirement: str, *, optional: bool = False,
              required: Sequence[str] = (), dimension: str | None = None) -> dict[str, Any]:
        missing = sorted(set(required) - {row[dimension] for row in selected}) if dimension else []
        complete = bool(selected) and all(row["all_declared_seeds_completed"] for row in selected)
        status = "not_declared_optional" if optional and not selected else "not_declared" if not selected else "declared_incomplete" if not complete else "partial_requirement_coverage" if missing else "available_development_aggregates"
        return {"requirement": requirement, "declared_configurations": len(selected),
                "completed_configurations": sum(row["all_declared_seeds_completed"] for row in selected),
                "status": status, "missing_required_controls": missing,
                "formal_three_seed_replication_completed": complete and all(row["completed_seeds"] == [42, 2026, 2027] for row in selected),
                "configuration_ids": [row["configuration_id"] for row in selected]}
    native = [row for row in rows if row["task_type"] == "train" and row["frames"] == 16 and row["class_weights"] == "none" and row["backbone_training"] == "finetune"]
    hierarchy = [row for row in rows if row["mode"] == "hierarchy" and row["task_type"] == "train"]
    hierarchy_diagnostics = cover(hierarchy, "Same trained hierarchy logits: seven/five hard/soft/oracle and real-router/conditional-fine diagnostics.")
    missing_diagnostics = [row["configuration_id"] for row in hierarchy if any(not run.get("diagnostics", {}).get("routing", {}).get("actual_router") or not run.get("diagnostics", {}).get("same_checkpoint_hard_soft", {}).get("hard") or not run.get("diagnostics", {}).get("same_checkpoint_hard_soft", {}).get("soft") for run in row["runs"] if run["completed"])]
    hierarchy_diagnostics["missing_completed_diagnostics"] = missing_diagnostics
    if missing_diagnostics:
        hierarchy_diagnostics["status"] = "missing_completed_aggregate_diagnostics"
    return {"modern_backbone_comparison": cover([row for row in native if row["mode"] == "flat" and row["taxonomy"] == "clinical"], "Four native-pretraining flat baselines; cross-backbone scores are descriptive, not equal-compute comparisons.", required=BACKBONES[:4], dimension="backbone"),
            "capacity_control": cover([row for row in native if row["mode"] == "capacity_control"], "Active parameter-matched generic capacity control."),
            "auxiliary_supervision": cover([row for row in native if row["mode"] == "aux_flat"], "Group/fine supervision while retaining flat predictions."),
            "group_semantics": cover([row for row in native if row["taxonomy"].startswith("random_")], "Three fixed matched-size/matched-singleton random taxonomies; report every control.", required=("random_17", "random_29", "random_43"), dimension="taxonomy"),
            "visual_train_only_taxonomy": cover([row for row in native if row["taxonomy"] == "visual_train_only"], "Optional training-only derived taxonomy; evaluation labels never choose groups.", optional=True),
            "same_checkpoint_hard_soft_and_routing": hierarchy_diagnostics,
            "one_factor_sensitivity": cover([row for row in rows if row["frames"] == 32 or row["class_weights"] == "sqrt_inverse"], "One-factor input-length/class-weight controls; do not mix both changes."),
            "representation_adaptation": cover([row for row in rows if row["task_type"] == "train" and row["backbone_training"] == "frozen"], "Frozen official pretrained representation versus adaptation; separate from previous FSN stagewise training."),
            "individual_loss_components": {"status": "available_development_comparisons" if any(set(row["varying_factors"]) & {"aux_weight", "group_weight", "conditional_weight"} and row["status"] == "complete_development_comparison" for row in contrasts) else "not_declared", "reason": "Mode comparisons do not isolate each auxiliary loss coefficient."},
            "robustness_probes": cover([row for row in rows if row["task_type"] == "evaluate"], "Explicit fixed-checkpoint none/static/shuffle evaluations, never extra training.", required=("none", "static", "shuffle"), dimension="probe"),
            "source_duration_repeat_slices": {"status": "available_development_aggregates" if any(run.get("diagnostics", {}).get("slices") for row in rows for run in row["runs"] if run.get("diagnostics")) else "missing_completed_aggregate_diagnostics"},
            "paired_cluster_uncertainty": {"status": "requires_separate_private_paired_analysis", "reason": "Aggregate confusion matrices cannot recover paired clip identities or recording-group bootstrap. Use cvm.analysis compare; this report does not invent confidence intervals."},
            "independent_test_and_patient_generalization": {"status": "not_established_by_validation_tables", "reason": "No test inference or patient-independence claim is generated by collecting validation reports."},
            "novelty_and_superiority": {"status": "not_inferred", "reason": "Implementation and comparison coverage do not establish novelty, significant improvement, or conference acceptance."},
            "paired_contrasts_declared": len(contrasts)}


def collect_plans(plan_paths: Sequence[Path]) -> dict[str, Any]:
    """Read declared jobs and existing SAFE aggregates; never run or fit models."""
    if not plan_paths:
        raise ValueError("at least one frozen run plan is required")
    records = {}
    directories = {}
    contrasts: dict[str, dict[str, Any]] = {}
    protocol_shas = set()
    plan_digests = []
    for plan_path in plan_paths:
        plan = _load(Path(plan_path))
        if plan.get("schema_version") != "fsn-cvm-plan-1" or plan.get("stage") not in STAGES:
            raise ValueError("unsupported frozen plan schema or stage")
        protocol_shas.add(_sha(plan.get("protocol_sha256"), required=True))
        plan_digests.append(_digest(Path(plan_path)))
        for job in plan.get("jobs", []):
            record = _read_plan_record(plan, job)
            key = (record["configuration_id"], record["seed"])
            if key in records:
                if Path(job["output"]).resolve() != directories[key]:
                    raise ValueError("duplicate configuration/seed declarations must explicitly reference the same run; favorable rerun selection is forbidden")
                if records[key]["_declaration"] != record["_declaration"]:
                    raise ValueError("one configuration/seed identifier has conflicting frozen declarations")
                records[key]["stages"] = sorted(set(records[key]["stages"] + record["stages"]))
                records[key]["purposes"] = sorted(set(records[key]["purposes"] + record["purposes"]))
            else:
                records[key], directories[key] = record, Path(job["output"]).resolve()
        for raw in plan.get("contrasts", []):
            contrast = {key: raw.get(key) for key in ("contrast_id", "baseline_configuration_id", "candidate_configuration_id", "comparison_kind", "varying_factors")}
            for key in ("contrast_id", "baseline_configuration_id", "candidate_configuration_id"):
                contrast[key] = _token(contrast[key])
            contrast["varying_factors"] = contrast["varying_factors"] or []
            contrast["comparison_kind"] = contrast["comparison_kind"] or "paired_training"
            if contrast["contrast_id"] in contrasts and contrasts[contrast["contrast_id"]] != contrast:
                raise ValueError("duplicate declared contrast identifiers have inconsistent definitions")
            contrasts[contrast["contrast_id"]] = contrast
    if len(protocol_shas) != 1 or not records:
        raise ValueError("all plans must declare jobs on one frozen protocol")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records.values():
        grouped[record["configuration_id"]].append(record)
    rows = []
    for identifier, runs in sorted(grouped.items()):
        completed = [run for run in runs if run["completed"]]
        if completed:
            fingerprints = {(run["code_hash"], run["pretraining"]["checkpoint_sha256"], json.dumps(run["input_settings"], sort_keys=True), json.dumps(run["training_settings"], sort_keys=True), json.dumps(run["objective_settings"], sort_keys=True), json.dumps(run["taxonomy_groups"])) for run in completed}
            if len(fingerprints) != 1:
                raise ValueError("seed pooling mixes implementation, pretraining, input, training settings, or taxonomy")
        first = runs[0]
        if any(run["_declaration"] != first["_declaration"] for run in runs):
            raise ValueError("seed replication must retain the same frozen configuration declaration")
        summary = {family: {metric: _stat([run["metrics"][family][metric] for run in completed])
                            for metric in METRICS if all(run["metrics"][family].get(metric) is not None for run in completed)}
                   for family in ("seven", "five")} if completed else None
        row = {"configuration_id": identifier, "backbone": first["backbone"], "mode": first["mode"], "taxonomy": first["taxonomy"],
               "task_type": first["task_type"], "frames": first["frames"], "class_weights": first["class_weights"],
               "backbone_training": first["backbone_training"], "probe": first["probe"],
               "purposes": sorted({purpose for run in runs for purpose in run["purposes"]}),
               "stages": sorted({stage for run in runs for stage in run["stages"]}), "declared_seeds": sorted(run["seed"] for run in runs),
               "completed_seeds": sorted(run["seed"] for run in completed), "missing_or_incomplete_seeds": sorted(run["seed"] for run in runs if not run["completed"]),
               "all_declared_seeds_completed": len(completed) == len(runs), "seed_summary": summary,
               "runs": [_public_record(run) for run in sorted(runs, key=lambda r: r["seed"])]}
        rows.append(row)
    comparison_rows = []
    for identifier, contrast in sorted(contrasts.items()):
        base_id, candidate_id = contrast["baseline_configuration_id"], contrast["candidate_configuration_id"]
        if base_id not in grouped or candidate_id not in grouped:
            raise ValueError("declared contrast references a configuration outside its plans")
        left = {run["seed"]: run for run in grouped[base_id]}
        right = {run["seed"]: run for run in grouped[candidate_id]}
        if left.keys() != right.keys():
            raise ValueError("declared paired contrast must match the complete planned seed set")
        paired = []
        for seed in sorted(left):
            if not left[seed]["completed"] or not right[seed]["completed"]:
                continue
            _paired_guard(left[seed], right[seed], contrast)
            paired.append({"seed": seed,
                           "seven_macro_f1_delta": right[seed]["metrics"]["seven"]["macro_f1"] - left[seed]["metrics"]["seven"]["macro_f1"],
                           "five_macro_f1_delta": right[seed]["metrics"]["five"]["macro_f1"] - left[seed]["metrics"]["five"]["macro_f1"],
                           "accuracy_delta": right[seed]["metrics"]["seven"]["accuracy"] - left[seed]["metrics"]["seven"]["accuracy"]})
        comparison_rows.append({**contrast, "declared_seeds": sorted(left), "completed_paired_seeds": [row["seed"] for row in paired],
                                "status": "complete_development_comparison" if len(paired) == len(left) else "declared_incomplete",
                                "paired_seed_deltas": paired,
                                "paired_seed_summary": {metric: _stat([row[metric] for row in paired]) for metric in ("seven_macro_f1_delta", "five_macro_f1_delta", "accuracy_delta")} if paired else None,
                                "cluster_bootstrap_interval": None, "significance_claim": False})
    return {"schema_version": "fsn-cvm-planned-experiment-report-1", "protocol_sha256": next(iter(protocol_shas)),
            "plan_sha256": plan_digests, "split": "validation", "experiment_rows": rows,
            "stage_tables": {stage: [row["configuration_id"] for row in rows if stage in row["stages"]] for stage in STAGES if any(stage in row["stages"] for row in rows)},
            "comparison_rows": comparison_rows, "review_evidence": _review_coverage(rows, comparison_rows),
            "all_declared_jobs_completed": all(record["completed"] for record in records.values()), "formal_claim_ready": False,
            "cross_backbone_disclosure": {"equal_compute_claim": False, "matched_pretraining_claim": False,
                                           "reason": "Native official checkpoints can differ in pretraining datasets, resolution, normalization, and model size. Only within-backbone controlled contrasts enforce identical pretrained hashes and undeclared input/training settings."},
            "interpretation": "Validation chose checkpoints. Seed summaries are descriptive; missing/incomplete jobs are shown without fabricated scores. Fixed-checkpoint probes are diagnostics, not new training or an independent test.",
            "privacy": {"aggregate_only": True, "raw_paths": False, "clip_ids": False, "recording_ids": False},
            "training_or_inference_started": False}


def render_markdown(report: Mapping[str, Any]) -> str:
    def number(value: float | None) -> str:
        return "unavailable" if value is None else f"{value:.4f}"

    def score(row: Mapping[str, Any], family: str, metric: str = "macro_f1") -> str:
        summary = row.get("seed_summary")
        if not summary:
            return "pending"
        item = summary[family].get(metric)
        if not item:
            return "unavailable"
        return f"{item['mean']:.4f}" + (f" ± {item['sd']:.4f}" if item["sd"] is not None else "")

    text = ["Development experiment tables; no independent-test, equal-compute, or significance claim.",
            "Scores use 0–1 units. ± is sample SD across training seeds, not a confidence interval.", "",
            "| Declared configuration | Stage | Completed seeds | Seven Macro-F1 | Five Macro-F1 | Weighted-F1 | Accuracy | Status |",
            "|---|---|---|---:|---:|---:|---:|---|"]
    for row in report.get("experiment_rows", []):
        text.append(f"| {row['configuration_id']} | {', '.join(row['stages'])} | {len(row['completed_seeds'])}/{len(row['declared_seeds'])} | {score(row, 'seven')} | {score(row, 'five')} | {score(row, 'seven', 'weighted_f1')} | {score(row, 'seven', 'accuracy')} | {'complete' if row['all_declared_seeds_completed'] else 'incomplete'} |")
    text += ["", "| Declared contrast | Completed pairs | Seven Macro-F1 delta (pp) | Five Macro-F1 delta (pp) |", "|---|---:|---:|---:|"]
    for row in report.get("comparison_rows", []):
        summary = row.get("paired_seed_summary")
        seven = f"{100 * summary['seven_macro_f1_delta']['mean']:+.2f}" if summary else "pending"
        five = f"{100 * summary['five_macro_f1_delta']['mean']:+.2f}" if summary else "pending"
        text.append(f"| {row['contrast_id']} | {len(row['completed_paired_seeds'])}/{len(row['declared_seeds'])} | {seven} | {five} |")
    text += ["", "| Native input/pretraining configuration | Dataset declaration | Crop | Frames | Backbone training | Pretrained SHA-256 |", "|---|---|---|---:|---|---|"]
    for row in report.get("experiment_rows", []):
        complete = next((run for run in row["runs"] if run["completed"]), None)
        pretraining = complete["pretraining"] if complete else row["runs"][0]["pretraining"]
        crop = complete.get("input_settings", {}).get("crop_size") if complete else None
        text.append(f"| {row['configuration_id']} | {', '.join(pretraining['datasets']) or 'unknown'} | {str(crop) if crop is not None else 'unknown'} | {row['frames']} | {row['backbone_training']} | {pretraining['checkpoint_sha256'] or 'unknown'} |")
    text += ["", "| Configuration / seed | Seven hard | Seven soft | Five hard | Five soft | Soft−hard seven (pp) | Real router Macro-F1 | Conditional fine recall |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report.get("experiment_rows", []):
        for run in row["runs"]:
            diagnostics = run.get("diagnostics")
            if not diagnostics:
                continue
            hard_soft = diagnostics["same_checkpoint_hard_soft"]
            if not hard_soft["hard"] or not hard_soft["soft"]:
                continue
            actual = diagnostics["routing"]["actual_router"] or {}
            get = lambda method: number((hard_soft.get(method) or {}).get("macro_f1"))
            delta = hard_soft["soft_minus_hard_macro_f1"]
            text.append(f"| {row['configuration_id']} / {run['seed']} | {get('hard')} | {get('soft')} | {get('hard_five')} | {get('soft_five')} | {100 * delta:+.2f} | {number(actual.get('macro_f1'))} | {number(actual.get('conditional_fine_recall_given_correct_route'))} |")
    text += ["", "| Configuration / seed | Hashed source / duration / repeat slice | Clips | Recording groups | Seven Macro-F1 |", "|---|---|---:|---:|---:|"]
    for row in report.get("experiment_rows", []):
        for run in row["runs"]:
            for alias, block in (run.get("diagnostics") or {}).get("slices", {}).items():
                value = "suppressed" if block["suppressed"] else number((block["main"] or {}).get("macro_f1"))
                text.append(f"| {row['configuration_id']} / {run['seed']} | {alias} | {block['num_samples']} | {block['num_recording_groups']} | {value} |")
    text += ["", "| Reviewer evidence | Coverage status | Missing required controls |", "|---|---|---|"]
    for name, block in report.get("review_evidence", {}).items():
        if isinstance(block, Mapping):
            text.append(f"| {name} | {block.get('status', 'unavailable')} | {', '.join(block.get('missing_required_controls', [])) or '—'} |")
    text += ["", "Native pretraining and spatial/input settings differ across backbones. Within-backbone contrasts reject undeclared mixing.",
             "Primary five-class Macro-F1 averages five classes from the full seven-class confusion matrix; singleton-to-five false positives remain.",
             "True-group oracles are explanatory bounds, not deployable performance. JSON retains per-class/group and route-error diagnostics.",
             "Paired recording-group bootstrap requires private matched prediction files through cvm.analysis; no intervals are invented here.", ""]
    return "\n".join(text)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--runs", type=Path, nargs="+")
    inputs.add_argument("--plans", "--plan", dest="plans", type=Path, nargs="+")
    parser.add_argument("--expected-seeds", default="42,2026,2027")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = collect_plans(args.plans) if args.plans else collect_runs(args.runs, [int(seed) for seed in args.expected_seeds.split(",")])
        if args.markdown_output and not args.plans:
            raise ValueError("declared table Markdown requires --plans")
        outputs = [args.output] + ([args.markdown_output] if args.markdown_output else [])
        if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
            raise ValueError("report outputs must be distinct new files; overwriting is prohibited")
        for path in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
        if args.markdown_output:
            with args.markdown_output.open("x", encoding="utf-8") as handle:
                handle.write(render_markdown(report))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        message = str(exc) if isinstance(exc, ValueError) else "experiment collection failed validation or output writing"
        parser.exit(2, "error: " + message + "\n")


if __name__ == "__main__":
    main()
