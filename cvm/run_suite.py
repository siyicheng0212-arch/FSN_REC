"""Generate a frozen CVM run plan; launching always requires --execute.

Four GPUs run independent jobs. There is no DDP, automatic promotion, resume,
extra seed wave, or test evaluation. A plan and launch lock are never replaced.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from cvm.train import ROOT, atomic_json, sha256_file, source_digest, source_hashes

FORMAL_SEEDS = (42, 2026, 2027)
CORE_BACKBONES = ("r2plus1d_18", "mvit_v2_s", "videomamba_tiny16", "videomae_base16")
STAGES = ("smoke", "pilot", "benchmark", "ablation", "sensitivity", "representation", "formal", "extended", "robustness")


def configuration_id(configuration: dict[str, Any]) -> str:
    settings = configuration.get("overrides", {})
    return (f"{configuration['backbone']}__{configuration['mode']}__{configuration['taxonomy']}"
            f"__f{settings.get('frames', 16)}__w{settings.get('class_weights', 'none')}"
            f"__bt{settings.get('backbone_training', 'finetune')}")


def declared_configurations(stage: str, backbones: tuple[str, ...] | None = None,
                            visual_taxonomy: str | None = None) -> list[dict[str, Any]]:
    if stage not in STAGES or stage == "robustness":
        raise ValueError("robustness configurations derive from an explicit completed training plan")
    selected = backbones or (("r2plus1d_18",) if stage in ("pilot", "smoke", "ablation", "sensitivity", "representation") else CORE_BACKBONES)
    if any(backbone not in CORE_BACKBONES for backbone in selected) or len(set(selected)) != len(selected):
        raise ValueError("backbones must be unique declared core architectures")
    if stage in ("ablation", "sensitivity", "representation") and selected != ("r2plus1d_18",):
        raise ValueError("ablation/sensitivity/representation are declared on R2Plus1D only")
    configurations: dict[str, dict[str, Any]] = {}

    def add(backbone: str, mode: str, taxonomy: str, purpose: str,
            frames: int = 16, class_weights: str = "none", taxonomy_file: str | None = None,
            backbone_training: str = "finetune") -> None:
        config = {"backbone": backbone, "mode": mode, "taxonomy": taxonomy,
                  "overrides": {"frames": frames, "class_weights": class_weights, "backbone_training": backbone_training},
                  "purpose": purpose, "purposes": [purpose], "task_type": "train"}
        if taxonomy_file:
            config["taxonomy_file"] = taxonomy_file
        key = configuration_id(config)
        if key in configurations:
            configurations[key]["purposes"] = sorted(set(configurations[key]["purposes"] + [purpose]))
        else:
            config["configuration_id"] = key
            configurations[key] = config

    if stage in ("smoke", "pilot"):
        for backbone in selected:
            for mode in ("flat", "capacity_control", "aux_flat", "hierarchy"):
                add(backbone, mode, "clinical", "runtime_smoke" if stage == "smoke" else "seed42_feasibility")
    if stage in ("benchmark", "formal", "extended"):
        for backbone in selected:
            for mode in ("flat", "hierarchy"):
                add(backbone, mode, "clinical", "modern_backbone_comparison")
    if stage in ("ablation", "formal", "extended") and "r2plus1d_18" in selected:
        for mode in ("flat", "capacity_control", "aux_flat", "hierarchy"):
            add("r2plus1d_18", mode, "clinical", "capacity_and_group_supervision_ablation")
        for taxonomy in ("random_17", "random_29", "random_43"):
            add("r2plus1d_18", "hierarchy", taxonomy, "matched_size_matched_singleton_taxonomy_control")
    if stage in ("sensitivity", "extended") and "r2plus1d_18" in selected:
        for mode in ("flat", "hierarchy"):
            for frames, weighting in ((16, "none"), (32, "none"), (16, "sqrt_inverse")):
                add("r2plus1d_18", mode, "clinical", "one_factor_at_a_time_sensitivity", frames, weighting)
    if stage in ("representation", "extended") and "r2plus1d_18" in selected:
        for mode in ("flat", "hierarchy"):
            for training in ("finetune", "frozen"):
                add("r2plus1d_18", mode, "clinical", "official_pretrained_representation_adaptation_control", backbone_training=training)
    if visual_taxonomy:
        if "r2plus1d_18" not in selected or stage not in ("ablation", "formal", "extended"):
            raise ValueError("visual taxonomy is an explicit optional R2 ablation/formal configuration")
        add("r2plus1d_18", "hierarchy", "visual_train_only", "training_features_derived_taxonomy_control", taxonomy_file=visual_taxonomy)
    return list(configurations.values())


def declared_contrasts(configurations: list[dict[str, Any]]) -> list[dict[str, str]]:
    contrasts = []
    available = {config["configuration_id"]: config for config in configurations}

    def pair(identifier: str, baseline: dict[str, Any], candidate: dict[str, Any], purpose: str,
             varying_factors: tuple[str, ...] = ()) -> None:
        base_id, candidate_id = configuration_id(baseline), configuration_id(candidate)
        if base_id in available and candidate_id in available:
            contrasts.append({"contrast_id": identifier, "baseline_configuration_id": base_id,
                              "candidate_configuration_id": candidate_id, "purpose": purpose,
                              "comparison_kind": "single_factor" if varying_factors else "paired_training",
                              "varying_factors": list(varying_factors)})

    for backbone in CORE_BACKBONES:
        for frames, weighting in ((16, "none"), (32, "none"), (16, "sqrt_inverse")):
            baseline = {"backbone": backbone, "mode": "flat", "taxonomy": "clinical", "overrides": {"frames": frames, "class_weights": weighting}}
            candidate = {**baseline, "mode": "hierarchy"}
            pair(f"hierarchy_vs_flat__{backbone}__f{frames}__w{weighting}", baseline, candidate,
                 "Does learned clinical hierarchy improve seven-class recognition at matched input/training settings?")
    base = {"backbone": "r2plus1d_18", "mode": "flat", "taxonomy": "clinical"}
    capacity, auxiliary, hierarchy = ({**base, "mode": mode} for mode in ("capacity_control", "aux_flat", "hierarchy"))
    pair("capacity_vs_flat__r2", base, capacity, "Effect of extra active generic capacity")
    pair("auxiliary_vs_flat__r2", base, auxiliary, "Clinical group/fine supervision with flat prediction")
    pair("auxiliary_vs_capacity__r2", capacity, auxiliary, "Taxonomy supervision versus parameter-matched generic supervision")
    pair("hierarchy_vs_auxiliary__r2", auxiliary, hierarchy, "Factorized predictions versus auxiliary-supervised flat predictions")
    for taxonomy in ("random_17", "random_29", "random_43", "visual_train_only"):
        pair(f"taxonomy_{taxonomy}_vs_clinical__r2", hierarchy, {**hierarchy, "taxonomy": taxonomy},
             "Taxonomy specificity; matched singleton allocation for the fixed random controls")
    for mode in ("flat", "hierarchy"):
        baseline = {**base, "mode": mode, "overrides": {"frames": 16, "class_weights": "none"}}
        pair(f"frames32_vs16__r2__{mode}", baseline, {**baseline, "overrides": {"frames": 32, "class_weights": "none"}},
             "One-factor temporal input sensitivity", ("frames",))
        pair(f"sqrt_inverse_vs_none__r2__{mode}", baseline, {**baseline, "overrides": {"frames": 16, "class_weights": "sqrt_inverse"}},
             "One-factor class-weight sensitivity; weights derived from training only", ("class_weights",))
        pair(f"frozen_vs_finetune__r2__{mode}", baseline, {**baseline, "overrides": {"frames": 16, "class_weights": "none", "backbone_training": "frozen"}},
             "Frozen official pretrained representation versus adaptation; not an exact reproduction of the old FSN stagewise procedure", ("backbone_training",))
    return contrasts


def generate_plan(args: argparse.Namespace) -> Path:
    from cvm.protocol import load_protocol
    if args.stage == "robustness":
        return generate_robustness_plan(args)
    if any(value is None for value in (args.protocol, args.cache_root, args.weights_json)):
        raise ValueError("training stages require --protocol, --cache-root and --checkpoints")
    if args.class_weights != "none":
        raise ValueError("declared main matrix uses no class weighting; use the sensitivity stage for the fixed sqrt_inverse contrast")
    protocol = load_protocol(args.protocol)
    weights = json.loads(args.weights_json.read_text(encoding="utf-8"))
    if not isinstance(weights, dict):
        raise ValueError("weights JSON must map declared backbone names to official local checkpoint paths")
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if not gpus or len(set(gpus)) != len(gpus) or any(not value.isdigit() for value in gpus):
        raise ValueError("GPU IDs must be unique nonnegative integers")
    selected = tuple("videomamba_tiny16" if item == "videomamba_tiny" else item for item in args.backbones.split(",")) if args.backbones else None
    configurations = declared_configurations(args.stage, selected, str(args.taxonomy_file.resolve()) if args.taxonomy_file else None)
    contrasts = declared_contrasts(configurations)
    for configuration in configurations:
        configuration["contrast_ids"] = [contrast["contrast_id"] for contrast in contrasts
                                         if configuration["configuration_id"] in (contrast["baseline_configuration_id"], contrast["candidate_configuration_id"])]
    seeds = (42,) if args.stage in ("smoke", "pilot") else FORMAL_SEEDS
    settings = {"batch_size": args.batch_size, "accum_steps": args.accum_steps,
                "epochs": 1 if args.stage == "smoke" else args.epochs,
                "warmup_epochs": 0 if args.stage == "smoke" else args.warmup_epochs,
                "patience": args.patience, "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
                "warmup_lr": args.warmup_lr, "weight_decay": args.weight_decay, "class_weights": args.class_weights,
                "workers": args.workers, "optimizer": args.optimizer, "amp": args.amp,
                "clip_grad": args.clip_grad, "augmentation": args.augmentation,
                "aux_weight": 1., "group_weight": 1., "conditional_weight": 1.}
    if settings["batch_size"] <= 0 or settings["accum_steps"] <= 0:
        raise ValueError("batch and accumulation must be positive")
    jobs = []
    for configuration in configurations:
        backbone = configuration["backbone"]
        entry = weights.get(backbone, weights.get("videomamba_tiny") if backbone == "videomamba_tiny16" else None)
        if isinstance(entry, str):
            entry = {"checkpoint": entry}
        if not isinstance(entry, dict) or not isinstance(entry.get("checkpoint"), str):
            raise ValueError(f"weights JSON lacks an explicit local checkpoint for {backbone}")
        checkpoint = Path(entry["checkpoint"]).expanduser().resolve()
        checkpoint_sha = sha256_file(checkpoint) if checkpoint.is_file() else None
        external_root = entry.get("external_repo") or (args.videomamba_root if backbone == "videomamba_tiny16" else args.videomae_root if backbone == "videomae_base16" else None)
        for seed in seeds:
            name = configuration["configuration_id"] + f"__seed{seed}"
            jobs.append({**configuration, "seed": seed, "name": name,
                         "output": str((args.output / "runs" / name).resolve()),
                         "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
                         "videomamba_root": str(Path(external_root).expanduser().resolve()) if external_root and backbone == "videomamba_tiny16" else None,
                         "videomae_root": str(Path(external_root).expanduser().resolve()) if external_root and backbone == "videomae_base16" else None,
                         "taxonomy_file_sha256": sha256_file(Path(configuration["taxonomy_file"])) if configuration.get("taxonomy_file") and Path(configuration["taxonomy_file"]).is_file() else None,
                         "main_decoder": "soft" if configuration["mode"] == "hierarchy" else "flat"})
    plan = {"schema_version": "fsn-cvm-plan-1", "stage": args.stage,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "protocol": str(args.protocol.resolve()), "protocol_sha256": protocol.protocol_sha256,
            "cache_root": str(args.cache_root.resolve()), "source_hashes": source_hashes(), "code_hash": source_digest(),
            "gpus": list(gpus), "settings": settings, "jobs": jobs, "configurations": configurations, "contrasts": contrasts,
            "task_type": "train", "configuration_count": len(configurations), "job_count": len(jobs),
            "fixed_formal_seeds": list(FORMAL_SEEDS), "test_evaluated": False,
            "full_declared_formal_matrix": args.stage in ("formal", "extended") and (selected is None or set(selected) == set(CORE_BACKBONES)),
            "matrix_overlap_policy": "stages share declared baseline configurations; prefer the formal/extended union rather than executing overlapping plans; no automatic artifact reuse/resume",
            "sensitivity_included": args.stage in ("sensitivity", "extended"),
            "representation_included": args.stage in ("representation", "extended"),
            "formal_claim_ready": False, "launch_requires_explicit_execute": True,
            "local_readiness": {"all_checkpoint_files_exist": all(job["checkpoint_sha256"] for job in jobs),
                                "official_checkpoint_load_validation": "required by execution preflight before any jobs",
                                "CUDA_and_dependencies": "not checked by plan generation"}}
    try:
        plan["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        plan["git_commit"] = None
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "run_plan.json"
    atomic_json(path, plan)
    print(json.dumps({"plan": str(path), "jobs": len(jobs), "stage": args.stage, "training_started": False,
                      "formal_claim_ready": False}, ensure_ascii=False), flush=True)
    return path


def generate_robustness_plan(args: argparse.Namespace) -> Path:
    """Explicit fixed-checkpoint validation probes; this cannot start training."""
    from cvm.protocol import load_protocol
    if args.trained_plan is None:
        raise ValueError("robustness requires --trained-plan pointing to a saved training plan")
    training_plan = json.loads(args.trained_plan.read_text(encoding="utf-8"))
    if training_plan.get("schema_version") != "fsn-cvm-plan-1" or training_plan.get("task_type", "train") != "train":
        raise ValueError("robustness requires a training plan, not another evaluation plan")
    protocol_path = args.protocol or Path(training_plan["protocol"])
    protocol = load_protocol(protocol_path)
    if protocol.protocol_sha256 != training_plan["protocol_sha256"]:
        raise ValueError("robustness must use exactly the training plan's frozen evaluation protocol")
    cache_root = args.cache_root or Path(training_plan["cache_root"])
    if cache_root.resolve() != Path(training_plan["cache_root"]).resolve():
        raise ValueError("robustness cannot substitute a different cache root")
    probes = tuple(value.strip() for value in args.probes.split(","))
    if not probes or "none" not in probes or len(set(probes)) != len(probes) or any(probe not in ("none", "static", "shuffle") for probe in probes):
        raise ValueError("probes must be unique none/static/shuffle values and include the unchanged none baseline")
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if not gpus or len(set(gpus)) != len(gpus) or any(not value.isdigit() for value in gpus):
        raise ValueError("GPU IDs must be unique nonnegative integers")
    selected = {"videomamba_tiny16" if value == "videomamba_tiny" else value for value in args.backbones.split(",")} if args.backbones else None
    source_jobs = [job for job in training_plan["jobs"]
                   if (selected is None or job["backbone"] in selected)
                   and (args.include_control_probes or (job["mode"] in ("flat", "hierarchy") and job["taxonomy"] == "clinical"))]
    if not source_jobs:
        raise ValueError("the supplied training plan has no selected probe candidates")
    jobs, configurations, contrasts = [], {}, []
    for source in source_jobs:
        source_id = source.get("configuration_id", configuration_id(source))
        checkpoint = Path(source["output"]) / "best.pt"
        checkpoint_sha = sha256_file(checkpoint) if checkpoint.is_file() else None
        for probe in probes:
            identifier = source_id + "__probe_" + probe
            name = identifier + f"__seed{source['seed']}"
            config = {"configuration_id": identifier, "train_configuration_id": source_id,
                      "backbone": source["backbone"], "mode": source["mode"], "taxonomy": source["taxonomy"],
                      "overrides": source.get("overrides", {}), "task_type": "evaluate", "probe": probe,
                      "purpose": "same_checkpoint_within_clip_temporal_probe", "split": "val"}
            configurations[identifier] = config
            jobs.append({**config, "seed": source["seed"], "name": name,
                         "output": str((args.output / "runs" / name).resolve()),
                         "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha,
                         "trained_run_output": source["output"], "trained_run_name": source["name"]})
        for probe in probes:
            if probe != "none":
                contrast = {"contrast_id": f"probe_{probe}_vs_none__{source_id}",
                            "baseline_configuration_id": source_id + "__probe_none",
                            "candidate_configuration_id": source_id + "__probe_" + probe,
                            "comparison_kind": "same_checkpoint_probe", "varying_factors": ["probe"],
                            "purpose": "Temporal input dependence at fixed checkpoint and selected frame set; not localization or quality-assessment proof"}
                if contrast not in contrasts:
                    contrasts.append(contrast)
    for job in jobs:
        job["contrast_ids"] = [contrast["contrast_id"] for contrast in contrasts
                               if job["configuration_id"] in (contrast["baseline_configuration_id"], contrast["candidate_configuration_id"])]
    plan = {"schema_version": "fsn-cvm-plan-1", "stage": "robustness", "task_type": "evaluate",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "protocol": str(protocol_path.resolve()), "protocol_sha256": protocol.protocol_sha256,
            "cache_root": str(cache_root.resolve()), "source_hashes": source_hashes(), "code_hash": source_digest(),
            "gpus": list(gpus), "settings": {"batch_size": args.batch_size, "workers": args.workers, "amp": args.amp},
            "jobs": jobs, "configurations": list(configurations.values()), "contrasts": contrasts,
            "configuration_count": len(configurations), "job_count": len(jobs),
            "reference_training_plan": str(args.trained_plan.resolve()), "reference_training_plan_sha256": sha256_file(args.trained_plan),
            "training_jobs": 0, "test_evaluated": False, "split": "val", "formal_claim_ready": False,
            "full_declared_formal_matrix": False, "launch_requires_explicit_execute": True,
            "probe_set_complete": set(probes) == {"none", "static", "shuffle"},
            "local_readiness": {"all_checkpoint_files_exist": all(job["checkpoint_sha256"] for job in jobs),
                                "CUDA_and_dependencies": "not checked by plan generation"}}
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "run_plan.json"
    atomic_json(path, plan)
    print(json.dumps({"plan": str(path), "jobs": len(jobs), "stage": "robustness", "training_started": False,
                      "training_jobs": 0, "test_evaluated": False}), flush=True)
    return path


def trainer_command(plan: dict[str, Any], job: dict[str, Any], python: str) -> list[str]:
    if job.get("task_type") == "evaluate":
        command = [python, "-m", "cvm.train", "evaluate", "--protocol", plan["protocol"],
                   "--cache-root", plan["cache_root"], "--output", job["output"],
                   "--checkpoint", job["checkpoint"], "--split", "val", "--probe", job["probe"]]
        for key in ("batch_size", "workers", "amp"):
            command.extend(("--" + key.replace("_", "-"), str(plan["settings"][key])))
        return command
    command = [python, "-m", "cvm.train", "train", "--protocol", plan["protocol"],
               "--cache-root", plan["cache_root"], "--output", job["output"],
               "--backbone", job["backbone"], "--mode", job["mode"],
               "--seed", str(job["seed"]), "--checkpoint", job["checkpoint"],
               "--main-decoder", job["main_decoder"], "--expected-protocol-sha256", plan["protocol_sha256"],
               "--expected-code-hash", plan["code_hash"]]
    if job.get("taxonomy_file"):
        command.extend(("--taxonomy-file", job["taxonomy_file"]))
    else:
        command.extend(("--taxonomy", job["taxonomy"]))
    if job.get("videomamba_root"):
        command.extend(("--videomamba-root", job["videomamba_root"]))
    if job.get("videomae_root"):
        command.extend(("--videomae-root", job["videomae_root"]))
    for key, value in {**plan["settings"], **job.get("overrides", {})}.items():
        command.extend(("--" + key.replace("_", "-"), str(value)))
    return command


def preflight(plan: dict[str, Any]) -> None:
    import torch
    from cvm.protocol import load_protocol
    from cvm.models import build_model
    from cvm.train import CachedProtocolDataset, model_arguments
    if source_digest() != plan["code_hash"]:
        raise RuntimeError("implementation changed after planning; create a new plan")
    protocol = load_protocol(Path(plan["protocol"]))
    if protocol.protocol_sha256 != plan["protocol_sha256"]:
        raise RuntimeError("protocol changed after planning")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no jobs started")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        raise RuntimeError("unset inherited CUDA_VISIBLE_DEVICES; the launcher assigns each job's physical GPU explicitly")
    for identifier in plan["gpus"]:
        index = int(identifier)
        if index >= torch.cuda.device_count():
            raise RuntimeError("a declared GPU is unavailable; no jobs started")
        if plan["settings"]["amp"] == "bfloat16":
            with torch.cuda.device(index):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("a declared GPU lacks bf16 support; no silent fallback")
    for job in plan["jobs"]:
        if Path(job["output"]).exists():
            raise FileExistsError("a planned run already exists; overwriting/resume is prohibited")
        if not job["checkpoint_sha256"] or not Path(job["checkpoint"]).is_file() or sha256_file(Path(job["checkpoint"])) != job["checkpoint_sha256"]:
            raise RuntimeError("a declared checkpoint is absent or changed; create a new ready plan before any jobs")
        if job.get("taxonomy_file") and (not job.get("taxonomy_file_sha256") or sha256_file(Path(job["taxonomy_file"])) != job["taxonomy_file_sha256"]):
            raise RuntimeError("frozen visual taxonomy is missing or changed after planning")
    if plan.get("task_type") == "evaluate":
        if sha256_file(Path(plan["reference_training_plan"])) != plan["reference_training_plan_sha256"]:
            raise RuntimeError("reference training plan changed after probe planning")
        checked = set()
        for job in plan["jobs"]:
            if job["checkpoint"] in checked:
                continue
            result_path = Path(job["trained_run_output"]) / "result.json"
            if not result_path.is_file():
                raise RuntimeError("a referenced training job has not ended; no checkpoint probes started")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not result.get("training_completed") or result.get("status") not in ("completed", "early_stopped"):
                raise RuntimeError("a referenced training job is incomplete/interrupted; the formal probe matrix requires completed training")
            checkpoint = torch.load(job["checkpoint"], map_location="cpu", weights_only=False)
            config = checkpoint["config"]
            if config["protocol_sha256"] != protocol.protocol_sha256 or config["code_hash"] != source_digest():
                raise RuntimeError("probe checkpoint does not match the frozen implementation/protocol")
            model = build_model(**model_arguments(config, pretrained=False))
            model.load_state_dict(checkpoint["model"], strict=True)
            del model, checkpoint
            gc.collect()
            checked.add(job["checkpoint"])
        CachedProtocolDataset(protocol, "val", Path(plan["cache_root"]))
        return
    CachedProtocolDataset(protocol, "train", Path(plan["cache_root"]))
    CachedProtocolDataset(protocol, "val", Path(plan["cache_root"]))
    # Validate ALL formal checkpoints and dependencies before launching any job.
    tested = set()
    for job in plan["jobs"]:
        key = (job["backbone"], job["checkpoint"], job.get("videomamba_root"), job.get("videomae_root"))
        definition = None
        if job.get("taxonomy_file"):
            from cvm.taxonomy import Taxonomy
            definition = json.loads(Path(job["taxonomy_file"]).read_text(encoding="utf-8"))
            if definition.get("protocol_sha256") != protocol.protocol_sha256 or definition.get("training_manifest_sha256") != sha256_file(protocol.manifest_paths["train"]) or definition.get("provenance", {}).get("split") != "train":
                raise RuntimeError("custom visual taxonomy lacks matching train-only provenance")
            Taxonomy.from_dict(definition)
        if key not in tested:
            config = {**job, "allow_download": False}
            if definition is not None:
                config["taxonomy_definition"] = definition
            model = build_model(**model_arguments(config, pretrained=True))
            del model
            gc.collect()
            tested.add(key)


def execute_plan(plan_path: Path, *, python: str, execute: bool) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "fsn-cvm-plan-1":
        raise ValueError("unsupported plan schema")
    if not execute:
        print(json.dumps({"plan": str(plan_path), "jobs": len(plan["jobs"]), "training_started": False,
                          "requires": "--execute", "stage": plan["stage"]}), flush=True)
        return
    preflight(plan)
    root = plan_path.resolve().parent
    lock = root / ".launch_once"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"launcher_pid": os.getpid(), "plan_sha256": sha256_file(plan_path),
                                 "code_hash": plan["code_hash"], "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    logs = root / "launcher_logs"
    logs.mkdir(exist_ok=False)
    (root / "run_commit.txt").write_text(f"git_commit={plan.get('git_commit')}\ncode_hash={plan['code_hash']}\n", encoding="utf-8")
    queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, job in enumerate(plan["jobs"]):
        queues[plan["gpus"][index % len(plan["gpus"])]].append(job)
    stop = threading.Event()
    mutex = threading.Lock()
    active: dict[str, subprocess.Popen] = {}
    exits: list[dict[str, Any]] = []

    def handle_stop(_number: int, _frame: Any) -> None:
        stop.set()
        with mutex:
            for process in active.values():
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)

    old_handlers = {number: signal.signal(number, handle_stop) for number in (signal.SIGINT, signal.SIGTERM)}

    def worker(gpu: str) -> None:
        for job in queues[gpu]:
            if stop.is_set():
                break
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
            log_path = logs / (job["name"] + ".log")
            with log_path.open("x", encoding="utf-8") as stream:
                process = subprocess.Popen(trainer_command(plan, job, python), cwd=ROOT, env=environment,
                                           stdout=stream, stderr=subprocess.STDOUT)
                with mutex:
                    active[gpu] = process
                    with (logs / "pids.tsv").open("a", encoding="utf-8") as handle:
                        handle.write(f"{job['name']}\t{gpu}\t{process.pid}\n")
                    if stop.is_set() and process.poll() is None:
                        process.send_signal(signal.SIGINT)
                return_code = process.wait()
            with mutex:
                active.pop(gpu, None)
                exits.append({"name": job["name"], "gpu": gpu, "exit_code": return_code})
                with (logs / "exit_codes.tsv").open("a", encoding="utf-8") as handle:
                    handle.write(f"{job['name']}\t{gpu}\t{return_code}\n")
            if return_code != 0:
                # Keep other currently running jobs intact; don't start queued jobs.
                stop.set()
                break

    try:
        with ThreadPoolExecutor(max_workers=len(plan["gpus"])) as executor:
            futures = [executor.submit(worker, gpu) for gpu in plan["gpus"] if queues[gpu]]
            for future in futures:
                future.result()
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)
        complete = len(exits) == len(plan["jobs"]) and all(item["exit_code"] == 0 for item in exits)
        atomic_json(root / "launcher_result.json", {"status": "completed" if complete else "stopped_or_failed",
                                                    "all_jobs_completed": complete, "exits": exits,
                                                    "planned_jobs": len(plan["jobs"]), "test_evaluated": False,
                                                    "training_jobs": sum(job.get("task_type", "train") == "train" for job in plan["jobs"]),
                                                    "evaluation_jobs": sum(job.get("task_type") == "evaluate" for job in plan["jobs"])})
    if not complete:
        raise SystemExit(1)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    for name in ("protocol", "cache-root"):
        plan.add_argument("--" + name, type=Path)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--weights-json", "--checkpoints", dest="weights_json", type=Path)
    plan.add_argument("--stage", "--phase", dest="stage", choices=STAGES, default="pilot")
    plan.add_argument("--videomamba-root", type=Path)
    plan.add_argument("--videomae-root", type=Path)
    plan.add_argument("--trained-plan", type=Path, help="Robustness only: explicit source training run_plan.json")
    plan.add_argument("--probes", default="none,static,shuffle", help="Robustness only; include the unchanged none baseline")
    plan.add_argument("--include-control-probes", action="store_true", help="Explicitly probe capacity/auxiliary/random-taxonomy runs as well")
    plan.add_argument("--gpus", default="0,1,2,3")
    plan.add_argument("--backbones", help="Declared feasibility subset; a subset is marked incomplete for formal claims")
    plan.add_argument("--taxonomy-file", type=Path)
    plan.add_argument("--batch-size", type=int, default=2)
    plan.add_argument("--accum-steps", type=int, default=16)
    plan.add_argument("--epochs", type=int, default=60)
    plan.add_argument("--warmup-epochs", type=int, default=5)
    plan.add_argument("--patience", type=int, default=10)
    plan.add_argument("--backbone-lr", type=float, default=1e-5)
    plan.add_argument("--head-lr", type=float, default=1e-4)
    plan.add_argument("--warmup-lr", type=float, default=1e-3)
    plan.add_argument("--weight-decay", type=float, default=.05)
    plan.add_argument("--class-weights", choices=("none", "sqrt_inverse", "inverse"), default="none")
    plan.add_argument("--workers", type=int, default=4)
    plan.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    plan.add_argument("--amp", choices=("none", "bfloat16", "float16"), default="bfloat16")
    plan.add_argument("--clip-grad", type=float, default=5.)
    plan.add_argument("--augmentation", choices=("none", "horizontal_flip"), default="horizontal_flip")
    launch = sub.add_parser("launch")
    launch.add_argument("--plan", type=Path, required=True)
    launch.add_argument("--python", default=sys.executable)
    launch.add_argument("--execute", action="store_true")
    return command


def main() -> None:
    args = parser().parse_args()
    if args.command == "plan":
        generate_plan(args)
    else:
        execute_plan(args.plan, python=args.python, execute=args.execute)


if __name__ == "__main__":
    main()
