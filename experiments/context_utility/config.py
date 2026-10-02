"""Explicit architecture contracts and strict evidence/checkpoint seals.

Historical private models are supported through a real factory plus a verified
audit. There is deliberately no guessed legacy architecture or prefix repair.
"""

import importlib
import inspect
import json
import math
from pathlib import Path

import torch
from torch import nn

from experiments.fsn_tcn.io import code_hashes as tcn_code_hashes
from experiments.fsn_tcn.model import TCNConfig, TemporalResidualTCN
from experiments.relation.data import sha256_file
from .adapters import LegacyPredictionAdapter


VARIANTS = ("continue", "aug", "scalar_aug", "dynamic_aug")
FEATURE_CONTRACT = "pooled_global_local_mean_frozen_A_logits_v1"
SEMANTIC_FIELDS = (
    "input_a_predictions", "feature_normalization", "input_layout",
    "temporal_architecture", "singleton", "loss", "optimizer",
)


def _positive_number(value, name, *, allow_zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero)):
        raise ValueError(f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} number")


def _positive_int(value, name, *, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")


def load_config(path):
    """Read one explicit configuration; do not fill undocumented defaults."""
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict) or config.get("schema") != "fsn-context-utility-v1":
        raise ValueError("unknown context utility config schema")
    seeds = config.get("seeds")
    if (not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be an explicit unique nonempty list of nonnegative integers")
    _positive_int(config.get("challenge_seed"), "challenge_seed", allow_zero=True)
    module = config.get("module")
    if not isinstance(module, dict):
        raise ValueError("module settings must be explicit")
    _positive_int(module.get("dim"), "module.dim")
    aug = config.get("augmentation")
    if not isinstance(aug, dict):
        raise ValueError("augmentation settings must be explicit")
    for key in ("anchors_per_segment", "max_neighbors"):
        _positive_int(aug.get(key), f"augmentation.{key}")
    _positive_number(aug.get("anchor_loss_weight"), "augmentation.anchor_loss_weight")
    opt = config.get("optimization")
    if not isinstance(opt, dict):
        raise ValueError("optimization settings must be explicit")
    for key in ("epochs", "patience", "batch_size"):
        _positive_int(opt.get(key), f"optimization.{key}")
    for key in ("lr", "clip_grad"):
        _positive_number(opt.get(key), f"optimization.{key}")
    _positive_number(opt.get("weight_decay"), "optimization.weight_decay", allow_zero=True)
    if opt.get("loss") != "CE":
        raise ValueError("only explicitly implemented unweighted seven-class CE is supported")
    if opt.get("optimizer") not in {"AdamW", "SGD"}:
        raise ValueError("optimizer must explicitly be AdamW or SGD")
    if opt["optimizer"] == "SGD":
        _positive_number(opt.get("momentum"), "optimization.momentum", allow_zero=True)
        if opt["momentum"] >= 1:
            raise ValueError("SGD momentum must be less than one")
    baseline = config.get("baseline")
    if not isinstance(baseline, dict) or baseline.get("kind") not in {"reference", "legacy"}:
        raise ValueError("baseline.kind must be reference or legacy")
    if baseline.get("feature_contract") != FEATURE_CONTRACT:
        raise ValueError("the frozen feature contract must be explicit and supported")
    if not isinstance(baseline.get("checkpoint_state_key"), (str, type(None))):
        raise ValueError("checkpoint_state_key must be an exact string or null for a raw state_dict")
    if "checkpoint_state_key" not in baseline:
        raise ValueError("checkpoint_state_key must be explicitly declared")
    if "checkpoint_target" in baseline and not isinstance(baseline["checkpoint_target"], str):
        raise ValueError("checkpoint_target must be an exact submodule path")
    paths = baseline.get("temporal_paths")
    if (not isinstance(paths, list) or not paths or any(not isinstance(p, str) or not p for p in paths)
            or len(set(paths)) != len(paths)):
        raise ValueError("temporal_paths must explicitly list unique convolution paths")
    if baseline["kind"] == "reference":
        parameters = baseline.get("config")
        if not isinstance(parameters, dict):
            raise ValueError("reference config must explicitly declare the architecture")
        if set(parameters) != {"width", "layers", "kernel_size", "dropout", "num_classes"}:
            raise ValueError("reference architecture must explicitly declare all five supported settings")
        architecture = TCNConfig(input_dim=1, **parameters)
        if architecture.num_classes != 7:
            raise ValueError("the reference must have seven classes")
        expected = [f"blocks.{i}.temporal" for i in range(architecture.layers)]
        if paths != expected:
            raise ValueError("reference temporal_paths must match its actual configured blocks")
        if baseline.get("checkpoint_target", ""):
            raise ValueError("reference checkpoint must target the whole base model")
    else:
        if (not isinstance(baseline.get("factory"), str) or baseline["factory"].count(":") != 1
                or not all(baseline["factory"].split(":"))):
            raise ValueError("legacy requires an explicit module:function factory")
        if not isinstance(baseline.get("kwargs"), dict):
            raise ValueError("legacy factory kwargs must be explicit")
        dependencies = baseline.get("source_dependencies")
        if (not isinstance(dependencies, list) or not dependencies
                or any(not isinstance(p, str) or not p for p in dependencies)):
            raise ValueError("legacy must list source_dependencies including actual historical model source")
        if not isinstance(baseline.get("legacy_audit"), str) or not baseline["legacy_audit"]:
            raise ValueError("legacy requires a verified legacy_audit file")
    return config


def code_hashes():
    root = Path(__file__).resolve().parents[2]
    return tcn_code_hashes() | {
        str(p.relative_to(root)): sha256_file(p)
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }


def _factory(baseline):
    module_name, function_name = baseline["factory"].split(":")
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise ValueError("legacy factory must be callable")
    return module, factory


def baseline_source_hashes(config):
    baseline = config["baseline"]
    if config.get("readiness") == "unavailable_until_verified":
        raise ValueError("legacy template is not runnable: verify actual source, configuration, checkpoint and audit first")
    if baseline["kind"] == "reference":
        paths = {Path(inspect.getfile(TemporalResidualTCN)).resolve()}
    else:
        module, factory = _factory(baseline)
        paths = {Path(inspect.getfile(module)).resolve(), Path(inspect.getfile(factory)).resolve()}
        dependencies = baseline.get("source_dependencies")
        if not isinstance(dependencies, list) or not dependencies:
            raise ValueError("legacy source dependencies are required")
        paths.update(Path(p).resolve() for p in dependencies)
    if any(not p.is_file() for p in paths):
        raise ValueError("baseline source dependency does not exist")
    return {str(p): sha256_file(p) for p in sorted(paths)}


def _architecture(model, paths):
    result = {}
    for path in paths:
        try:
            conv = model.get_submodule(path)
        except (AttributeError, KeyError) as exc:
            raise ValueError(f"declared temporal path does not exist: {path}") from exc
        if not isinstance(conv, nn.Conv1d) or conv.kernel_size[0] <= 1:
            raise ValueError(f"declared temporal path must be a cross-time Conv1d: {path}")
        result[path] = {
            "class": f"{type(conv).__module__}.{type(conv).__name__}",
            "in_channels": conv.in_channels, "out_channels": conv.out_channels,
            "kernel_size": list(conv.kernel_size), "stride": list(conv.stride),
            "padding": list(conv.padding) if isinstance(conv.padding, tuple) else conv.padding,
            "dilation": list(conv.dilation), "groups": conv.groups,
            "bias": conv.bias is not None, "padding_mode": conv.padding_mode,
        }
    return result


def _legacy_audit(config, model, architecture, sources, checkpoint_sha, fingerprint):
    path = Path(config["baseline"]["legacy_audit"]).resolve()
    if not path.is_file():
        raise ValueError("legacy audit is unavailable; do not substitute a guessed historical baseline")
    audit = json.loads(path.read_text())
    if audit.get("schema") != "fsn-context-utility-legacy-audit-v1":
        raise ValueError("unknown legacy audit schema")
    if fingerprint is None or audit.get("fingerprint") != fingerprint:
        raise ValueError("legacy audit must seal the exact current frozen evidence fingerprint")
    if audit.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("legacy checkpoint SHA disagrees with verified audit")
    # Check external model class definitions as well as the factory/declarations.
    for submodule in model.modules():
        cls = type(submodule)
        if cls.__module__.startswith(("torch.", "experiments.context_utility.")):
            continue
        class_path = Path(inspect.getfile(cls)).resolve()
        if str(class_path) not in sources:
            raise ValueError(f"actual legacy model source missing from dependencies: {class_path}")
    if audit.get("source_sha256") != sources:
        raise ValueError("legacy model/factory source hashes disagree with verified audit")
    semantics = audit.get("semantics")
    if (not isinstance(semantics, dict) or any(key not in semantics for key in SEMANTIC_FIELDS)
            or any(semantics[key] in (None, "", "unknown", "TODO") for key in SEMANTIC_FIELDS)):
        raise ValueError("legacy audit must describe verified predictions, normalization, layout, blocks, singleton, loss and optimizer")
    if semantics["temporal_architecture"] != architecture:
        raise ValueError("legacy audited temporal architecture differs from the actual factory model")
    if (semantics["loss"] != config["optimization"]["loss"]
            or semantics["optimizer"] != config["optimization"]["optimizer"]):
        raise ValueError("legacy verified loss/optimizer must match the supported continuation configuration")
    if isinstance(model, LegacyPredictionAdapter):
        for field, actual in (("input_a_predictions", model.prediction_input),
                              ("input_layout", model.input_layout), ("singleton", model.singleton)):
            if semantics[field] != actual:
                raise ValueError(f"legacy adapter semantics disagree with verified {field}")
    return {"path": str(path), "sha256": sha256_file(path), "semantics": semantics}


def build_base(config, input_dim, checkpoint, device="cpu", fingerprint=None):
    """Instantiate the exact declared base, verify provenance, then strict-load.

    This supports a historical raw state_dict only with an explicit null state
    key and a complete matching legacy audit. Reference checkpoints must carry
    their original evidence/config seals and come from the full-open strategy.
    """
    baseline = config["baseline"]
    sources = baseline_source_hashes(config)
    if baseline["kind"] == "reference":
        base = TemporalResidualTCN(TCNConfig(input_dim=input_dim, **baseline["config"]))
    else:
        _, factory = _factory(baseline)
        base = factory(input_dim=input_dim, **baseline["kwargs"])
    if not isinstance(base, nn.Module):
        raise ValueError("baseline factory must return an nn.Module")
    architecture = _architecture(base, baseline["temporal_paths"])
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise ValueError("baseline checkpoint is unavailable")
    checkpoint_sha = sha256_file(checkpoint)
    # weights_only avoids executing pickle code in external checkpoints. Legacy
    # assets must be converted explicitly if they contain custom Python objects.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    verified_legacy = None
    checkpoint_source_check = {"available": False, "checked": False, "key": None}
    if baseline["kind"] == "reference":
        if not isinstance(saved, dict) or fingerprint is None or saved.get("fingerprint") != fingerprint:
            raise ValueError("reference checkpoint must match current frozen evidence fingerprint")
        if saved.get("strategy") != "all":
            raise ValueError("reference starting checkpoint must be a full-open all-strategy TCN")
        old_baseline = saved.get("config", {}).get("baseline", {})
        if old_baseline.get("kind") != "reference" or old_baseline.get("config") != baseline["config"]:
            raise ValueError("reference checkpoint architecture differs from the declared baseline")
        recorded_sources = saved.get("protocol", {}).get("code_sha256", {})
        model_key = "experiments/fsn_tcn/model.py"
        if model_key in recorded_sources:
            model_path = Path(inspect.getfile(TemporalResidualTCN)).resolve()
            if recorded_sources[model_key] != sha256_file(model_path):
                raise ValueError("reference checkpoint base-model source SHA differs from current reference implementation")
            checkpoint_source_check = {"available": True, "checked": True,
                                       "key": model_key, "sha256": recorded_sources[model_key]}
    else:
        verified_legacy = _legacy_audit(config, base, architecture, sources, checkpoint_sha, fingerprint)
        if isinstance(saved, dict) and "fingerprint" in saved and saved["fingerprint"] != fingerprint:
            raise ValueError("legacy checkpoint evidence fingerprint disagrees with audit")
    state_key = baseline["checkpoint_state_key"]
    if state_key is None:
        state = saved
    else:
        if not isinstance(saved, dict) or state_key not in saved:
            raise ValueError(f"checkpoint lacks explicitly declared state key {state_key!r}")
        state = saved[state_key]
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("declared checkpoint state must be a tensor-only state_dict")
    target_path = baseline.get("checkpoint_target", "")
    try:
        target = base.get_submodule(target_path) if target_path else base
    except (AttributeError, KeyError) as exc:
        raise ValueError("explicit checkpoint_target does not exist") from exc
    target.load_state_dict(state, strict=True)
    audit = {
        "baseline_kind": baseline["kind"], "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha, "checkpoint_state_key": state_key,
        "checkpoint_target": target_path, "checkpoint_fingerprint_checked": True,
        "checkpoint_base_source_check": checkpoint_source_check,
        "source_sha256": sources, "architecture": architecture,
        "model_class": f"{type(base).__module__}.{type(base).__name__}",
        "input_dim": input_dim, "legacy_audit": verified_legacy,
        "strict_state_load": True,
    }
    return base.to(device), audit
