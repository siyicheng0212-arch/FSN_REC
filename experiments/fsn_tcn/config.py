"""One explicit, sealed configuration shared by all four independent groups."""
import importlib
import inspect
import json
import math
from pathlib import Path

import torch

from experiments.relation.data import sha256_file
from .model import TCNConfig, TemporalResidualTCN


STRATEGIES = ("all", "source_rule", "random", "learned")


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("schema") != "fsn-tcn-minimal-v1":
        raise ValueError("unknown FSN-TCN config schema")
    seeds = config.get("seeds")
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
            or any(type(x) is not int or x < 0 for x in seeds)):
        raise ValueError("seeds must be an explicit nonempty unique list of nonnegative integers")
    if type(config.get("challenge_seed")) is not int or config["challenge_seed"] < 0:
        raise ValueError("one nonnegative challenge_seed must be fixed across every model and training seed")
    threshold = config.get("threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError("fix a finite threshold in (0,1) before this experiment")
    for name in ("relation", "optimization", "baseline"):
        if not isinstance(config.get(name), dict):
            raise ValueError(f"missing {name} configuration")
    for section in ("relation", "optimization"):
        settings = config[section]
        for name in ("epochs", "patience", "batch_size"):
            if type(settings.get(name)) is not int or settings[name] < 1:
                raise ValueError(f"{section}.{name} must be a positive integer")
        for name in ("lr", "weight_decay", "clip_grad"):
            value = settings.get(name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"invalid {section}.{name}")
    if config["optimization"].get("optimizer") != "AdamW" or config["optimization"].get("loss") != "CE":
        raise ValueError("minimal suite uses one AdamW optimizer and ordinary seven-class CE")
    if config["baseline"].get("kind") not in {"reference", "external"}:
        raise ValueError("baseline.kind must explicitly be reference or external")
    if config["baseline"]["kind"] == "reference":
        TCNConfig(input_dim=1, **config["baseline"]["config"])
    else:
        if not isinstance(config["baseline"].get("factory"), str) or ":" not in config["baseline"]["factory"]:
            raise ValueError("external baseline requires module:factory returning the two-input TCN contract")
        if not isinstance(config["baseline"].get("kwargs"), dict):
            raise ValueError("external baseline kwargs must be explicit")
    if (isinstance(config.get("hard_negative_ratio"), bool)
            or not isinstance(config.get("hard_negative_ratio"), (int, float))
            or not math.isfinite(config["hard_negative_ratio"]) or config["hard_negative_ratio"] <= 0):
        raise ValueError("positive training hard-negative ratio required")
    return config


def baseline_source_hashes(config):
    baseline = config["baseline"]
    if baseline["kind"] == "reference":
        path = Path(inspect.getfile(TemporalResidualTCN)).resolve()
        return {str(path): sha256_file(path)}
    module_name, name = baseline["factory"].split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, name)
    if not callable(factory):
        raise ValueError("external factory is not callable")
    paths = {Path(inspect.getfile(module)).resolve(), Path(inspect.getfile(factory)).resolve()}
    dependencies = baseline.get("source_dependencies")
    if not isinstance(dependencies, list):
        raise ValueError("external baseline must explicitly list source_dependencies, including imported model code")
    paths.update(Path(path).resolve() for path in dependencies)
    return {str(path): sha256_file(path) for path in sorted(paths)}


def build_base(config, input_dim, device="cpu"):
    baseline = config["baseline"]
    if baseline["kind"] == "reference":
        model = TemporalResidualTCN(TCNConfig(input_dim=input_dim, **baseline["config"]))
    else:
        module_name, name = baseline["factory"].split(":", 1)
        model = getattr(importlib.import_module(module_name), name)(input_dim=input_dim, **baseline["kwargs"])
    if not isinstance(model, torch.nn.Module):
        raise ValueError("baseline factory must return an nn.Module")
    return model.to(device)
