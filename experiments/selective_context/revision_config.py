"""Fixed plan for a gate over the genuine, unchanged historical A+TCN."""

from __future__ import annotations

import json
import math
from pathlib import Path


SCHEMA = "fsn-tcn-revision-v1"
VARIANTS = ("scalar", "class_conditioned", "logits_only", "visual_logits")


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value, name, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {
        "schema", "seeds", "variants", "module", "optimization"
    } or config["schema"] != SCHEMA:
        raise ValueError("unknown/incomplete revision config schema")
    if (not isinstance(config["seeds"], list) or not config["seeds"]
            or len(set(config["seeds"])) != len(config["seeds"])):
        raise ValueError("seeds must be an explicit unique nonempty list")
    for value in config["seeds"]:
        _integer(value, "seed", minimum=0)
    if not isinstance(config["variants"], list) or tuple(config["variants"]) != VARIANTS:
        raise ValueError("all matched predeclared gate variants must be present in fixed order")
    module = config["module"]
    if not isinstance(module, dict) or set(module) != {"gate_dim"}:
        raise ValueError("module must explicitly declare gate_dim")
    _integer(module["gate_dim"], "gate_dim")
    opt = config["optimization"]
    if not isinstance(opt, dict) or set(opt) != {
        "epochs", "patience", "chains_per_step", "lr", "weight_decay", "grad_clip"
    }:
        raise ValueError("optimization settings are incomplete or unsupported")
    for name in ("epochs", "patience", "chains_per_step"):
        _integer(opt[name], name)
    for name in ("lr", "grad_clip"):
        _number(opt[name], name, positive=True)
    _number(opt["weight_decay"], "weight_decay")
    return config
