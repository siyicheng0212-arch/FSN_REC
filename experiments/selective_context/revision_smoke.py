"""Real CUDA preflight for a gate over the verified historical A+TCN.

This check uses unmodified train7372 evidence and precisely the historical
``full_chain`` or ``eligible_segments`` units.  It never trains the frozen A
or old TCN, and a CPU fixture cannot certify formal training readiness.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.fsn_tcn.io import load_chain, write_json
from .revision_config import VARIANTS
from .revision_evaluate import _historical_segments
from .revision_train import ROOT, make_revision_model, prepare_revision_inputs, revision_smoke_seal


def select_smoke_units(bundle, chain_layout):
    """Select an actual multi-clip and singleton historical train unit.

    An eligible segment is selected only at its existing historical boundary;
    a full chain is never trimmed or resegmented to manufacture a short input.
    Choosing the shortest existing multi-clip unit makes preflight economical
    without changing its meaning.  Return chain references, never clip IDs.
    """
    multi, singleton = None, None
    for chain in bundle.chains["train"]:
        for start, stop in _historical_segments(chain, chain_layout):
            length = stop - start
            if length < 1:
                raise ValueError("historical train chain has an empty segment")
            candidate = (chain, start, stop)
            if length == 1:
                if singleton is None:
                    singleton = candidate
            elif multi is None or length < multi[2] - multi[1]:
                multi = candidate
    if multi is None or singleton is None:
        raise ValueError("real train split needs both a multi-clip historical unit and a singleton")
    return {"multi": multi, "singleton": singleton}


def _read_unit(bundle, unit, device):
    chain, start, stop = unit
    # For eligible_segments the full original chain is read before selecting
    # the exact old-model unit; for full_chain start=0 and stop=len(chain).
    features, a_logits, labels = load_chain(bundle, chain, device)
    features, a_logits, labels = (x[start:stop] for x in (features, a_logits, labels))
    if (features.ndim != 2 or a_logits.shape != (len(features), 7)
            or labels.shape != (len(features),) or not features.is_cuda
            or not a_logits.is_cuda or not labels.is_cuda
            or not bool(torch.isfinite(features).all())
            or not bool(torch.isfinite(a_logits).all())
            or not bool(((0 <= labels) & (labels < 7)).all())):
        raise ValueError("historical train unit has invalid real CUDA evidence")
    return features, a_logits, labels


def _correction_magnitude(base, unit):
    features, a_logits, _ = unit
    with torch.no_grad():
        old = base(features, a_logits)
        if old.shape != a_logits.shape or not bool(torch.isfinite(old).all()):
            raise RuntimeError("historical train unit produced invalid old TCN scores")
        return float((old - a_logits).abs().max())


def choose_informative_multi(prepared, device, initial, *, minimum_difference=1e-6):
    """Use an actual historical train unit where the old TCN proposes a change.

    A zero old-minus-A difference makes every acceptance gradient zero.  Scan
    shortest complete original units first, stopping at the first usable unit;
    never shorten a unit or manufacture a synthetic donor to pass smoke.
    """
    base, bundle = prepared["base"].eval(), prepared["bundle"]
    difference = _correction_magnitude(base, initial)
    if difference > minimum_difference:
        return initial, difference, 1
    candidates = [(stop - start, len(chain["ordered_clip_ids"]), chain, start, stop)
                  for chain in bundle.chains["train"]
                  for start, stop in _historical_segments(chain, prepared["parity"]["chain_layout"])
                  if stop - start > 1]
    attempts = 1
    for _, _, chain, start, stop in sorted(candidates, key=lambda item: item[:2]):
        candidate = _read_unit(bundle, (chain, start, stop), device)
        difference = _correction_magnitude(base, candidate)
        attempts += 1
        if difference > minimum_difference:
            return candidate, difference, attempts
    raise RuntimeError("old TCN proposes no nonzero train-unit correction; G cannot be trained")


def _finite_gate(alpha, length, device):
    if (alpha.shape != (length,) or alpha.device != device
            or not bool(torch.isfinite(alpha).all())
            or bool(((alpha < 0) | (alpha > 1)).any())):
        raise RuntimeError("revision acceptance gate returned an invalid coefficient")


def run_checks(prepared, units, variant, device):
    """CPU-testable helper; only ``run_smoke`` can attest real CUDA evidence."""
    model = make_revision_model(prepared, variant, prepared["config"]["seeds"][0], device)
    model.eval()
    if any(parameter.requires_grad for parameter in model.base_tcn.parameters()):
        raise RuntimeError("old TCN was not frozen in the revision wrapper")
    old_state = {key: tensor.detach().clone()
                 for key, tensor in model.base_tcn.state_dict().items()}
    original = prepared["base"].eval()
    audit = {"variant": variant, "historical_unit_lengths": {},
             "unit_matches_old_exactly": False, "zero_matches_A_exactly": False,
             "singleton_exercised": False, "steps": 0, "CE_losses": [],
             "nonzero_gate_gradient_tensors": [], "updated_gate_tensors": [],
             "old_TCN_unchanged": False,
             "gate_trainable_parameters": model.added_parameter_count()}
    with torch.no_grad():
        for name, (features, a_logits, _) in units.items():
            reference = original(features, a_logits)
            if (reference.shape != a_logits.shape or not bool(torch.isfinite(reference).all())):
                raise RuntimeError("old TCN has invalid scores on a historical train unit")
            # The unit path returns the *old tensor itself*, avoiding changes
            # from even floating point interpolation at alpha=1.
            unit, details = model(features, a_logits, gate_mode="unit", return_details=True)
            zero = model(features, a_logits, gate_mode="zero")
            learned, gate = model(features, a_logits, return_details=True)
            if not torch.equal(unit, reference) or not torch.equal(details["tcn_logits"], reference):
                raise RuntimeError("unit gate changed old TCN output on the historical unit")
            if not torch.equal(zero, a_logits):
                raise RuntimeError("zero gate does not return exact frozen A logits")
            if learned.shape != a_logits.shape or not bool(torch.isfinite(learned).all()):
                raise RuntimeError("learned gate returned invalid scores")
            _finite_gate(gate["alpha"], len(features), features.device)
            if not torch.equal(details["alpha"], torch.ones_like(details["alpha"])):
                raise RuntimeError("unit gate did not produce alpha=1")
            audit["historical_unit_lengths"][name] = len(features)
        audit.update(unit_matches_old_exactly=True, zero_matches_A_exactly=True,
                     singleton_exercised=(len(units["singleton"][0]) == 1))
    if not audit["singleton_exercised"] or len(units["multi"][0]) <= 1:
        raise RuntimeError("the smoke did not exercise both historical unit sizes")
    settings = prepared["config"]["optimization"]
    trainable = {name: parameter for name, parameter in model.named_parameters()
                 if parameter.requires_grad}
    if (not trainable or len(trainable) == len(list(model.named_parameters()))
            or any(name.startswith("base_tcn.") for name in trainable)):
        raise RuntimeError("revision optimizer must contain only gate parameters")
    optimizer = torch.optim.AdamW(trainable.values(), lr=settings["lr"],
                                  weight_decay=settings["weight_decay"])
    for _ in range(3):
        model.train()
        if model.base_tcn.training:
            raise RuntimeError("frozen old TCN entered train mode")
        before = {name: parameter.detach().clone() for name, parameter in trainable.items()}
        optimizer.zero_grad(set_to_none=True)
        total_loss, count = None, 0
        for features, a_logits, labels in units.values():
            logits, details = model(features, a_logits, return_details=True)
            _finite_gate(details["alpha"], len(features), features.device)
            part = F.cross_entropy(logits.float(), labels, reduction="sum")
            total_loss = part if total_loss is None else total_loss + part
            count += len(labels)
        loss = total_loss / count
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite revision gate CUDA smoke CE")
        loss.backward()
        if any(parameter.grad is not None for parameter in model.base_tcn.parameters()):
            raise RuntimeError("old TCN unexpectedly received a gradient")
        active = 0
        for parameter in trainable.values():
            gradient = parameter.grad
            if gradient is not None:
                if not bool(torch.isfinite(gradient).all()):
                    raise RuntimeError("nonfinite revision gate gradient")
                active += int(bool((gradient != 0).any()))
        if active == 0:
            raise RuntimeError("revision gate did not receive a nonzero CUDA gradient")
        torch.nn.utils.clip_grad_norm_(trainable.values(), settings["grad_clip"],
                                       error_if_nonfinite=True)
        optimizer.step()
        changed = sum(not torch.equal(parameter.detach(), before[name])
                      for name, parameter in trainable.items())
        if changed == 0 or any(not bool(torch.isfinite(parameter).all())
                               for parameter in trainable.values()):
            raise RuntimeError("revision gate did not update finite parameters")
        audit["CE_losses"].append(float(loss.detach()))
        audit["nonzero_gate_gradient_tensors"].append(active)
        audit["updated_gate_tensors"].append(changed)
        audit["steps"] += 1
    if any(not torch.equal(old_state[key], value) for key, value
           in model.base_tcn.state_dict().items()):
        raise RuntimeError("historical old TCN state changed during gate smoke")
    audit["old_TCN_unchanged"] = True
    return audit


def run_smoke(prepared, device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real CUDA is required for formal revision gate smoke")
    torch.empty(1, device=device)  # Force actual device allocation before declaring success.
    selected = select_smoke_units(prepared["bundle"], prepared["parity"]["chain_layout"])
    units = {name: _read_unit(prepared["bundle"], chosen, device)
             for name, chosen in selected.items()}
    units["multi"], difference, attempts = choose_informative_multi(prepared, device, units["multi"])
    torch.cuda.reset_peak_memory_stats(device)
    variants = {name: run_checks(prepared, units, name, device) for name in VARIANTS}
    torch.cuda.synchronize(device)
    if set(variants) != set(VARIANTS) or any(group["steps"] != 3 for group in variants.values()):
        raise RuntimeError("all four gate variants require three real CUDA optimizer steps")
    return {"real_train_split_historical_units": True,
            "old_TCN_full_unit_exact": True, "A_zero_gate_exact": True,
            "historical_singleton_checked": True,
            "multi_old_A_max_abs_logit_difference": difference,
            "informative_unit_attempts": attempts,
            "variants": variants,
            "A_trainable_parameters": 0, "old_TCN_trainable_parameters": 0,
            "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "test_split_read": False}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-index", "source-protocol-dir", "legacy-config", "legacy-checkpoint",
                 "historical-predictions", "chain-layout", "config", "output"):
        choices = ("full_chain", "eligible_segments") if name == "chain-layout" else None
        p.add_argument("--" + name, required=True, choices=choices)
    p.add_argument("--historical-column-map")
    p.add_argument("--historical-logits-key")
    p.add_argument("--device", default="cuda:0")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError("CUDA smoke report must be fresh and outside the code worktree")
    report = {"schema": "fsn-tcn-revision-smoke-v1", "all_passed": False,
              "real_cuda_tcn_revision_smoke_completed": False,
              "formal_training_started": False}
    try:
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("real CUDA is required; a CPU fixture never certifies formal training")
        prepared = prepare_revision_inputs(args, device=device)
        report.update(revision_smoke_seal(args, prepared))
        report.update(run_smoke(prepared, device))
        report["all_passed"] = True
        report["real_cuda_tcn_revision_smoke_completed"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report, exclusive=True)
        raise
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report, exclusive=True)
    return report


if __name__ == "__main__":
    main()
