"""Real frozen-evidence CUDA checks, after the raw-cache frozen-A preflight."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.relation.data import sha256_file
from .config import baseline_source_hashes, build_base, load_config
from .io import code_hashes, load_bundle, load_chain, write_json
from .model import SegmentedTCN
from .train import gradients_finite, seed_everything


def run_tcn_checks(base, features, a_logits, labels):
    """Reusable CPU/CUDA engineering checks; no claim of a completed experiment."""
    if len(features) < 4:
        raise ValueError("checks require two real pairs (at least four rows)")
    model = SegmentedTCN(base)
    base.eval()
    all_on = [True] * (len(features) - 1)
    all_off = [False] * (len(features) - 1)
    with torch.no_grad():
        direct = base(features, a_logits)
        if not torch.equal(direct, model(features, a_logits, all_on)):
            raise RuntimeError("all-on wrapper differs from supplied baseline")
        if not torch.equal(base(features[:1], a_logits[:1]), a_logits[:1]):
            raise RuntimeError("supplied baseline lacks required singleton fallback; do not claim historical parity")
        if not torch.equal(model(features, a_logits, all_off), a_logits):
            raise RuntimeError("all-off did not return exact frozen A")
        cut = [True, False] + [True] * (len(features) - 3)
        reference = model(features, a_logits, cut)
        changed_features = features.clone()
        changed_logits = a_logits.clone()
        changed_features[2:] = -17 * changed_features[2:] + 3
        changed_logits[2:] = -11 * changed_logits[2:] + 5
        changed = model(changed_features, changed_logits, cut)
        if not torch.equal(reference[:2], changed[:2]):
            raise RuntimeError("right segment influenced left predictions across a cut")
    # Test after nonzero updates as well; a zero residual head can conceal leaks.
    before = {name: p.detach().clone() for name, p in base.named_parameters()}
    optimizer = torch.optim.AdamW(base.parameters(), lr=.001, weight_decay=0.0)
    losses, seen = [], set()
    base.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = model(features, a_logits, cut)
        loss = F.cross_entropy(output, labels)
        if not bool(torch.isfinite(loss)) or not loss.requires_grad:
            raise RuntimeError("TCN smoke has no finite trainable loss")
        loss.backward()
        gradients_finite(base)
        seen.update(name for name, p in base.named_parameters()
                    if p.grad is not None and bool(p.grad.abs().sum() > 0))
        optimizer.step()
        losses.append(float(loss.detach()))
    changed_parameters = [name for name, p in base.named_parameters() if not torch.equal(before[name], p)]
    if not changed_parameters or not seen:
        raise RuntimeError("TCN parameters did not update")
    base.eval()
    with torch.no_grad():
        reference = model(features, a_logits, cut)
        changed = model(changed_features, changed_logits, cut)
        if not torch.equal(reference[:2], changed[:2]):
            raise RuntimeError("trained TCN leaked right information across a cut")
        if not torch.equal(model(features, a_logits, all_off), a_logits):
            raise RuntimeError("trained all-off fallback changed A")
    leaf = features.detach().clone().requires_grad_(True)
    model(leaf, a_logits, cut)[:2].square().sum().backward()
    if leaf.grad is not None and bool(leaf.grad[2:].abs().sum() > 0):
        raise RuntimeError("gradient crossed a cut")
    return {"all_passed": True, "all_on_exact_baseline": True,
            "singleton_exact_A": True, "all_off_exact_A": True,
            "right_to_left_isolated_before_and_after_updates": True,
            "cut_gradient_isolation": True, "losses": losses,
            "updated_parameter_tensors": len(changed_parameters),
            "nonzero_gradient_parameter_tensors": len(seen)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-index", "source-protocol-dir", "config", "raw-cache-report", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    out = Path(args.output)
    report = {"all_passed": False, "status": "checking", "real_cuda_tcn_smoke_completed": False}
    write_json(out, report, exclusive=True)
    try:
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("formal TCN smoke requires real CUDA")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        config = load_config(args.config)
        bundle = load_bundle(args.feature_index, args.source_protocol_dir)
        raw = json.loads(Path(args.raw_cache_report).read_text())
        if (raw.get("all_passed") is not True or raw.get("real_cache_cuda_smoke_completed") is not True
                or raw.get("checkpoint_sha256") != bundle.index.checkpoint_sha256
                or raw.get("manifest_sha256") != bundle.fingerprint["manifest_sha256"]
                or raw.get("source_audit_sha256") != bundle.fingerprint["source_audit_sha256"]):
            raise ValueError("raw-cache frozen-A smoke is missing or uses different inputs")
        pieces = []
        for chain in bundle.chains["train"]:
            for i, eligible in enumerate(chain["eligible"]):
                if eligible:
                    pieces.append(load_chain(bundle, {"ordered_clip_ids": chain["ordered_clip_ids"][i:i + 2]}, device))
                    break
            if len(pieces) == 2:
                break
        if len(pieces) != 2:
            raise ValueError("two original legal train pairs are needed for CUDA checks")
        features, a_logits, labels = (torch.cat([piece[i] for piece in pieces], dim=0) for i in range(3))
        seed_everything(config["seeds"][0])
        base = build_base(config, features.shape[1], device)
        report.update(run_tcn_checks(base, features, a_logits, labels))
        torch.cuda.synchronize(device)
        report.update(status="complete", real_cuda_tcn_smoke_completed=True,
                      fingerprint=bundle.fingerprint, source_sha256=code_hashes(),
                      baseline_source_sha256=baseline_source_hashes(config),
                      config_sha256=sha256_file(args.config), device=str(device),
                      raw_cache_report_sha256=sha256_file(args.raw_cache_report),
                      gpu_name=torch.cuda.get_device_name(device),
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      probe_note="two legal real training pairs with an explicit separating cut; not a new dataset chain")
    except Exception as error:
        report.update(status="failed", all_passed=False, error_type=type(error).__name__, error=str(error))
        write_json(out, report)
        raise
    write_json(out, report)


if __name__ == "__main__":
    main()
