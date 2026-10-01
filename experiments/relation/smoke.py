"""Check frozen A, trainable R and independent D fallback before feature export.

The formal CLI requires real CUDA and the sealed train7372/val823 source policy.
The reusable helper also runs on CPU with supplied evidence; that is a unit
check, not a report of real-cache CUDA success.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import read_jsonl, sha256_file
from .decoder import decode
from .evidence import EVIDENCE_KEYS, POSITION_BASIS, load_original
from .network import RelationConfig, RelationNet
from .protocol import validate_source_artifacts, relation_source_hashes


def relation_code_sha256():
    return relation_source_hashes()


def _checked_evidence(evidence, device):
    checked = {}
    shapes = {"logits": (2, 7), "global_tokens": (2, 8, 1280),
              "local_tokens": (2, 12, 2048), "global_positions": (2, 8),
              "local_positions": (2, 12)}
    for key, shape in shapes.items():
        value = evidence.get(key)
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
            raise ValueError(f"invalid finite evidence shape: {key}, expected {shape}")
        checked[key] = value.detach().clone().to(device=device, dtype=torch.float32)
    return checked


def run_relation_smoke(evidence_left, evidence_right, device="cpu"):
    """Three real optimizer steps for C/D pair targets [1,0], without training A.

    Evidence contains two pairs per batch. Targets encode the source policy,
    not seven-class actions. Every trainable parameter tensor must receive a
    finite nonzero gradient during at least one step; individual weight entries
    are not required to be nonzero. Cloned leaf inputs test gradient isolation.
    """
    device = torch.device(device)
    left, right = (_checked_evidence(x, device) for x in (evidence_left, evidence_right))
    for side in (left, right):
        for key in EVIDENCE_KEYS:
            side[key].requires_grad_(True)
    targets = torch.tensor([1., 0.], device=device)
    modes = {}
    for mode in ("mlp", "dual"):
        torch.manual_seed(42)
        model = RelationNet(RelationConfig(mode=mode)).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        seen = {name: False for name, p in model.named_parameters() if p.requires_grad}
        losses = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(left, right), targets)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"{mode}: nonfinite loss")
            loss.backward()
            losses.append(float(loss.detach().cpu()))
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError(f"{mode}: nonfinite gradient for {name}")
                    seen[name] |= bool(parameter.grad.abs().sum() > 0)
            if any(side[key].grad is not None for side in (left, right) for key in EVIDENCE_KEYS):
                raise RuntimeError(f"{mode}: gradients reached frozen A evidence")
            optimizer.step()
        inactive = [name for name, received in seen.items() if not received]
        if inactive:
            raise RuntimeError(f"{mode}: inactive parameter tensors: {inactive}")
        modes[mode] = {"losses": losses, "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad),
            "parameter_tensors_with_finite_nonzero_gradient": len(seen),
            "all_active_parameter_tensors_received_gradients": True,
            "frozen_evidence_gradient_isolation": True}
    # Use a nonuniform transition to ensure fallback is due to the gate being off.
    transition = np.full((7, 7), .05)
    np.fill_diagonal(transition, .70)
    visual_logits = torch.cat((left["logits"], right["logits"])).detach().cpu().numpy()
    decoded = decode(visual_logits, transition, np.zeros(3), eligible=np.ones(3, dtype=bool))
    if not np.array_equal(decoded["predictions"], np.argmax(visual_logits, axis=1)):
        raise RuntimeError("all-off decoder changed Original predictions")
    return {"modes": modes, "decoder_all_off_exact_visual_argmax": True,
            "evidence_shapes": {key: list(value.shape) for key, value in left.items()},
            "position_basis": POSITION_BASIS, "all_passed": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest-dir", "cache-root", "original-checkpoint", "source-protocol-dir", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"all_passed": False, "status": "checking", "device": args.device,
              "source_sha256": relation_code_sha256(),
              "real_cache_cuda_smoke_completed": False}
    # Reserve a fresh report before work; a failed check leaves explicit evidence.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    try:
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("formal cache smoke requires an available CUDA device")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        audit = validate_source_artifacts(args.source_protocol_dir)
        manifest_dir, protocol_dir = Path(args.manifest_dir), Path(args.source_protocol_dir)
        hashes = {s: sha256_file(manifest_dir / (s + ".jsonl")) for s in ("train", "val")}
        if hashes != audit["manifest_sha256"]:
            raise ValueError("smoke manifests differ from sealed source-policy manifests")
        report.update(manifest_sha256=hashes,
                      checkpoint_sha256=sha256_file(args.original_checkpoint),
                      source_audit_sha256=sha256_file(protocol_dir / "audit.json"),
                      sampling="36/8/12/128", label_policy=audit["label_policy"])
        edges = read_jsonl(protocol_dir / "edges.jsonl")
        pairs = [next((e for e in edges if e.get("split") == "train" and e.get("status") == status), None)
                 for status in ("C", "D")]
        if any(pair is None for pair in pairs):
            raise ValueError("smoke requires at least one train C and one train D edge")
        from experiments.aligned_data import ResolvedFullClipDataset
        dataset = ResolvedFullClipDataset(manifest_dir / "train.jsonl", args.cache_root,
                                         expected_split="train")
        indices = {row.clip_id: i for i, row in enumerate(dataset.records)}
        frames = [torch.stack([dataset[indices[pair[key]]]["video"] for pair in pairs])
                  .to(device).float().div_(255) for key in ("left_clip_id", "right_clip_id")]
        extractor = load_original(args.original_checkpoint, device, require_full_protocol=True)
        if any(p.requires_grad or p.grad is not None for p in extractor.parameters()):
            raise RuntimeError("A is not frozen before smoke")
        evidence, differences = [], []
        for side in frames:
            with torch.no_grad():
                direct = extractor.visual(side)["logits"]
                extracted = extractor(side)
            difference = float((direct - extracted["logits"]).abs().max().cpu())
            differences.append(difference)
            if not bool(torch.isfinite(direct).all()) or difference > 1e-6:
                raise RuntimeError("evidence extraction changed Original logits")
            evidence.append(extracted)
        report["max_direct_extracted_logit_difference"] = max(differences)
        report.update(run_relation_smoke(*evidence, device=device))
        if any(p.requires_grad or p.grad is not None for p in extractor.parameters()):
            raise RuntimeError("A is not frozen after smoke")
        torch.cuda.synchronize(device)
        report.update(status="complete", A_frozen_no_gradients=True,
                      real_cache_cuda_smoke_completed=True,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    except Exception as exc:
        report.update(status="failed", all_passed=False, error_type=type(exc).__name__, error=str(exc))
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        raise
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"all_passed": report["all_passed"], "report": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
