"""Frozen VideoMAE V2 evidence and exact D reuse for the FSN source-policy R+D run.

Only the fixed train7372/val823 development split is read. This module does
not train A or R, read test, or modify the OpenGVLab checkout. Its ``export``
command requires the completed, SHA-sealed seven-class VideoMAE best.pt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.aligned_protocol import DATA_PROTOCOL, EXPECTED_COUNTS, EXPECTED_MANIFEST_SHA
from experiments.metrics import compute_classification_metrics
from experiments.modern_baselines import videomae2 as base
from experiments.modern_baselines import train_videomae2 as trainer
from experiments.relation.data import load_edges, load_feature_index, read_jsonl, sha256_file
from experiments.relation.decoder import fit_transition
from experiments.relation.protocol import SOURCE_POLICY, validate_source_artifacts


SCHEMA = "fsn-videomae2-source-relation-evidence-v1"
POSITION_BASIS = "sampled_cache_tubelet_center_fraction_not_verified_pts"
SAMPLING = "36f_cache_to_fixed_16f_view_to_8x2_tubelets_14x14_patches"
TUBELETS = base.INPUT_FRAMES // 2
PATCHES_PER_TUBELET = (base.INPUT_SIZE // 16) ** 2
CHANNELS = 768


def tubelet_positions() -> np.ndarray:
    """Positions of eight *sampled* two-frame tubelets in the 36-frame cache."""
    indices = np.asarray(base.FRAME_INDICES, dtype=np.float32).reshape(TUBELETS, 2)
    return indices.mean(axis=1) / np.float32(base.CACHE_FRAMES - 1)


def summarize_tubelets(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Late ViT patch mean (scene) and max (salient patch) for each tubelet."""
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or \
            tokens.shape[0] < 1 or tuple(tokens.shape[1:]) != \
            (TUBELETS * PATCHES_PER_TUBELET, CHANNELS) or \
            not bool(torch.isfinite(tokens).all()):
        raise base.BaselinePreflightError("unexpected final ViT patch-token geometry")
    arranged = tokens.reshape(tokens.shape[0], TUBELETS, PATCHES_PER_TUBELET, CHANNELS)
    return arranged.mean(dim=2), arranged.amax(dim=2)


def require_exact_A_replay(exported: torch.Tensor, replay: torch.Tensor) -> None:
    """Every val logit must equal an independent best.pt forward on that batch."""
    if not isinstance(exported, torch.Tensor) or not isinstance(replay, torch.Tensor) or \
            exported.ndim != 2 or exported.shape[1] != 7 or \
            exported.shape != replay.shape or not torch.equal(exported, replay):
        raise base.BaselinePreflightError(
            "a val clip's exported logits differ from separate best.pt inference")


class FrozenVideoMAE2Evidence(nn.Module):
    """Capture the final block once while calling the unchanged A forward path.

    ``global_tokens`` are spatial means and ``local_tokens`` spatial maxima of
    the same late-layer patch tokens. Both have shape [B,8,768]. The R endpoint
    branch sees sampled tubelet centers, not original video edit boundaries.
    Action logits and class labels are never placed in R's four input arrays.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        if not getattr(model, "_fsn_finetuned_checkpoint_sha256", None):
            raise base.BaselinePreflightError("sealed FSN VideoMAE best.pt must be loaded first")
        if getattr(model.patch_embed, "num_patches", None) != TUBELETS * PATCHES_PER_TUBELET or \
                getattr(model.patch_embed, "tubelet_size", None) != 2 or \
                getattr(model.patch_embed, "patch_size", None) != (16, 16) or \
                getattr(model, "num_features", None) != CHANNELS or \
                len(model.blocks) != 12 or getattr(model, "with_cp", None) is not False or \
                model.fc_norm is None or not isinstance(model.norm, nn.Identity) or \
                getattr(model.head, "out_features", None) != 7:
            raise base.BaselinePreflightError("trained ViT architecture differs from fixed mean-pool ViT-B/16")
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    @torch.inference_mode()
    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        if not isinstance(video, torch.Tensor) or video.ndim != 5 or \
                tuple(video.shape[1:]) != (3, base.INPUT_FRAMES, base.INPUT_SIZE, base.INPUT_SIZE) or \
                not video.is_floating_point() or not bool(torch.isfinite(video).all()):
            raise base.BaselinePreflightError("VideoMAE evidence needs finite [B,3,16,224,224] input")
        self.model.eval()
        captured: list[torch.Tensor] = []

        def capture(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor):
                raise base.BaselinePreflightError("final ViT block did not return patch tokens")
            captured.append(output.detach().clone())

        handle = self.model.blocks[-1].register_forward_hook(capture)
        try:
            logits = self.model(video)
        finally:
            handle.remove()
        if len(captured) != 1 or tuple(logits.shape) != (video.shape[0], 7) or \
                not bool(torch.isfinite(logits).all()):
            raise base.BaselinePreflightError("expected one finite A forward with seven logits")
        tokens = captured[0]
        scene, salient = summarize_tubelets(tokens)
        # A itself pools all 1568 late patch tokens before fc_norm and head.
        replay = self.model.head(self.model.head_dropout(self.model.fc_norm(tokens.mean(dim=1))))
        if not torch.allclose(logits.float(), replay.float(), rtol=1e-3, atol=1e-3):
            raise base.BaselinePreflightError("captured ViT tokens do not reconstruct A logits")
        positions = torch.as_tensor(tubelet_positions(), device=logits.device).expand(video.shape[0], -1)
        result = {"logits": logits.detach(), "global_tokens": scene,
                  "local_tokens": salient, "global_positions": positions,
                  "local_positions": positions.clone()}
        if any(not bool(torch.isfinite(value).all()) for value in result.values()):
            raise base.BaselinePreflightError("nonfinite VideoMAE evidence")
        return result


def _fresh_output(path: Path, protected: tuple[Path, ...]) -> Path:
    path = Path(path).resolve()
    if path.exists() or not path.parent.is_dir():
        raise FileExistsError("output must be a fresh directory with an existing parent")
    if path == Path(path.anchor) or any(path == root or path.is_relative_to(root)
                                              for root in (Path(x).resolve() for x in protected)):
        raise ValueError("output must be outside source code, inputs, and prior results")
    return path


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _check_metadata(datasets, source_dir: Path) -> dict[str, dict]:
    rows = read_jsonl(source_dir / "manifest_metadata.jsonl")
    metadata = {row["clip_id"]: row for row in rows}
    if len(metadata) != sum(EXPECTED_COUNTS.values()) or len(rows) != len(metadata):
        raise ValueError("source metadata must cover all 8195 unique development clips")
    expected_ids = set()
    for split in ("train", "val"):
        for record in datasets[split].base.records:
            expected_ids.add(record.clip_id)
            row = metadata.get(record.clip_id)
            if row is None or any(row.get(key) != value for key, value in (
                    ("split", split), ("group_id", record.group_id),
                    ("label_id", record.label_id),
                    ("source_collection", record.source_collection),
                    ("source_video_key", hashlib.sha256(record.video_path.encode()).hexdigest()))):
                raise ValueError("source metadata differs from fixed VideoMAE manifest")
            for key, value in (("clip_start_sec", record.clip_start_sec),
                               ("clip_end_sec", record.clip_end_sec),
                               ("duration", record.clip_end_sec - record.clip_start_sec)):
                number = row.get(key)
                if isinstance(number, bool) or not isinstance(number, (int, float)) or \
                        not math.isfinite(number) or not math.isclose(number, value, abs_tol=1e-6):
                    raise ValueError(f"source metadata {key} differs from fixed VideoMAE manifest")
    if expected_ids != metadata.keys():
        raise ValueError("source metadata includes non-development clips")
    return metadata


def _check_completed_training_run(finetuned: Path, checkpoint: dict, digest: str) -> None:
    """Verify best.pt against the training runner's completed sidecars."""
    root = finetuned.parent
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    identity = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    if status.get("status") != "completed" or status.get("exit_code") != 0 or \
            result.get("schema") != trainer.RUN_SCHEMA or \
            result.get("training_completed") is not True or \
            result.get("best_checkpoint_sha256") != digest or \
            result.get("identity_sha256") != checkpoint.get("run_identity_sha256") or \
            identity.get("identity_sha256") != checkpoint.get("run_identity_sha256") or \
            identity.get("source_sha256") != trainer._source_seal() or \
            identity.get("data", {}).get("manifest_sha256") != EXPECTED_MANIFEST_SHA or \
            result.get("best_epoch") != checkpoint.get("epoch") or \
            result.get("best_val_macro_f1") != checkpoint.get("val_macro_f1"):
        raise base.BaselinePreflightError("completed VideoMAE run sidecars disagree with best.pt")


def export(args: argparse.Namespace) -> dict[str, Any]:
    """Export complete frozen A evidence; interrupted output remains unusable."""
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    official_source, distilled, finetuned = (Path(x).resolve() for x in (
        args.official_source, args.distilled_checkpoint, args.finetuned_checkpoint))
    manifest_dir, cache_dir, source_dir = (Path(x).resolve() for x in (
        args.manifest_dir, args.cache_dir, args.source_protocol_dir))
    repo = Path(__file__).resolve().parents[2]
    output = _fresh_output(args.output, (repo, official_source, distilled.parent,
                                         finetuned.parent, manifest_dir, cache_dir, source_dir))
    if finetuned.name != "best.pt":
        raise base.BaselinePreflightError("A input must be completed VideoMAE best.pt")
    base.verify_official_source(official_source)
    base.verify_official_checkpoint(distilled)
    source_audit = validate_source_artifacts(source_dir)
    datasets, data_audit = base.audit_frozen_data(manifest_dir, cache_dir)
    if source_audit.get("manifest_sha256") != data_audit["manifest_sha256"] or \
            source_audit.get("counts") != data_audit["counts"]:
        raise ValueError("source policy and VideoMAE use different train/val manifests")
    metadata = _check_metadata(datasets, source_dir)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise base.BaselinePreflightError("formal VideoMAE evidence export requires CUDA bf16")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise base.BaselinePreflightError("selected CUDA device lacks bf16 support")
    model, pretrained_audit = base.build_pretrained_model(official_source, distilled)
    trained_audit = base.load_finetuned_fsn_checkpoint(model, finetuned, args.finetuned_sha256)
    with finetuned.open("rb") as stream:
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    if base.sha256_file(finetuned) != args.finetuned_sha256 or \
            checkpoint.get("selection") != trainer.SELECTION or \
            not isinstance(checkpoint.get("val_macro_f1"), (float, int)) or \
            not math.isfinite(checkpoint["val_macro_f1"]):
        raise base.BaselinePreflightError("A selection record is missing or changed")
    _check_completed_training_run(finetuned, checkpoint, args.finetuned_sha256)
    frozen = FrozenVideoMAE2Evidence(model.to(device))
    code_files = (Path(__file__), Path(base.__file__), Path(trainer.__file__),
                  repo / "experiments" / "aligned_data.py",
                  repo / "experiments" / "aligned_protocol.py",
                  repo / "experiments" / "relation" / "data.py",
                  repo / "experiments" / "relation" / "network.py",
                  repo / "experiments" / "relation" / "decoder.py",
                  repo / "experiments" / "relation" / "evaluate_suite.py")
    protocol = {"schema": SCHEMA, "status": "exporting", "A_frozen": True,
                "checkpoint_sha256": args.finetuned_sha256,
                "pretrained_checkpoint_sha256": base.OFFICIAL_CHECKPOINT_SHA256,
                "official_source_commit": base.OFFICIAL_COMMIT,
                "official_model_sha256": base.OFFICIAL_MODEL_SHA256,
                "data_protocol": DATA_PROTOCOL, "label_policy": SOURCE_POLICY,
                "source_audit_sha256": sha256_file(source_dir / "audit.json"),
                "train_manifest_sha256": EXPECTED_MANIFEST_SHA["train"],
                "val_manifest_sha256": EXPECTED_MANIFEST_SHA["val"],
                "position_basis": POSITION_BASIS, "sampling": SAMPLING,
                "global_tokens": "last_block_spatial_mean_per_sampled_tubelet",
                "local_tokens": "last_block_spatial_max_per_sampled_tubelet",
                "tubelet_positions": tubelet_positions().tolist(),
                "test_read": False, "validation_role": "val823_development",
                "contains_private_clip_identifiers": True,
                "code_sha256": {str(path.relative_to(repo)): sha256_file(path) for path in code_files},
                "data_audit": data_audit, "pretrained_audit": pretrained_audit,
                "trained_audit": trained_audit,
                "val_per_clip_A_logits_equal_separate_best_replay": True}
    output.mkdir()
    (output / "features").mkdir()
    _write_json(output / "protocol.json", protocol)
    val_scores, val_targets, val_metadata = [], [], []
    seen = set()
    with (output / "index.jsonl").open("x", encoding="utf-8") as index_stream:
        for split in ("train", "val"):
            dataset = datasets[split]
            loader = DataLoader(dataset, batch_size=4, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
            count = 0
            for batch in loader:
                video = batch["video"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    values = frozen(video)
                    if split == "val":
                        with torch.inference_mode():
                            replay_logits = frozen.model(video)
                if split == "val":
                    require_exact_A_replay(values["logits"], replay_logits)
                if split == "val":
                    val_scores.append(values["logits"].float().cpu())
                    val_targets.append(batch["label"].cpu())
                    val_metadata.extend({"source_collection": source, "duration": float(duration)}
                                        for source, duration in zip(batch["source"], batch["duration"]))
                for offset, clip_id in enumerate(batch["clip_id"]):
                    if clip_id in seen or dataset.base.records[count].clip_id != clip_id:
                        raise ValueError("duplicate or reordered development clip during export")
                    seen.add(clip_id)
                    name = hashlib.sha256(clip_id.encode()).hexdigest() + ".npz"
                    relative = "features/" + name
                    target = output / relative
                    with target.open("xb") as stream:
                        np.savez_compressed(stream, **{key: value[offset].float().cpu().numpy()
                                                       for key, value in values.items()})
                    source = metadata[clip_id]
                    row = {"clip_id": clip_id, "group_id": batch["group_id"][offset],
                           "split": split, "feature_path": relative,
                           "feature_sha256": sha256_file(target),
                           "checkpoint_sha256": args.finetuned_sha256,
                           "position_basis": POSITION_BASIS,
                           "label_id": int(batch["label"][offset]),
                           "source_collection": batch["source"][offset],
                           "source_video_key": source["source_video_key"],
                           "clip_start_sec": source["clip_start_sec"],
                           "clip_end_sec": source["clip_end_sec"],
                           "duration": source["duration"]}
                    index_stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    count += 1
                index_stream.flush()
            if count != EXPECTED_COUNTS[split]:
                raise ValueError("incomplete fixed split export")
            print(json.dumps({"split": split, "exported": count}), flush=True)
    if len(seen) != sum(EXPECTED_COUNTS.values()):
        raise ValueError("export did not cover exactly train7372/val823")
    val_metrics = compute_classification_metrics(torch.cat(val_scores), torch.cat(val_targets), val_metadata)
    if val_metrics["all"]["macro_f1"] != float(checkpoint["val_macro_f1"]):
        raise base.BaselinePreflightError("exported val823 A predictions differ from best.pt selection replay")
    if base.sha256_file(finetuned) != args.finetuned_sha256:
        raise base.BaselinePreflightError("A checkpoint changed during feature export")
    protocol.update({"status": "complete", "index_sha256": sha256_file(output / "index.jsonl"),
                     "split_counts": EXPECTED_COUNTS, "A_val_metrics": val_metrics["all"]})
    _write_json(output / "protocol.json", protocol)
    exported_index = load_feature_index(output / "index.jsonl")
    validate_source_artifacts(source_dir, exported_index, source_dir / "edges.jsonl")
    return {"index": str(output / "index.jsonl"), "checkpoint_sha256": args.finetuned_sha256,
            "counts": EXPECTED_COUNTS, "A_val_macro_f1": val_metrics["all"]["macro_f1"]}


def rebind_transition(args: argparse.Namespace) -> dict[str, Any]:
    """Copy D's exact bytes after proving it is train clinical-C derived."""
    index_path, original_path = Path(args.index).resolve(), Path(args.reference_index).resolve()
    source_dir, reference_dir = Path(args.source_protocol_dir).resolve(), Path(args.reference_transition).resolve()
    repo = Path(__file__).resolve().parents[2]
    output = _fresh_output(args.output, (repo, index_path.parent, original_path.parent,
                                         source_dir, reference_dir))
    new_index, original_index = load_feature_index(index_path), load_feature_index(original_path)
    edges_path = source_dir / "edges.jsonl"
    source_audit = validate_source_artifacts(source_dir, new_index, edges_path)
    validate_source_artifacts(source_dir, original_index, edges_path)
    new_seal = json.loads((index_path.parent / "protocol.json").read_text())
    if new_seal.get("schema") != SCHEMA or new_seal.get("status") != "complete" or \
            new_seal.get("checkpoint_sha256") != new_index.checkpoint_sha256 or \
            new_seal.get("source_audit_sha256") != sha256_file(source_dir / "audit.json"):
        raise ValueError("new feature index is not a completed sealed VideoMAE export")
    reference_audit = json.loads((reference_dir / "audit.json").read_text())
    expected = {"index_sha256": sha256_file(original_path),
                "edges_sha256": sha256_file(edges_path),
                "checkpoint_sha256": original_index.checkpoint_sha256,
                "label_policy": SOURCE_POLICY,
                "source_audit_sha256": sha256_file(source_dir / "audit.json"),
                "fit_split": "train", "trusted_edge_status": "C",
                "data_protocol": DATA_PROTOCOL}
    if any(reference_audit.get(key) != value for key, value in expected.items()):
        raise ValueError("reference D is not bound to the same Original source-policy protocol")
    edges = load_edges(edges_path, new_index)
    labels = {clip_id: row["label_id"] for clip_id, row in new_index.clips.items()
              if row["split"] == "train"}
    fitted = fit_transition(labels, [row for row in edges if row["split"] == "train"],
                            smoothing=reference_audit["smoothing"])
    if fitted["audit"]["accepted_C"] <= 0 or any(
            reference_audit.get(key) != value for key, value in fitted["audit"].items()):
        raise ValueError("reference D audit differs from train-only clinical transition counts")
    source_file = reference_dir / "transition.npz"
    with np.load(source_file, allow_pickle=False) as stored:
        if set(stored.files) != {"transition", "counts"} or \
                not np.array_equal(stored["counts"], fitted["counts"]) or \
                not np.allclose(stored["transition"], fitted["transition"], rtol=0, atol=1e-12):
            raise ValueError("reference D matrix differs from this split's clinical training edges")
    output.mkdir()
    shutil.copyfile(source_file, output / "transition.npz")
    if sha256_file(source_file) != sha256_file(output / "transition.npz"):
        raise IOError("D table copy changed bytes")
    audit = {**reference_audit,
             "index_sha256": sha256_file(index_path),
             "checkpoint_sha256": new_index.checkpoint_sha256,
             "reference_index_sha256": expected["index_sha256"],
             "reference_checkpoint_sha256": original_index.checkpoint_sha256,
             "reference_transition_sha256": sha256_file(source_file),
             "transition_reused_exact_bytes": True,
             "held_out_edges_excluded": len(edges) - fitted["audit"]["edge_count"],
             "videomae_export_protocol_sha256": sha256_file(index_path.parent / "protocol.json"),
             "source_manifest_sha256": source_audit["manifest_sha256"]}
    _write_json(output / "audit.json", audit)
    return {"transition": str(output / "transition.npz"),
            "transition_sha256": sha256_file(source_file),
            "accepted_train_clinical_C": fitted["audit"]["accepted_C"],
            "reused_exact_bytes": True}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporting = commands.add_parser("export", help="export frozen train/val VideoMAE evidence")
    for name in ("official-source", "distilled-checkpoint", "manifest-dir", "cache-dir",
                 "finetuned-checkpoint", "finetuned-sha256", "source-protocol-dir", "output"):
        exporting.add_argument("--" + name, required=True)
    exporting.add_argument("--device", default="cuda:0")
    exporting.add_argument("--workers", type=int, default=4)
    rebinding = commands.add_parser("rebind-transition", help="reuse Original D exact bytes")
    for name in ("index", "reference-index", "reference-transition", "source-protocol-dir", "output"):
        rebinding.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    result = export(args) if args.command == "export" else rebind_transition(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
