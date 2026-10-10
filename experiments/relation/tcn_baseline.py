"""Offline A+TCN control on the sealed, frozen Original evidence.

The TCN reads the same per-clip Original global/local tokens and seven logits
as the relation suite. It is trained with action labels from *train* only.
Convolutions stop at every ineligible source-protocol adjacency, so neither
training nor inference sees another recording or a clip across a structural
cut. Bidirectional temporal convolutions make this an offline method.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .data import load_edges, load_feature_index, read_jsonl, sha256_file
from .evaluate_suite import _directions, _paired, classification_metrics
from .protocol import SOURCE_POLICY, validate_source_artifacts


@dataclass(frozen=True)
class TCNConfig:
    global_dim: int
    local_dim: int
    hidden_dim: int = 64
    blocks: int = 3
    dropout: float = .1
    # None preserves the historical powers-of-two schedule and serialized
    # evidence schema. Explicit schedules enable controlled local-context tests.
    dilations: tuple[int, ...] | None = None

    def __post_init__(self):
        if min(self.global_dim, self.local_dim, self.hidden_dim, self.blocks) < 1:
            raise ValueError("TCN dimensions and blocks must be positive")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if self.dilations is not None:
            try:
                schedule = tuple(self.dilations)
            except TypeError as error:
                raise ValueError("dilations must contain one positive integer per block") from error
            if (len(schedule) != self.blocks
                    or any(type(value) is not int or value < 1 for value in schedule)):
                raise ValueError("dilations must contain one positive integer per block")
            object.__setattr__(self, "dilations", schedule)

    @property
    def dilation_schedule(self):
        return self.dilations if self.dilations is not None else tuple(
            2**level for level in range(self.blocks))

    @property
    def receptive_field_clips(self):
        # Each width-three convolution adds 2*dilation to the receptive field.
        return 1 + 2 * sum(self.dilation_schedule)


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        update = self.mix(self.dropout(F.gelu(self.conv(sequence))))
        # Mask after every block: padding cannot leak back into a valid clip.
        return (sequence + update) * valid[:, None, :]


class FrozenEvidenceTCN(nn.Module):
    """A logits plus a trainable TCN correction; initial output equals A."""

    def __init__(self, config: TCNConfig):
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(nn.Linear(config.global_dim + config.local_dim + 7,
                                            config.hidden_dim),
                                  nn.LayerNorm(config.hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList(TemporalBlock(config.hidden_dim, dilation, config.dropout)
                                    for dilation in config.dilation_schedule)
        self.classifier = nn.Conv1d(config.hidden_dim, 7, 1)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features: torch.Tensor, a_logits: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        if (features.ndim != 3 or a_logits.shape != (*features.shape[:2], 7)
                or valid.shape != features.shape[:2]
                or features.shape[-1] != self.config.global_dim + self.config.local_dim + 7):
            raise ValueError("TCN input shapes do not match its frozen evidence config")
        if valid.dtype != torch.bool or not bool(valid.any(dim=1).all()):
            raise ValueError("every sequence must have at least one valid clip")
        if not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(a_logits).all()):
            raise ValueError("frozen evidence must be finite")
        hidden = self.stem(features).transpose(1, 2) * valid[:, None, :]
        for block in self.blocks:
            hidden = block(hidden, valid)
        delta = self.classifier(hidden).transpose(1, 2)
        return a_logits + delta * valid[:, :, None]


@dataclass
class Segment:
    clip_ids: list[str]
    features: np.ndarray
    a_logits: np.ndarray
    labels: np.ndarray
    source_kinds: list[str]


def build_segments(index, metadata_rows, chains, edges, split: str) -> list[Segment]:
    """Validate sealed adjacency and partition every split clip exactly once."""
    if split not in {"train", "val"}:
        raise ValueError("only train and val are permitted")
    metadata = {row["clip_id"]: row for row in metadata_rows}
    if len(metadata) != len(metadata_rows) or metadata.keys() != index.clips.keys():
        raise ValueError("source metadata must cover the index exactly once")
    expected = {clip for clip, row in index.clips.items() if row["split"] == split}
    edge_pairs = {}
    for edge in edges:
        if edge["split"] != split:
            continue
        pair = (edge["left_clip_id"], edge["right_clip_id"])
        if pair in edge_pairs or edge.get("label_origin") != SOURCE_POLICY:
            raise ValueError("duplicate or non-source-policy candidate edge")
        edge_pairs[pair] = edge["status"]
    seen, used_edges, segments, loaded, chain_ids = set(), set(), [], {}, set()

    def append_segment(ids):
        if not ids:
            return
        features, logits, labels, source_kinds = [], [], [], []
        for clip in ids:
            if clip not in loaded:
                evidence = index.read(clip)
                vector = np.concatenate((evidence["global_tokens"].mean(axis=0),
                                         evidence["local_tokens"].mean(axis=0),
                                         evidence["logits"])).astype(np.float32)
                loaded[clip] = (vector, np.asarray(evidence["logits"], dtype=np.float32))
            vector, a_logit = loaded[clip]
            features.append(vector)
            logits.append(a_logit)
            labels.append(metadata[clip]["label_id"])
            source_kinds.append(metadata[clip]["source_kind"])
        segments.append(Segment(list(ids), np.stack(features), np.stack(logits),
                                np.asarray(labels, dtype=np.int64), source_kinds))

    for chain in chains:
        if chain.get("split") != split:
            raise ValueError("chain split differs from requested split")
        chain_id = chain.get("chain_id")
        if not isinstance(chain_id, str) or not chain_id or chain_id in chain_ids:
            raise ValueError("chain IDs must be unique")
        chain_ids.add(chain_id)
        ordered, eligible = chain.get("ordered_clip_ids"), chain.get("eligible")
        if (not isinstance(ordered, list) or not ordered or
                not isinstance(eligible, list) or len(eligible) != len(ordered) - 1 or
                any(type(flag) is not bool for flag in eligible)):
            raise ValueError("chain must have explicit ordered clips and adjacency mask")
        if len(set(ordered)) != len(ordered) or seen.intersection(ordered):
            raise ValueError("clip appears more than once in source chains")
        for clip in ordered:
            if clip not in expected or clip not in metadata:
                raise ValueError("chain includes an unknown or cross-split clip")
            row, indexed = metadata[clip], index.clips[clip]
            if (row["split"], row["group_id"], row["label_id"]) != (
                    indexed["split"], indexed["group_id"], indexed["label_id"]):
                raise ValueError("source metadata and A index disagree")
            if row.get("source_kind") not in {"clinical", "network"}:
                raise ValueError("unmapped source kind")
        seen.update(ordered)
        current = [ordered[0]]
        for left, right, flag in zip(ordered, ordered[1:], eligible):
            pair = (left, right)
            if flag:
                lhs, rhs = metadata[left], metadata[right]
                if any(lhs.get(key) != rhs.get(key) for key in (
                        "group_id", "source_video_key", "timebase_key", "recording_key", "source_kind")):
                    raise ValueError("candidate joins different recordings, clocks or source kinds")
                expected_status = "C" if lhs["source_kind"] == "clinical" else "D"
                if edge_pairs.get(pair) != expected_status:
                    raise ValueError("candidate edge and chain mask disagree")
                used_edges.add(pair)
                current.append(right)
            else:
                if pair in edge_pairs:
                    raise ValueError("structural cut must not contain an edge")
                append_segment(current)
                current = [right]
        append_segment(current)
    if seen != expected or used_edges != edge_pairs.keys():
        raise ValueError("chains/edges must cover all and only this split's clips/candidates")
    if sum(len(segment.clip_ids) for segment in segments) != len(expected):
        raise AssertionError("segment partition did not cover the split exactly")
    return segments


class SegmentDataset(Dataset):
    def __init__(self, segments):
        self.segments = segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, position):
        return self.segments[position]


def collate_segments(segments):
    length = max(len(item.clip_ids) for item in segments)
    batch = len(segments)
    dim = segments[0].features.shape[-1]
    features = torch.zeros(batch, length, dim, dtype=torch.float32)
    a_logits = torch.zeros(batch, length, 7, dtype=torch.float32)
    labels = torch.full((batch, length), -100, dtype=torch.long)
    valid = torch.zeros(batch, length, dtype=torch.bool)
    for row, item in enumerate(segments):
        count = len(item.clip_ids)
        features[row, :count] = torch.from_numpy(item.features)
        a_logits[row, :count] = torch.from_numpy(item.a_logits)
        labels[row, :count] = torch.from_numpy(item.labels)
        valid[row, :count] = True
    return features, a_logits, labels, valid, segments


def _forward_loss(model, batch, device):
    features, a_logits, labels, valid, _ = batch
    features, a_logits, labels, valid = (item.to(device) for item in (
        features, a_logits, labels, valid))
    prediction = model(features, a_logits, valid)
    per_clip = F.cross_entropy(prediction.transpose(1, 2), labels, reduction="none",
                               ignore_index=-100)
    return per_clip[valid].mean(), prediction, valid


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, predictions, visual, sources, private = [], [], [], [], []
    for batch in loader:
        _, score, true, valid, segments = batch
        _, output, _ = _forward_loss(model, batch, device)
        predicted = output.argmax(dim=-1).cpu().numpy()
        original = score.argmax(dim=-1).numpy()
        for row, segment in enumerate(segments):
            count = int(valid[row].sum())
            if count != len(segment.clip_ids):
                raise AssertionError("batch mask differs from segment length")
            labels.extend(true[row, :count].tolist())
            predictions.extend(predicted[row, :count].tolist())
            visual.extend(original[row, :count].tolist())
            sources.extend(segment.source_kinds)
            private.extend({"clip_id": clip, "label_id": int(true[row, column]),
                            "a_prediction": int(original[row, column]),
                            "tcn_prediction": int(predicted[row, column])}
                           for column, clip in enumerate(segment.clip_ids))
    labels, predictions, visual, sources = map(np.asarray, (labels, predictions, visual, sources))
    result = classification_metrics(labels, predictions)
    result.update(paired_vs_A=_paired(labels, predictions, visual),
                  sweep_reperfusion=_directions(labels, predictions))
    result["slices"] = {kind: classification_metrics(labels[sources == kind],
                                                      predictions[sources == kind])
                        for kind in ("clinical", "network")}
    return result, private


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _validate_args(args):
    if any(getattr(args, name) < 1 for name in ("epochs", "patience", "batch_size", "hidden_dim", "blocks")):
        raise ValueError("positive epochs, patience, batch size and TCN size are required")
    if args.dilations is not None and len(args.dilations) != args.blocks:
        raise ValueError("--dilations must provide exactly one value per TCN block")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("invalid optimizer settings")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("TCN output already exists; choose a fresh private directory")
    return device, output


def train(args):
    device, output = _validate_args(args)
    index = load_feature_index(args.index)
    protocol_dir = Path(args.source_protocol_dir)
    audit = validate_source_artifacts(protocol_dir, index, protocol_dir / "edges.jsonl")
    if audit.get("label_policy") != SOURCE_POLICY:
        raise ValueError("TCN requires the same sealed source protocol as R+D")
    metadata = read_jsonl(protocol_dir / "manifest_metadata.jsonl")
    edges = load_edges(protocol_dir / "edges.jsonl", index)
    train_segments = build_segments(index, metadata, read_jsonl(protocol_dir / "chains_train.jsonl"),
                                    edges, "train")
    val_segments = build_segments(index, metadata, read_jsonl(protocol_dir / "chains_val.jsonl"),
                                  edges, "val")
    if not train_segments or not val_segments:
        raise ValueError("both train and val segments are required")
    suite = protocol_dir.parent
    prior_seal = json.loads((suite / "evaluation" / "private_protocol.json").read_text())
    prior_eval = json.loads((suite / "evaluation" / "public_safe_summary.json").read_text())
    if (prior_seal.get("index_sha256") != sha256_file(args.index)
            or prior_seal.get("source_audit_sha256") != sha256_file(protocol_dir / "audit.json")
            or prior_seal.get("checkpoint_sha256") != index.checkpoint_sha256
            or prior_seal.get("chains_sha256") != sha256_file(protocol_dir / "chains_val.jsonl")):
        raise ValueError("TCN inputs differ from the completed frozen-A suite")
    a_labels = np.concatenate([segment.labels for segment in val_segments])
    a_predictions = np.concatenate([segment.a_logits.argmax(1) for segment in val_segments])
    baseline = classification_metrics(a_labels, a_predictions)
    for key in ("macro_f1", "accuracy", "confusion"):
        if baseline[key] != prior_eval["variants"]["A_only"][key]:
            raise ValueError(f"TCN A-only replay differs from the sealed suite: {key}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    config = TCNConfig(index.global_dim, index.local_dim, args.hidden_dim, args.blocks,
                        args.dropout, args.dilations)
    model = FrozenEvidenceTCN(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(SegmentDataset(train_segments), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), collate_fn=collate_segments)
    val_loader = DataLoader(SegmentDataset(val_segments), batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_segments)
    seals = {"index_sha256": sha256_file(args.index),
             "source_audit_sha256": sha256_file(protocol_dir / "audit.json"),
             "checkpoint_sha256": index.checkpoint_sha256,
             "train_manifest_sha256": audit["manifest_sha256"]["train"],
             "val_manifest_sha256": audit["manifest_sha256"]["val"],
             "script_sha256": sha256_file(__file__)}
    private_protocol = {"input_paths": {"index": str(Path(args.index).resolve()),
                                         "source_protocol_dir": str(protocol_dir.resolve())},
                        "output_path": str(output.resolve()), "seals": seals,
                        "config": asdict(config), "optimizer": {"name": "AdamW", "lr": args.lr,
                                                                   "weight_decay": args.weight_decay},
                        "selection": "best val823 macro-F1; ties keep earlier checkpoint",
                        "seed": args.seed, "batch_size_segments": args.batch_size,
                        "max_epochs": args.epochs, "patience": args.patience,
                        "train_only_supervision": True, "test_evaluated": False,
                        "A_frozen": True, "offline_bidirectional_context": True,
                        "representation": "mean(global_tokens), mean(local_tokens), A logits",
                        "adjacency": "source-protocol candidate chains split at every ineligible mask",
                        "contains_private_paths": True}
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "private_protocol.json", private_protocol)
    history, best, stale, start = [], -math.inf, 0, time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, count = 0., 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _, valid = _forward_loss(model, batch, device)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite TCN training loss")
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise RuntimeError("nonfinite TCN gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            examples = int(valid.sum())
            loss_sum += float(loss.detach()) * examples
            count += examples
        val, _ = evaluate(model, val_loader, device)
        value = val["macro_f1"]
        history.append({"epoch": epoch, "train_cross_entropy": loss_sum / count,
                        "val_macro_f1": value, "val_accuracy": val["accuracy"]})
        if value > best + 1e-12:
            best, stale = value, 0
            temporary = output / "best.pt.tmp"
            torch.save({"model": model.state_dict(), "config": asdict(config),
                        "protocol": seals, "epoch": epoch, "val_macro_f1": value}, temporary)
            temporary.replace(output / "best.pt")
        else:
            stale += 1
        _write_json(output / "history.json", history)
        print(f"epoch {epoch}: train CE {loss_sum / count:.6f}; val macro-F1 {value:.6f}; "
              f"best {best:.6f}", flush=True)
        if stale >= args.patience:
            break
    saved = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model"], strict=True)
    metric, private = evaluate(model, val_loader, device)
    a_labels = np.asarray([row["label_id"] for row in private])
    a_pred = np.asarray([row["a_prediction"] for row in private])
    summary = {"evaluation_role": "val823_development_not_independent_test",
               "A_frozen_same_logits_as_relation_suite": True,
               "offline_bidirectional_context": True,
               "source_policy_is_not_continuity_ground_truth": True,
               "train_clip_count": sum(len(s.clip_ids) for s in train_segments),
               "val_clip_count": len(private),
               "train_segments": len(train_segments), "val_segments": len(val_segments),
               "dilations": list(config.dilation_schedule),
               "receptive_field_clips": config.receptive_field_clips,
               "trainable_parameters": sum(p.numel() for p in model.parameters()),
               "best_epoch": saved["epoch"], "epochs_completed": len(history),
               "elapsed_seconds": time.monotonic() - start,
               "variants": {"A_only": classification_metrics(a_labels, a_pred),
                            "A_TCN": metric}}
    _write_json(output / "public_safe_summary.json", summary)
    with (output / "private_val_predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row in private:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return summary


def _parse_dilations(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part.strip()) for part in value.split(","))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("dilations must be comma-separated positive integers") from error
    if not result or any(value < 1 for value in result):
        raise argparse.ArgumentTypeError("dilations must be comma-separated positive integers")
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "source-protocol-dir", "output"):
        result.add_argument(f"--{name}", required=True)
    result.add_argument("--device", default="cpu")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--epochs", type=int, default=30)
    result.add_argument("--patience", type=int, default=5)
    result.add_argument("--batch-size", type=int, default=16,
                        help="number of variable-length chain segments per optimizer batch")
    result.add_argument("--hidden-dim", type=int, default=64)
    result.add_argument("--blocks", type=int, default=3)
    result.add_argument("--dilations", type=_parse_dilations, default=None,
                        help="comma-separated positive dilation per block; default is 1,2,4,...")
    result.add_argument("--dropout", type=float, default=.1)
    result.add_argument("--lr", type=float, default=.001)
    result.add_argument("--weight-decay", type=float, default=.01)
    return result


if __name__ == "__main__":
    train(parser().parse_args())
