"""Export frozen Original evidence from existing train/val frame caches only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.aligned_data import ResolvedFullClipDataset
from experiments.aligned_data import load_aligned_manifest
from experiments.aligned_protocol import DATA_PROTOCOL, validate_manifest_protocol
from experiments.relation.evidence import POSITION_BASIS, load_original


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--source-protocol-dir", help="sealed fixed7372/823 source-policy artifacts")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        parser.error("invalid batch size/workers")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("feature output must be a fresh directory")
    if args.source_protocol_dir:
        from .protocol import validate_source_artifacts
        source_audit = validate_source_artifacts(args.source_protocol_dir)
        counts, manifest_hashes = {}, {}
        for split in ("train", "val"):
            manifest = getattr(args, split + "_manifest")
            counts[split] = len(load_aligned_manifest(manifest, split)[0])
            manifest_hashes[split] = sha256(manifest)
        validate_manifest_protocol(counts, manifest_hashes)
        if source_audit["manifest_sha256"] != manifest_hashes:
            raise ValueError("feature export must use the source builder's fixed manifests")
    datasets = {split: ResolvedFullClipDataset(getattr(args, split + "_manifest"),
                 args.cache_root, expected_split=split) for split in ("train", "val")}
    ids = [{row.clip_id for row in datasets[s].records} for s in ("train", "val")]
    groups = [{row.group_id for row in datasets[s].records} for s in ("train", "val")]
    if ids[0] & ids[1] or groups[0] & groups[1]:
        raise ValueError("train/val clip or recording-group leakage")
    checkpoint_sha = sha256(args.checkpoint)
    evidence = load_original(args.checkpoint, args.device,
                             require_full_protocol=bool(args.source_protocol_dir))
    out.mkdir(parents=True)
    (out / "features").mkdir()
    protocol = {"status": "exporting", "checkpoint_sha256": checkpoint_sha,
                "position_basis": POSITION_BASIS, "sampling": "36/8/12/128",
                "train_manifest_sha256": sha256(args.train_manifest),
                "val_manifest_sha256": sha256(args.val_manifest), "seed": 42,
                "A_frozen": True, "contains_private_clip_identifiers": True}
    if args.source_protocol_dir:
        protocol.update(data_protocol=DATA_PROTOCOL, label_policy="source_rule_v1",
                        source_audit_sha256=sha256(Path(args.source_protocol_dir) / "audit.json"))
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    torch.manual_seed(42)
    with (out / "index.jsonl").open("x", encoding="utf-8") as index:
        for split, dataset in datasets.items():
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers)
            count = 0
            for batch in loader:
                features = evidence(batch["video"].to(args.device).float().div_(255))
                for i, clip_id in enumerate(batch["clip_id"]):
                    name = hashlib.sha256(clip_id.encode()).hexdigest() + ".npz"
                    relative = "features/" + name
                    np.savez_compressed(out / relative, **{
                        key: value[i].float().cpu().numpy() for key, value in features.items()})
                    row = {"clip_id": clip_id, "group_id": batch["group_id"][i],
                           "split": split, "feature_path": relative,
                           "feature_sha256": sha256(out / relative),
                           "checkpoint_sha256": checkpoint_sha,
                           "position_basis": POSITION_BASIS,
                           "label_id": int(batch["label"][i]),
                           "source_collection": batch["source"][i]}
                    record = dataset.records[count]
                    if record.clip_id != clip_id:
                        raise RuntimeError("export batch order differs from manifest record order")
                    row.update(source_video_key=hashlib.sha256(record.video_path.encode()).hexdigest(),
                               clip_start_sec=record.clip_start_sec, clip_end_sec=record.clip_end_sec,
                               duration=record.clip_end_sec - record.clip_start_sec)
                    index.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                index.flush()
            print(json.dumps({"split": split, "exported": count}), flush=True)
    protocol["status"] = "complete"
    protocol["index_sha256"] = sha256(out / "index.jsonl")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
