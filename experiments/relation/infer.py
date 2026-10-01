"""Decode development chains without reading their clinical/action edge labels.

chains.jsonl declares chain_id, ordered_clip_ids, and optional eligible booleans.
Eligibility means same-recording playback adjacency supplied by the caller; it
does not certify clinical continuity. No ordering is inferred from clip IDs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.relation.data import load_feature_index
from experiments.relation.export import sha256
from experiments.relation.network import RelationConfig, RelationNet
from experiments.relation.pipeline import predict_chain


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True)
    p.add_argument("--relation-checkpoint", required=True)
    p.add_argument("--transition", required=True, help="directory from fit_transition")
    p.add_argument("--chains", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--mode", choices=("hard", "soft"), default="hard")
    p.add_argument("--threshold", type=float, default=.9)
    p.add_argument("--strength", type=float, default=1.)
    p.add_argument("--potential-bound", type=float)
    args = p.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("inference output must be a fresh directory")
    index = load_feature_index(args.index)
    index_sha = sha256(args.index)
    checkpoint = torch.load(args.relation_checkpoint, map_location="cpu", weights_only=True)
    protocol = checkpoint["protocol"]
    if protocol["checkpoint_sha256"] != index.checkpoint_sha256:
        raise ValueError("R and feature index were generated with different A checkpoints")
    if protocol["index_sha256"] != index_sha:
        raise ValueError("development inference must use the exact feature index used to train R")
    relation = RelationNet(RelationConfig(**checkpoint["config"])).to(args.device)
    relation.load_state_dict(checkpoint["model"], strict=True)
    transition_dir = Path(args.transition)
    audit = json.loads((transition_dir / "audit.json").read_text())
    if audit["checkpoint_sha256"] != index.checkpoint_sha256:
        raise ValueError("transition table belongs to a different A evidence protocol")
    if audit["index_sha256"] != index_sha or audit["edges_sha256"] != protocol["edges_sha256"]:
        raise ValueError("transition table and R must share the fixed index/annotation protocol")
    with np.load(transition_dir / "transition.npz", allow_pickle=False) as data:
        transition = data["transition"].copy()
    rows, seen = [], set()
    for line in Path(args.chains).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        chain_id, ids = row.get("chain_id"), row.get("ordered_clip_ids")
        if not isinstance(chain_id, str) or not chain_id or chain_id in seen:
            raise ValueError("chain_id must be unique and nonempty")
        seen.add(chain_id)
        if (not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids)
                or len(set(ids)) != len(ids) or any(x not in index.clips for x in ids)):
            raise ValueError("invalid explicit clip order")
        if any(index.clips[x]["split"] != "val" for x in ids):
            raise ValueError("this prototype inference CLI only evaluates development val clips")
        eligible = row.get("eligible", [False] * (len(ids) - 1))
        if not isinstance(eligible, list) or len(eligible) != len(ids) - 1 or any(
                not isinstance(x, bool) for x in eligible):
            raise ValueError("invalid structural eligibility")
        for i, flag in enumerate(eligible):
            if flag and index.clips[ids[i]]["group_id"] != index.clips[ids[i + 1]]["group_id"]:
                raise ValueError("eligible playback neighbors must declare the same recording group")
        features = [index.read(clip_id) for clip_id in ids]
        result = predict_chain(relation, features, transition, eligible=eligible,
                               mode=args.mode, threshold=args.threshold, strength=args.strength,
                               potential_bound=args.potential_bound)
        rows.append({"chain_id": chain_id, "clip_ids": ids, **result})
    if not rows:
        raise ValueError("no candidate chains provided")
    out.mkdir(parents=True)
    with (out / "predictions.jsonl").open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "protocol.json").write_text(json.dumps({
        "evaluation_role": "development_validation_not_independent_test",
        "contains_private_clip_identifiers": True,
        "index_sha256": sha256(args.index), "chains_sha256": sha256(args.chains),
        "relation_checkpoint_sha256": sha256(args.relation_checkpoint),
        "transition_sha256": sha256(transition_dir / "transition.npz"),
        "threshold": args.threshold, "mode": args.mode, "strength": args.strength,
        "potential_bound": args.potential_bound, "position_basis": "cache_index_fraction_not_verified_pts",
    }, indent=2))


if __name__ == "__main__":
    main()
