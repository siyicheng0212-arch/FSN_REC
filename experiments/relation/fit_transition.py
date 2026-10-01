"""Fit a seven-class directed table on audited C training edges only."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.relation.data import load_edges, load_feature_index
from experiments.relation.decoder import fit_transition
from experiments.relation.export import sha256


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True)
    p.add_argument("--edges", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--smoothing", type=float, default=1.)
    args = p.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("transition output must be a fresh directory")
    index = load_feature_index(args.index)
    edges = load_edges(args.edges, index)
    train_edges = [row for row in edges if row["split"] == "train"]
    # Action labels here estimate D's table; they never become R's inputs.
    labels = {key: row["label_id"] for key, row in index.clips.items()
              if row["split"] == "train" and "label_id" in row}
    fitted = fit_transition(labels, train_edges, smoothing=args.smoothing)
    if fitted["audit"]["accepted_C"] == 0:
        raise ValueError("no audited C training edges; transition fitting is unavailable")
    out.mkdir(parents=True)
    np.savez(out / "transition.npz", transition=fitted["transition"], counts=fitted["counts"])
    audit = {**fitted["audit"], "index_sha256": sha256(args.index),
             "edges_sha256": sha256(args.edges), "checkpoint_sha256": index.checkpoint_sha256,
             "held_out_edges_excluded": len(edges) - len(train_edges),
             "old_v3_equivalence": "transition-only at q=1, not old initial-prior decoder"}
    (out / "audit.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
