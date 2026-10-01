"""R-to-D integration for an explicitly ordered list of cached A evidence."""
import numpy as np
import torch

from experiments.relation.decoder import decode
from experiments.relation.evidence import EVIDENCE_KEYS


@torch.no_grad()
def predict_chain(relation, features, transition, eligible=None, **decode_options):
    if not features:
        raise ValueError("a chain must contain at least one clip")
    n = len(features)
    if eligible is None:
        eligible = np.zeros(n - 1, dtype=bool)
    eligible = np.asarray(eligible)
    if n == 1 and eligible.shape == (0,):
        eligible = eligible.astype(bool)
    if eligible.shape != (n - 1,) or eligible.dtype.kind != "b":
        raise ValueError("eligible must be boolean with one entry per adjacent pair")
    logits = np.stack([np.asarray(row["logits"]) for row in features])
    q = np.zeros(n - 1, dtype=np.float64)
    relation.eval()
    device = next(relation.parameters()).device
    for i in np.flatnonzero(eligible):
        sides = [{key: torch.as_tensor(row[key], dtype=torch.float32, device=device).unsqueeze(0)
                  for key in EVIDENCE_KEYS} for row in (features[i], features[i + 1])]
        prediction = relation(*sides)
        if prediction.shape != (1,) or not bool(torch.isfinite(prediction).all()):
            raise RuntimeError("invalid R prediction")
        q[i] = prediction.sigmoid().item()
    result = decode(logits, transition, q, eligible=eligible, **decode_options)
    result["relation_probability"] = q.tolist()
    result["eligible"] = eligible.tolist()
    return result
