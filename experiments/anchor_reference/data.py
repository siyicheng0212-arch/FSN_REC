"""Fixed paired views; donors never provide an anchor's supervision."""
import hashlib
import json
import numpy as np
from experiments.relation.tcn_baseline import Segment, collate_segments


def train_replacements(segments, metadata, seed=42):
    by_id = {r["clip_id"]: r for r in metadata}
    pool = [(si,j,c) for si,s in enumerate(segments) for j,c in enumerate(s.clip_ids)
            if by_id[c]["source_kind"] == "clinical"]
    if any(by_id[c]["split"] != "train" for s in segments for c in s.clip_ids):
        raise ValueError("training views must contain train clips only")
    rng = np.random.default_rng(seed)
    rows, excluded = [], {}
    for si,s in enumerate(segments):
        for j,c in enumerate(s.clip_ids):
            anchor = by_id[c]
            if anchor["source_kind"] != "clinical" or len(s.clip_ids) < 2:
                excluded["nonclinical_or_singleton"] = excluded.get("nonclinical_or_singleton",0)+1
                continue
            choices = [r for r in pool if by_id[r[2]]["recording_key"] != anchor["recording_key"]]
            preferred = [r for r in choices if by_id[r[2]]["source_collection"] == anchor["source_collection"]]
            choices = preferred or choices
            if not choices:
                excluded["no_other_train_record"] = excluded.get("no_other_train_record",0)+1
                continue
            swaps = []
            for pos in range(len(s.clip_ids)):
                if pos != j:
                    ds,dj,_ = choices[int(rng.integers(len(choices)))]
                    swaps.append([pos,ds,dj])
            rows.append({"segment":si,"anchor":j,"swaps":swaps,"same_collection":bool(preferred)})
    if not rows:
        raise ValueError("no eligible train-only neighbor replacement anchors")
    return {"split":"train","seed":seed,"rows":rows,"excluded":excluded}


def anchor_view(segments, row, augmented):
    s = segments[row["segment"]]; j = row["anchor"]
    if not 0 <= j < len(s.clip_ids):
        raise ValueError("invalid anchor")
    features, scores = s.features.copy(), s.a_logits.copy()
    if augmented:
        for pos, ds, dj in row["swaps"]:
            if pos == j or not 0 <= pos < len(s.clip_ids):
                raise ValueError("replacement cannot touch anchor or padding")
            features[pos] = segments[ds].features[dj]
            scores[pos] = segments[ds].a_logits[dj]
    labels = np.full_like(s.labels, -100); labels[j] = s.labels[j]
    if not np.array_equal(features[j],s.features[j]) or not np.array_equal(scores[j],s.a_logits[j]):
        raise AssertionError("anchor evidence changed")
    if not np.array_equal(features[:,-7:], scores):
        raise ValueError("donor visual features and A scores were not paired")
    return Segment(s.clip_ids.copy(),features,scores,labels,s.source_kinds.copy())


def anchor_batch(segments, plan, indices, augmented):
    rows = [plan["rows"][int(i)] for i in indices]
    return collate_segments([anchor_view(segments,r,augmented) for r in rows]), [r["anchor"] for r in rows]


def sampling_plan(segment_count, anchor_count, epochs=60, seed=42):
    rng = np.random.default_rng(seed)
    return {"normal_orders":[rng.permutation(segment_count).tolist() for _ in range(epochs)],
            "anchor_orders":[rng.permutation(anchor_count).tolist() for _ in range(epochs)],
            "seed":seed,"epochs":epochs}


def plan_hash(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
