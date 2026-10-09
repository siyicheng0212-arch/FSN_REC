"""Label-free, immutable-anchor neighbor interventions; private plans stay local."""
from collections import Counter
import numpy as np
import torch
from experiments.relation.tcn_baseline import Segment, collate_segments


def singleton_fallback(logits, a_logits, segments):
    result = logits.clone()
    offset = 0
    for segment in segments:
        if len(segment.clip_ids) == 1:
            result[offset] = a_logits[offset]
        offset += len(segment.clip_ids)
    if offset != len(logits):
        raise ValueError("fallback coverage mismatch")
    return result


def plans(segments, metadata, seed=42):
    by_id = {r["clip_id"]: r for r in metadata}
    pool = [(si, j, c) for si, s in enumerate(segments) for j, c in enumerate(s.clip_ids)
            if by_id[c]["source_kind"] == "clinical"]
    rng = np.random.default_rng(seed)
    result = {"shuffle": [], "cross_clinical": [], "shuffle_excluded": Counter(),
              "cross_excluded": Counter(), "seed": seed, "split": "val"}
    for si, s in enumerate(segments):
        for j, c in enumerate(s.clip_ids):
            anchor = by_id[c]
            if anchor["split"] != "val":
                raise ValueError("development challenge must use val only")
            neighbors = [k for k in range(len(s.clip_ids)) if k != j]
            if len(neighbors) >= 2:
                # Nonzero circular shift is a derangement of all non-anchor slots.
                shift = int(rng.integers(1, len(neighbors)))
                donor = np.roll(neighbors, shift).tolist()
                result["shuffle"].append({"segment": si, "anchor": j,
                    "swaps": [[pos, si, dp] for pos, dp in zip(neighbors, donor)]})
            else:
                result["shuffle_excluded"]["fewer_than_two_nonanchor_slots"] += 1
            if anchor["source_kind"] != "clinical" or not neighbors:
                result["cross_excluded"]["nonclinical_or_singleton"] += 1
                continue
            choices = [x for x in pool if by_id[x[2]]["recording_key"] != anchor["recording_key"]]
            preferred = [x for x in choices if by_id[x[2]]["source_collection"] == anchor["source_collection"]]
            choices = preferred or choices
            if not choices:
                result["cross_excluded"]["no_other_record"] += 1
                continue
            swaps = []
            for pos in neighbors:
                ds, dj, dc = choices[int(rng.integers(len(choices)))]
                if by_id[dc]["split"] != "val":
                    raise ValueError("cross split donor forbidden")
                swaps.append([pos, ds, dj])
            result["cross_clinical"].append({"segment": si, "anchor": j,
                "swaps": swaps, "same_collection": bool(preferred)})
    return result


def view(segments, row):
    s = segments[row["segment"]]
    f, a = s.features.copy(), s.a_logits.copy()
    for pos, ds, dj in row["swaps"]:
        if pos == row["anchor"]:
            raise ValueError("anchor cannot be replaced")
        f[pos] = segments[ds].features[dj]
        a[pos] = segments[ds].a_logits[dj]
    j = row["anchor"]
    if not np.array_equal(f[j], s.features[j]) or not np.array_equal(a[j], s.a_logits[j]):
        raise ValueError("anchor evidence changed")
    # All outputs except the preserved anchor are discarded; no donor label is evaluated.
    return Segment(s.clip_ids.copy(), f, a, s.labels.copy(), s.source_kinds.copy())


@torch.no_grad()
def infer(model, segments, device):
    model.eval(); rows = []
    for start in range(0, len(segments), 16):
        batch = collate_segments(segments[start:start+16])
        output = model(batch[0].to(device), batch[1].to(device), batch[3].to(device)).cpu()
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("nonfinite replay")
        rows.extend(output[i, :len(s.clip_ids)] for i, s in enumerate(batch[4]))
    return torch.cat(rows)


@torch.no_grad()
def intervene(model, segments, native, plan_rows, device):
    offsets = np.cumsum([0] + [len(s.clip_ids) for s in segments]).tolist()
    result = native.clone(); eligible = []
    model.eval()
    for start in range(0, len(plan_rows), 16):
        rows = plan_rows[start:start+16]
        batch = collate_segments([view(segments, r) for r in rows])
        output = model(batch[0].to(device), batch[1].to(device), batch[3].to(device)).cpu()
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("nonfinite synthetic intervention")
        for b, row in enumerate(rows):
            flat = offsets[row["segment"]] + row["anchor"]
            result[flat] = output[b, row["anchor"]]
            eligible.append(flat)
    return result, eligible
