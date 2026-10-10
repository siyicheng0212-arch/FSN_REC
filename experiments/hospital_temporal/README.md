# Prioritize temporal experiments without reusing contaminated visual evidence

The current protocol is train7307 / Menzhen-development888. Menzhen is used
for checkpoint/model selection, not independent testing. R+D is disabled.
The former train7372 contains799 of these888 Menzhen clips. Old FSN-trained
visual weights, features and temporal checkpoints cannot be relabeled as new
hospital-held-out results.

This entry currently implements **CPU-only data/chain auditing**. It neither
exports features nor trains a TCN. The ongoing, fixed VideoMAE checkout and
output are not modified. Complete the new A, strictly replay its selected
checkpoint, then export all7307+888 frozen features and logits under that new
seal before any formal temporal experiment. A is frozen for every T comparison.

## Run now, without occupying the GPU

```sh
PYTHONDONTWRITEBYTECODE=1 "$FSN_PYTHON" -m pytest -q \
  tests/test_hospital_temporal.py -p no:cacheprovider

# Default read-only plan; no GPU probe, output creation or training.
PYTHONDONTWRITEBYTECODE=1 "$FSN_PYTHON" \
  -m experiments.hospital_temporal.prepare \
  --protocol-dir "$FSN_NEW_HOSPITAL_PROTOCOL" \
  --historical-train-manifest "$FSN_OLD_TRAIN_MANIFEST"

# To preserve an audit, append these options after verifying the plan:
# --output "$FSN_NEW_PRIVATE_CHAIN_AUDIT" --write
```

All variables mean real absolute paths. Private chains, IDs and source paths
stay outside Git. Only `public_safe_summary.json` contains identity-free
aggregate counts. An existing output is rejected. The script opens only the
two sealed new manifests and the optional explicit old train manifest; no
additional test file, cache or model weights are opened.

Candidate rules preserve the prior structural masks: same split, recording,
group, source video and annotation time base; unique start; no overlap or nested
interval; gap at most five seconds. Every excluded adjacency cuts a segment;
no clip is removed or used to bridge a cut. Clinical and teaching candidates
are both allowed. Neither source names nor action labels enable a connection.
Source names are only aggregate reporting fields. Candidate adjacency does not
establish a real editing boundary or clinical continuity.

## First temporal comparisons after new frozen A is available

1. A-only (same sealed logits; no extra training).
2. Matched clip-only correction head (no neighbors): test extra head capacity.
3. The actual existing ordinary residual TCN: mean/global and max/local
   VideoMAE tubelet features plus seven raw A logits; Linear/LayerNorm/GELU,
   three width-three dilated residual blocks1/2/4, correction added to A.
4. The same TCN checkpoint with isolated clips explicitly retaining A: this
   is an inference protection control, not an independently trained model.

The historical TCN does not already protect isolated clips. Never silently
change that behavior and call it an exact replay. Symmetric convolution is
offline bidirectional context, not online inference. Fifteen receptive-field
positions mean candidate clips, not video frames or fixed seconds.

These training/export/evaluation stages still require implementation for the
new seal; this audit entry does not claim that a queued TCN run exists.
First diagnose changed/improved/worsened/wrong-to-wrong counts, both
sweep/reperfusion directions, each class, isolated/neighbor clips, sources and
durations. Evaluate original/no/shuffled/cross-record neighbors under one
checkpoint and fixed random state. Synthetic perturbation tests are not real
editing-boundary evaluation. Only then define a minimal T′ modification with
matched capacity/budget and pre-register it before development scoring; do not
blindly add another gate. No new seeds or old resumed experiments are launched
by this preparation command.
