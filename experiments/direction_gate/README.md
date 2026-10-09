# Prediction-direction / confidence residual acceptance

This adds gates after the **completed VideoMAE + ordinary historical-class TCN**.
A and the TCN are strictly frozen; this is not a new visual backbone or a claim
of true edit-boundary detection. Reuse train7372/val823, existing numeric features
and sealed legal chain layout. Do not read test, re-export A, resume an old run,
change a running checkout or overwrite any output.

`z = A + sigmoid(s) * (TCN - A)`. The prediction direction refers to the A and
TCN predictions of **the same clip**, not neighboring ground-truth actions.

Four fresh gates start with the same alpha=0.95 and new optimizer states:

| Gate | Definition | Trained parameters |
| --- | --- | --- |
| scalar | one bias | 1 |
| class_conditioned | seven biases indexed by predicted A class | 7 |
| direction | class bias + train-support-shrunk 7x7 prediction-pair offset | 56 |
| direction_confidence | direction + six linear confidence terms | 62 |

An offset is multiplied by `train_pair_count/(train_pair_count+20)`. An unseen
pair has zero offset and falls back to its A-class bias. Six q inputs are A/TCN
softmax top-two margins, normalized entropies, probability total variation and
TCN-target probability change. q is standardized using **train only**, with
scale floor .05. No source/name/path/recording/truth enters the gate. Evidence
and normalizers are detached. Forced alpha=1 / alpha=0 return the exact TCN / A.
Ordinary singleton forward is preserved, not silently changed to A fallback.

## Fixed first-round protocol

Seed42, AdamW .001 / weight_decay .01, batch256 clips, gradclip5, unweighted
seven-class CE. **30 actual epochs for every gate, no early stop**: four gates
share private precomputed sampling orders and identical optimizer step counts.
Select earliest best val full-seven-class F1; Accuracy is from that checkpoint.
This is a fixed gate-training budget, not equal cost to visual/TCN training.
No neighbor augmentation or extra loss is used in this first structure-only
comparison. No MLP, manual direction override or val-tuned shrinkage is added.

Before training, replay all 823 historical ordinary-TCN predictions and compare
clip-by-clip labels, A predictions and TCN predictions. Strictly load the original
model state. Record the train A/TCN correct/fix/harm counts and prediction-pair
supports; very easy in-sample evidence is an explicit limitation, not hidden.

A fixed synthetic challenge replaces **one neighbor per clinical anchor**,
preferring the same source collection but requiring a different recording and
the same split. Donor features and A logits are replaced together. Each forward
only assesses the unchanged anchor, never donor positions with old labels. All
native clips remain. Train and val challenge donors are separate; native gate
training uses no challenge and no val donor. Sampling does not use action labels.
These views intentionally violate continuity and only test this synthetic case.
Source collection is used by the private challenge sampler, not by the gate.

## Safety / execution

Default plan validates evidence / complete base-run / SHA and clean full commit.
It probes no GPU and creates no output. `--execute` uses one selected physical
RTX3090 UUID mapped to child cuda:0, never DDP. Serial stages:
CPU tests -> frozen-base diagnose/cache -> real CUDA disposable three updates
per gate -> scalar -> class -> direction -> direction_confidence -> evaluation.
Any failure stops subsequent stages; do not automatically retry or modify the
fixed checkout. Busy GPU compute PIDs block launch; driver bookkeeping <=64MiB
and strictly zero utilization use the already-tested ordinary-TCN release guard.
Private output reservation is atomic and every stage records PID/log/exit code.

```bash
env -u CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1 \
  "$FSN_GATE_PYTHON" -m experiments.direction_gate.run \
  --index "$FSN_GATE_INDEX" --source-protocol-dir "$FSN_GATE_SOURCE" \
  --base-run "$FSN_GATE_BASE_RUN" --output "$FSN_GATE_OUTPUT" \
  --expected-commit "$FSN_GATE_COMMIT" --gpu 0 \
  --gpu-uuid "$FSN_GATE_GPU_UUID" --python "$FSN_GATE_PYTHON"
# Add --execute only after verified plan + human execution authorization.
```

nohup the same command into a fresh outer log with shell noclobber. Fix all
variables to actual absolute paths. Do not put private paths in Git documents.

## Outputs and limitations

Private: feature-derived cache, sampling orders, replacement plans, input seals,
weights, histories, PID/logs and per-clip predictions. Only reviewed aggregate
`evaluation/public_safe_summary.json` is suitable for export. The summary has
seven-class P/R/F1/support, confusion, paired changes against A/TCN, preserved
TCN fixes / prevented TCN harm, both sweep/reperfusion directions, singleton,
source/duration, identity-free recording aggregates, gate distributions and
actual resource/time/parameter counts. Slice key `VideoMAE_TCN` in reused metrics
means the **gated output** inside each gate section, not the unchanged TCN.
Record aggregates are not patient independence. Class6 has only six val clips.
val823 is repeatedly used development validation; no significance or paper
novelty claim follows from one seed or a renamed gate. Sparse prediction pairs,
in-sample train evidence and synthetic challenges must be reported. A successful
smoke or launch is not completed training. No auto-tuning or additional seeds.

CPU tests:
`python -m unittest tests.test_relation_tcn_baseline tests.test_videomae_tcn tests.test_direction_gate -v`
