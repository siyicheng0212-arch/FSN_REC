# Frozen VideoMAE: mature temporal-head controls

This is a bounded, seed42 follow-up comparison, **not** continuation or
repetition of a completed gate/context-evidence run. The existing visual A,
full train7372/val823 split and legal candidate chains remain fixed. No video,
frame cache, test set, visual backbone training or new A export is needed.
val823 is repeatedly used development validation, not an independent test.

## Three serial tasks

| Task | Initialization and output | Structure | Loss |
| --- | --- | --- | --- |
| `plain_tcn_fresh` | Fresh ordinary TCN; A logits plus zero-initialized correction | width64, three kernel3 residual blocks, dilation1/2/4, dropout.1 | valid-clip CE |
| `mstcnpp` | Fresh native classifier; final refinement logits | width64, 11 dual-dilation prediction-generation layers, three 10-layer refinements, dropout.5 | sum of four stage CE + .15 truncated log-probability smoothing |
| `asformer` | Fresh native classifier; final decoder logits | width64, 10 hierarchical convolution/attention layers, three decoders, reductions2/2, channel mask.3, dropout.5 | same four-stage loss |

All heads receive the exact existing1543-dimensional vector: temporal mean of
eight spatial-mean ViT tokens (768), temporal mean of eight spatial-max ViT
tokens (768), then seven raw A logits. These are two poolings of the **same**
VideoMAE backbone, not two independently trained global/local encoders. No new
normalization, crop, label definition, identity or source-name input is added.

Each candidate segment is a standalone sequence. Variable-length batches are
grouped by exact length before upstream inference, so padding cannot become a
neighbor, change InstanceNorm statistics or leak a different recording into
attention. Every singleton is retained. Native MS-TCN++/ASFormer classify it;
they are not silently forced to fall back to A. This differs from the prior
center-preserving residual method's exact singleton A behavior.

All three use seed42, 30 actual epochs (no early stopping), segment batch16,
AdamW lr.001/wd.01, clip-grad5, identical pre-generated segment orders, and
earliest-tie full-seven-class validation Macro-F1 checkpoint selection. Every
epoch visits all7372 train clips once; there are no anchor replacement views
in formal training. Actual optimizer steps, gradients, elapsed time, parameter
counts and CUDA allocation are reported. Training two models with the same
epochs/optimizer is **not** equal compute, capacity or an architecture-only
causal experiment. In particular native multi-stage classifiers differ in
initialization, residual output and loss from ordinary A+TCN.

## Pinned upstream code and adaptations

- [MS-TCN++ paper](https://arxiv.org/abs/2006.09220),
  [author code at f423a9e](https://github.com/sj-li/MS-TCN2/tree/f423a9e65f4ccb1cd7322eb9f94946a19e787993).
- [ASFormer paper](https://arxiv.org/abs/2110.08568),
  [author code at e1bbe4f](https://github.com/ChinaYi/ASFormer/tree/e1bbe4f3ed083748f91467c51a63ac2a8b9277ad).

Pristine upstream model text and both MIT licenses are in `vendor/`. Adapted
runtime files omit original trainers/visualization imports. Exact changes:

1. Fix the public MS-TCN++ `MS_TCB` syntax typo at its constructor; retain its
   native dual-dilation and refinement computations.
2. Apply ASFormer's author-documented correction from
   `window_mask[:, :, i:i+bl]` to `window_mask[:, i, i:i+bl]`.
3. Register ASFormer window masks as device-movable buffers; allocate padding
   on the input device; fix batch/window stacking order to support batch>1.
4. For singleton non-affine InstanceNorm, use its mathematical zero-variance
   result (zeros). Keep convolution, attention, residual and classifier paths;
   never invent a singleton A fallback.
5. For ASFormer windows containing all real keys, avoid constructing large
   virtual matrices. Preserve the author's 1e-6 padded-key softmax mass
   analytically. CPU tests compare this fast path to explicit native padding.
6. Smoothing only uses actual valid within-segment adjacent pairs and divides
   by their count times seven. Upstream code divides a padded tensor mean;
   that convention would give different regularization merely because of a
   batch companion's length. Singleton smoothing is zero, never NaN.

The source task is dense-frame action segmentation; **here one position is a
clip with a seven-class label**. A dilated span measured in clips is not a
physical-time interval. Short chains (median two clips) and nonuniform clip
duration differ materially from original benchmarks. Do not claim published
benchmark reproduction, causal online operation, or that candidate adjacency
has been manually verified as a continuous treatment boundary.

## Plan, test, execute

Use a clean independent worktree at a full fixed runtime commit and a fresh
private output outside all code/input/old-output trees. Define task-specific
variables on the server using actual verified paths; never paste credentials.

```sh
# CPU models, upstream parity, padding/singleton, gradients and launcher tests
"$FSN_HEAD_PYTHON" -m unittest tests.test_relation_tcn_baseline \
  tests.test_videomae_tcn tests.test_direction_gate tests.test_temporal_heads -v

# Default is read-only: validates all feature/manifest/sealed-baseline SHA,
# records the queue, does not probe GPU and does not create output.
env -u CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1 \
  "$FSN_HEAD_PYTHON" -m experiments.temporal_heads.run \
  --index "$FSN_HEAD_INDEX" --source-protocol-dir "$FSN_HEAD_SOURCE" \
  --base-run "$FSN_HEAD_ORDINARY_TCN" \
  --prior-context-run "$FSN_HEAD_COMPLETED_CONTEXT" \
  --output "$FSN_HEAD_NEW_OUTPUT" --expected-commit "$FSN_HEAD_COMMIT" \
  --gpu 0 --gpu-uuid "$FSN_HEAD_GPU_UUID" --python "$FSN_HEAD_PYTHON"

# Same command + --execute under nohup. No other flag may change.
# Physical GPU0 is mapped by UUID to each independent child's cuda:0, not DDP.
```

The launcher checks actual GPU name/UUID, compute processes and release before
each stage; idle utilization alone is not success. Its queue is unit tests →
old823 per-clip replay audit → three real-frozen-feature CUDA updates per head →
three independent training tasks → strict best-checkpoint evaluation.
The completed prior six-model context suite is read only, checked against the
same A and baseline seals, and included as historical comparisons without
pretending it had the same initialization or normal/anchor-view budget.

`.launch_once`, fixed source commit/SHA and input protocol, stage PID/exit-code
TSVs, per-epoch history, best/result and peak allocation are recorded. Source
and input seals are rechecked between stages. A failed stage stops all later
tasks and preserves the scene. There is no resume, automatic retry, overwrite,
OOM batch change, extra seed or hyperparameter search.

## Evaluation and evidence limits

Only after all stage exit codes0 and complete results: replay each selected
best checkpoint, report complete seven-class P/R/F1/support/confusion, accuracy,
paired fix/harm/wrong-to-another-wrong versus A, historical ordinary TCN and
same-budget fresh ordinary TCN, retained historical corrections, both
sweep↔reperfusion directions, source/duration/recording aggregates, singleton
results, parameters/time/memory. A fixed clinical cross-record neighbor-swap
diagnostic uses the same prior seed42 construction (val-only donors within val,
never entering training); report actual logit/prediction changes. It is a
synthetic challenge, not validated natural video-edit detection.

Previous native train A/TCN accuracy was100%; these frozen train features are
not honest out-of-fold predictions. Fresh temporal heads do not erase that
limitation. Do not train on val errors, invent independent test evidence, call
source-rule labels real boundaries, or make a paper conclusion from one seed.
The six-sample fixed-needle class is explicitly reported rather than hidden.

Only code/config/tests/attributed documentation and de-identified aggregate
reports may go to a separate non-main GitHub branch. Numeric features, model
weights, source records, IDs, plans, raw logs and per-clip predictions remain
private. No merge to main and no later experiment automatically starts.
