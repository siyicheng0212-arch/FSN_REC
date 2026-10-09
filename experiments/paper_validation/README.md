# Fixed development-validation controls

All data remain the sealed **train7372 / val823**. `val823` is repeatedly used
for checkpoint selection: **not an independent test**. No test is loaded and
no new visual encoder is trained or exported. The protocol is exploratory
after earlier development results, not an untouched preregistered benchmark.

## Reuse, do not retrain completed jobs

The audit replays six completed temporal checkpoints at seeds42/10/1217,
plus two already completed mature heads. It verifies SHA, strict state load,
best-F1 selection, history, actual 30 epochs / 5910 steps and historical
per-clip predictions for the six paired temporal runs.

For each ordinary temporal checkpoint it also runs **the same checkpoint
with exact A logits for native singleton chains**. This is an inexpensive
control, not a newly trained network. Context increment must be compared
against this protection, not only against an unprotected TCN.

The same evaluation also reports all seven precision/recall/F1/support and
confusion, both sweep/reperfusion directions, improved/worsened/wrong-to-other-
wrong, clinical/network, duration, singleton/neighbor and identity-free record
aggregates. The six-support class remains in the seven-class score; its F1 is
particularly unstable. Record isolation is not patient isolation.

Fixed interventions (seed42, never use labels to select donors):

1. Native candidate chains.
2. No neighbors: same checkpoint applied separately to each target.
3. Shuffle all non-anchor slots within that original chain with a nonzero
   circular shift. Fewer than two non-anchor slots are excluded, not counted
   as a successful permutation.
4. Replace every non-anchor slot with a clinical clip from a different
   validation recording, preferring the same collection.

The current clip, its A logits and label remain fixed; donor features and
donor A logits always travel together. Only the unchanged anchor is scored.
Plans and logits are private. Report eligible-only and overall results and
actual changed predictions/logit differences. These conditions test synthetic
robustness, not detection of naturally edited boundaries. Source is used for
sampling/stratification, never passed to the model or supplied as an oracle.

## Historical AdaFocus transfer

The adapter requires the actual historical source SHA, frozen index SHA,
A checkpoint SHA and TCN checkpoint SHA. It verifies original input semantics
(mean global1280, mean local2048, seven **raw logits**), original width64 /
three residual blocks / kernel3 / dilation1,2,4 / original singleton behavior,
and replays historical 823 predictions exactly before training.

Two paired seed42 continuations initialize from the same historical TCN:
`legacy_continue` and `context_delta_continue`. Same 30 epochs, AdamW .001 /
weight decay .01, batch16 chains, clip5, normal/unchanged-anchor CE at .5/.5,
same orders and dropout seeds, no donor augmentation and no gate. Only the
TCN is trained; A is frozen. A matched comparison on this second A checks
transfer, not three-seed stability of the whole visual system.

Context increment is `A + T(full) - T(center-only)` with shared weights and
dropout. Singleton increment is exactly zero. This is neither causal nor
strict boundary isolation and contains no learned connection classifier.

## Commands

Use a clean independent worktree at the recorded full commit. All variables
below must be real absolute private paths and verified UUIDs. Default commands
are read-only plans: no GPU probe or output creation. Add `--execute` only
after inspecting their seals. Unset outer `CUDA_VISIBLE_DEVICES`; each worker
gets one GPU UUID and uses `cuda:0`, never DDP. New outputs must not exist.

```sh
env -u CUDA_VISIBLE_DEVICES "$PY" -m experiments.paper_validation.run \
  --mode audit --index "$VIDEO_INDEX" --source-protocol-dir "$SOURCE" \
  --base-run "$VIDEO_BASE" --multiseed-root "$PAIRED_COMPLETED" \
  --mature-run "$PARTIALLY_COMPLETED_HEADS" --output "$NEW_AUDIT" \
  --expected-commit "$COMMIT" --gpu 1 --gpu-uuid "$GPU1_UUID"

env -u CUDA_VISIBLE_DEVICES "$PY" -m experiments.paper_validation.run \
  --mode legacy --index "$ADAF_INDEX" --source-protocol-dir "$SOURCE" \
  --base-run "$VERIFIED_OLD_BASE" --variant legacy_continue \
  --output "$NEW_OLD_CONTINUE" --expected-commit "$COMMIT" \
  --gpu 2 --gpu-uuid "$GPU2_UUID"
# Separate paired job: variant context_delta_continue, fresh output and GPU3.

# ASFormer is from scratch, not resuming its interrupted checkpoint:
env -u CUDA_VISIBLE_DEVICES "$PY" -m experiments.temporal_heads.run \
  --index "$VIDEO_INDEX" --source-protocol-dir "$SOURCE" \
  --base-run "$VIDEO_BASE" --prior-context-run "$COMPLETED_CONTEXT42" \
  --variants asformer --output "$NEW_ASFORMER" --expected-commit "$AS_COMMIT" \
  --gpu 0 --gpu-uuid "$GPU0_UUID"
```

Every job writes atomic once-only reservation, process mapping, stage logs,
exit codes and completion state. Failure is preserved, no automatic retry.
Completed plain/MS-TCN++ checkpoints can be replayed even if the old suite's
ASFormer was interrupted; this does not call that old whole suite completed.
Mature heads use their native classifier/multistage loss and normal chains;
context continuation uses a warm checkpoint and two views. Epochs/steps are
documented, **not** equal-compute or architecture-only causal comparisons.

CPU prechecks:
```sh
"$PY" -m unittest tests.test_paper_validation tests.test_context_evidence \
  tests.test_context_multiseed tests.test_temporal_heads -v
```

Only code, tests, configuration and identity-free aggregate reports may be
published. Never publish manifests, IDs, private plans/audits, predictions,
features, weights, raw logs or private paths. No main merge. No automatic
additional seeds, tuning or new architecture after these registered jobs.
