# Anchor-reference TCN: a preregistered exploratory follow-up

This is **code preparation**, not a completed training experiment. There are
no new server/CUDA results bundled here. The default launcher prints a private,
read-only plan. Training requires an explicit `--execute`.

## Motivation and scope

The earlier residual TCN corrected some A errors but also spoiled correct A
predictions. Removing its center-only correction lost useful corrections on
the weaker AdaFocus backbone. On the stronger VideoMAE backbone, the
context-increment method did not consistently beat plain singleton protection.
Neither observation proves that a real editing boundary caused an error.

This follow-up keeps self-classification and changes **actual noncenter
convolution contributions**, instead of multiplying the final prediction or
subtracting away self-classification. The gate reads the frozen target and
neighbor visual vectors, their signed/absolute differences, cosine similarity,
and signed convolution offset in **clip units**, not seconds or verified PTS.
Its projection is shared across the three layers. It is directed: generally
`g(t <- j) != g(j <- t)`. No source string, identity, path, label or ground-truth
edge enters the forward pass.

For each width-three convolution with its actual dilation:

```
u[t] = W_center h[t] + sum(g(t <- j) W_offset h[j]) + bias
g = 2 sigmoid(MLP(project(f[t]), project(f[j]), differences, cosine, offset))
```

The center tap and bias are unchanged. Invalid/padded neighbors contribute zero.
GELU, pointwise mix, residual and dropout follow the verified three-block TCN.
The last gate layer is initialized to zero, so `g=1`; it is still computed and
receives gradients. All groups have identical shared base initialization and
an independent module-initialization RNG. Forced `g=1` reproduces the native
plain model with **the same weights**, not its separately trained checkpoint.
Forced `g=0` at every layer equals the same native classifier on single clips;
it does **not** force the output to A.

## Important recipe distinction

All new groups predict seven logits with a **native classifier**, without adding
A's logits directly to the output. A's seven raw logits remain in the original
input alongside mean-global and mean-local features. A is frozen and never
loaded for training, re-exported or fine-tuned.

This removes the exactly-A initial residual output and allows a nonzero
classification learning problem at initialization. It does **not** solve the
in-sample/overconfident A problem: the frozen A was trained on all train7372 and
its training predictions are not out-of-fold. Initial and epoch1/10/30/60 train
errors/disagreements are recorded. If the head again becomes trivial, report
that fact; do not silently introduce noise, label corruption or a new protocol.

The historical residual A+TCN is strictly reloaded and its 823 individual
predictions replayed before running this experiment, but its weights are
**not** used to initialize the new native classifier. Historical results are
references, not matched-budget evidence. This is not called a reproduction of
the historical residual baseline. Any benefit must first beat the new matched
plain/augmented/scalar controls.

## Registered variants and common budget

| Variant | Neighbor reference | Preserved-anchor view |
|---|---|---|
| plain | fixed 1 | original neighbors |
| plain_aug | fixed 1 | replaced neighbors |
| reference | target-dependent pair MLP | original neighbors |
| reference_aug | target-dependent pair MLP | replaced neighbors |
| scalar_aug | one shared trainable scalar | same replaced neighbors |

The first four form a structure-by-augmentation comparison. The fifth checks
whether globally weakening/amplifying neighbors suffices. The dynamic module
has more parameters; scalar comparison and same-checkpoint audits do not remove
that capacity difference. A capacity-matched control remains necessary before
claiming that parameter count cannot explain a gain.

Fixed seed42, 60 epochs, no early stop; AdamW lr.001/wd.01, cosine epoch schedule
to .00001, gradient clipping5, batch16 segments, unchanged three-block width64
layout/dropout.1. Float32 matches the numerical-feature TCN task; this is not a
raw-video bf16 backbone-training experiment. The normal-chain CE and preserved
anchor CE have equal .5 weights. Every group uses the same number of views,
optimizer steps and private orders, with actual elapsed time/memory reported.
The second view uses the same train clinical anchors even in nonaugmented
groups; no group gets an extra pool or different sampling schedule.

Donors are from **train7372 only**, another clinical recording, same collection
preferred, without using action labels to choose them. All nonanchor positions
are replaced with paired visual vectors/A logits. Only the preserved anchor
has a target (`-100` elsewhere); donor positions cannot accidentally receive
the original labels. Source metadata is used for this sampling and reporting,
not supplied as a model input or inference oracle. Synthetic replacement is
not a naturally observed editing boundary.

## Data and historical assets

Reuse the existing full train7372/val823 source protocol and the matching sealed
8195-entry frozen feature index. Preserve its manifests, candidate chains,
five-second structural eligibility and split. Do not use the old6586/786 split,
test, raw videos, weights of A, or frame caches. No source files are edited.

`--backbone videomae2` validates the sealed export and the completed ordinary
VideoMAE TCN run. `--backbone adafocus` uses the already verified historical
adapter: source/index/checkpoint SHA, genuine configuration and strict state
load. Missing assets or a historical prediction mismatch stop the run. A
template or new reference TCN is not a replacement for the old model.

## CPU checks and read-only plan

In a clean independent checkout at the intended commit:

```sh
"$FSN_PYTHON" -m unittest tests.test_anchor_reference tests.test_relation_tcn_baseline -v
```

Set variables to real absolute server paths. `FSN_OUTPUT` must be a new private
directory outside the repository and inputs, with an existing parent.
`FSN_GPU_UUIDS` must contain one distinct UUID per selected GPU, measured after
the actual server boot, in the same order as `--gpus`; never reuse stale mappings.
Do not put a filled private environment file or printed plan into Git.

```sh
env -u CUDA_VISIBLE_DEVICES "$FSN_PYTHON" -m experiments.anchor_reference.run_suite \
  --backbone videomae2 \
  --index "$FSN_INDEX" --source-protocol-dir "$FSN_SOURCE_PROTOCOL" \
  --historical-run "$FSN_HISTORICAL_RUN" --output "$FSN_OUTPUT" \
  --expected-commit "$FSN_COMMIT" --gpus 0,1,2,3 \
  --gpu-uuids "$FSN_GPU_UUIDS" --python "$FSN_PYTHON"
```

The plan does not query GPUs or create output. It still fully reads/hashes the
numeric frozen evidence and verifies the original protocol. Use
`--backbone adafocus` with its own index/historical run and **different output**
for a second independent backbone suite. GPU and input readiness are checked
again, never inferred from another completed suite.

## Explicit execution, one to four cards

Only after inspecting the plan and checking the actual server/driver state,
use the identical command with `--execute`. To persist it, run that command
under `nohup env -u CUDA_VISIBLE_DEVICES ... --execute`, with a new external
private log and capture `$!`. Do not overwrite an existing log or output.

Four cards run plain/GPU0, plain_aug/GPU1, reference/GPU2,
reference_aug/GPU3 independently. Once **all four exit0 with valid results**,
scalar_aug runs on GPU0. This is five independent tasks, not DDP. With
`--gpus 0,1` and two matching UUIDs the waves are [plain,plain_aug],
[reference,reference_aug], [scalar_aug]. One GPU queues all five. All five
real frozen-batch CUDA three-step checks pass **before any formal training**.

For three cards use `--gpus 0,1,2` and three matching UUIDs. Wave1 is
plain/GPU0, plain_aug/GPU1, reference/GPU2. Only when all three finish normally,
wave2 starts reference_aug/GPU0 and scalar_aug/GPU1; GPU2 stays unused. The
recipe, views and optimization budget are unchanged; only scheduling differs.

The launcher records PID, UUID, physical index, logs, stage exits and status.
Each child sees exactly its UUID as cuda:0. It refuses an inherited outer
CUDA_VISIBLE_DEVICES, a busy card, wrong UUID, unavailable/empty `nvidia-smi`,
changed input/code, nonfinite gradients, missing gate updates or existing
output. It accepts <=64MiB idle driver bookkeeping only with utilization0 and
no compute PID for two observations. It never resets a GPU or kills a peer.
On any failure, already-started peers finish naturally and later waves stop;
there is no retry/resume/automatic parameter change.

## Evaluation and decision criteria

The best-F1 checkpoint is strictly loaded and replayed; Accuracy comes from
that same checkpoint. Full seven-class metrics, confusion matrices, A/old-TCN
and matched-control paired corrections, sweep/reperfusion errors, source,
duration, singleton/context and identity-free recording aggregates are saved.
The six-sample class stays in Macro-F1 and is explicitly flagged as unstable.

Fixed val challenges: no neighbor; true nonanchor derangement within chain;
all nonanchor replacements from another clinical val recording. Eligible
cohorts/support are reported separately. Every method uses the same private
plan; current target evidence and truth stay unchanged. No validation donor
enters training. All are **development/synthetic robustness**, not independent
test or natural-boundary recognition.

Reference models also report coefficient distributions by signed span/layer,
same-checkpoint g=1, g=0 and actual changed predictions/logits. The fixed-seed
permutation groups by signed span and audit source, preserves the distribution,
and records both moved slots and actually changed coefficients. If nothing
changes, the control is explicitly ineffective, not favorable evidence.

Prespecified pilot criteria for reference_aug: beat both plain_aug and
scalar_aug in full-seven-class F1, no more A-correct→wrong cases than plain_aug,
and retain >=90% of its A-error corrections. A zero-correction comparator
makes that retention criterion unidentifiable. These are exploratory checks,
not statistical significance or a publication guarantee. Do not choose a
new architecture/seed/threshold after looking at val823 to make them pass.

Only `evaluation/public_safe_summary.json` may be exported after inspection.
Private plans, predictions, logs, features, labels, IDs and weights stay private.
Publish code/tests/config and safe aggregates on a nonmain branch, never merge
main automatically. The dataset is not independently patient-isolated just
because group/recording identities are disjoint. A clean independent test and
stability checks remain missing; current val823 is repeatedly used development.

## Prior art and novelty limits

Visual plus temporal refinement is established, e.g.
[TeCNO](https://arxiv.org/abs/2003.10751).
[Trans-SVNet](https://arxiv.org/abs/2103.09712) already queries temporal embeddings
with spatial information. This experiment is therefore **not** sold as the
first current-observation-driven context model. Its candidate contribution is
the center-preserving, directed per-kernel neighbor reference and explicit
mis-correction/correction-retention protocol for mixed FSN clips. Whether it
adds value beyond scalar/augmentation controls is an unanswered experiment.

Larger mature heads have different recipes and were designed for dense action
segmentation ([MS-TCN++](https://arxiv.org/abs/2006.09220),
[ASFormer](https://arxiv.org/abs/2110.08568)). Do not equate this clip-level
Macro-F1 with segmental F1@IoU, or claim architecture superiority from a single
fixed-budget adaptation with a different objective.
