# FSN-IEA: opt-in within-clip evidence pilot

Status: **implementation candidate; no performance or clinical-role claim**.

## Motivation and falsifiable question

The first-order v3 decoder improves uncertain clip predictions using neighboring
annotated clips, but that is a dataset-order baseline, not a verified clinical
process model. The equal-budget three-window sampler alone underperformed the
uniform full-clip sampler. Moreover, a two-uniform-model ensemble outperformed
uniform-plus-three-window fusion at the same two-model cost. Those findings do
not justify claiming a dense-motion benefit.

The narrow pilot question is whether information **already present in the same
36 uniformly sampled frames** can improve a single visual model when the
AdaFocus policy selects mostly non-adjacent frames. No neighboring clip,
transition prior, extra decoded frame, role pseudo-label, or clinical order is
used.

## Candidate mechanism

The released Uni-AdaFocus path remains the full-clip appearance backbone. An
opt-in lightweight module downsamples the same 36 frames, encodes signed
adjacent-frame differences, forms nine consecutive four-frame evidence tokens,
and pools them with seven class-specific weights. Its seven-logit residual is
added to the original logits. The residual output layer is initialized to
zero, so the candidate is exactly equal to Original at initialization. A
deterministic quality factor suppresses the residual when frame repetition
means no motion was actually observed. The original five-term AdaFocus loss
is preserved by applying the same residual to its two final-path logits.

The evidence tokens are **not** clinician or patient activity labels. Their
anatomical interpretation must be supported by separate human visibility
annotations; current seven-class targets cannot supply those labels.

## Evaluation gate

1. Freeze manifests, video-level groups, source checkpoint SHA, and code commit.
   Check checkpoint loading, initial logit equivalence, forward/backward,
   repeated-frame behavior, optimizer coverage, single-batch overfit, and a
   short smoke run before any formal run.
2. Develop on a deterministic, group-disjoint inner split drawn from the
   7,372 training clips. Do not use the 823-clip validation set repeatedly to
   choose architecture or hyperparameters. No independent test is available in
   the current train-plus-val protocol.
3. Compare Original and IEA with the same 36 frames, checkpoint, seed,
   training schedule, and candidate count. Add a parameter-matched RGB branch
   and a within-clip frame-order shuffle as mechanism controls. Include the
   already measured two-uniform-model ensemble as a separate higher-cost
   reference, not a single-model baseline.
4. Report overall and per-class precision/recall/F1, both directions of
   sweeping/reperfusion errors, source and duration slices, short-clip
   repetition, throughput, parameters, and GPU memory. Inspect action-related
   subtitles and clinically verified visibility before claiming that a token
   represents needle or patient activity.

Proceed to a three-seed formal comparison only if the inner-split pilot is
stable and improves the clinically important error directions without merely
raising one class at the expense of the other. Otherwise retain this as a
negative result and do not turn it into a paper claim.

The frozen 10%-by-source-group inner split uses source training-manifest
SHA-256 `093d0adf08dc1d382c0d2e1863a2f4724cb24ebd5b3d0af070e0c76ca989f1d3`,
seed `20260929`, and yields 6,586 inner-training clips from 1,279 groups and
786 inner-validation clips from 142 disjoint groups. Every class is represented
in both parts, including 116/10 fixation clips. The inner manifest hashes
are `f0fc0ace561cd59c471c86d6ec674144de0d5ede6acf447a5a021787538ebf0a`
and `d6449196c3798a233a593db19da9045c92e02ba62bb6fa5a55d41bf118ee6580`.
The original `split: train` field is intentionally retained because both
subsets reuse the immutable training-frame cache; the output filenames and
`inner_split_role` field define their new logical roles.
