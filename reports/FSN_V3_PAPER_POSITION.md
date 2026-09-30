# FSN-v3 Annotated Segment Order Prior and Paper Position

This note identifies what the current FSN-v3 results support, what a
paper could claim, and which tests are still missing. The main finding
is that annotated segment order helps prediction on the existing
development-validation split; neither clinical workflow understanding
nor performance on an independent test has been established.

## What the current evidence actually establishes

FSN-v3 keeps each clip's Original visual logits fixed, estimates a soft
directed seven-class transition table **only from training labels**, and
uses offline Viterbi over time-sorted *annotated segments from one media
record*. It does not segment an untrimmed raw video, make a causal
single-clip prediction, or verify that the media record is one continuous
treatment. Viterbi can use predictions from later clips in the same
record. A single clip has no transition correction.

On the repeatedly examined 823-clip **development-validation** split,
three-seed mean Macro-F1 is 0.77595 for independent visual logits,
0.80893 for directed transitions only, and 0.80983 for full v3.
Initial-label frequency (0.77981), global persistence (0.77754), and
unordered adjacency (0.78190) explain little of the gain. Shuffling
training order lowers Macro-F1 to 0.75420; shuffling inference order
lowers it to 0.72505. The supported inference is that **the direction
of annotated-segment order in this dataset is predictive**. It is not
evidence that every source follows one clinical procedure.

Across three seeds, v3 changed 137/2,469 predictions, corrected 96
and harmed 24. Sweeping→reperfusion errors fell 114→79, but the
reverse rose 86→93: the ambiguous pair was not solved bidirectionally.
See the original mechanism audit for full controls.

The new safe source-stratified recheck is in
[source_strata_order_audit.json](../public_results/fsn/v3/source_strata_order_audit.json).
On the same frozen manifest, clinical-source clips (n=464, 77 media
records) improved from mean Macro-F1 0.7688→0.8078; online-source clips
(n=359, 88 records) improved 0.7081→0.7287, **although online seed 42
slightly regressed**. All 27 singleton clips remained unchanged.
This contradicts a claim that the gain exists only in clinical sources,
but it does not show online edits preserve clinical order. Source type,
timestamp gaps, and sequence length are *not* ground truth for
continuous filming versus montage editing.

## Paper Claim and Comparators

Provisional title: **“When Does Annotated-Segment Order Help
Fine-Grained Fu's Subcutaneous Needling Recognition? A Multi-Source
Evaluation and Bias Audit.”**

Frame this as a task/evaluation and applicability paper, not as a
new Viterbi algorithm. CNN visual predictions followed by HMM smoothing
are established in surgical workflow recognition
([Cadène et al., 2016](https://arxiv.org/abs/1610.05541)).
Temporal segmentation baselines such as
[MS-TCN](https://arxiv.org/abs/1903.01945)
and causal surgical phase modeling such as
[TeCNO](https://arxiv.org/abs/2003.10751) are relevant comparisons,
although their frame/segment prediction budgets must be matched.

The paper's defensible contributions would be:

1. A precisely defined seven-action, multi-source recognition task,
   label policy, video-group-disjoint protocol, and explicit
   single-clip versus ordered-record settings. Do not claim a public
   benchmark unless the data/annotations can actually be released.
2. An unusually explicit order-bias audit: directionless, start-prior,
   persistence, train-order shuffle, inference-order shuffle, and
   random grouping controls with the visual logits held fixed.
3. Source-, continuity-, class-, and duration-stratified reporting,
   including both sweep↔reperfusion directions, singleton fallback,
   and offline future-context cost.
4. If later validated, a **conditional** decoder that enables order
   information only within independently verified continuous runs;
   edited/unknown/singleton input falls back to visual-only.

The fourth point is a *future hypothesis*, not a current v3 result.
An automatically learned continuity gate would need separate
train-only edit/continuity annotations and must beat an equal-information
“ordinary HMM + simple shot-boundary detector” baseline; otherwise
it is not a strong method contribution.

## Required before a final performance claim

- Lock a **new** patient/video-disjoint test set before fitting thresholds,
  choosing decoder strength, or deciding the final claim. The current
  823 clips have guided development and cannot be relabeled as an
  independent test.
- Have humans annotate whether each media record contains continuous
  treatment, interrupted footage, or an edited montage; report these
  strata separately. Clinical versus online source is not a substitute
  for this audit.
- Compare visual-only, ordinary temporal averaging/persistence,
  directed HMM/Viterbi, directionless transitions, and feasible
  MS-TCN/TeCNO-style methods on the same input, visual backbone,
  grouping and compute/information budget. Report offline and
  causal/online settings separately.
- Use confidence intervals clustered by media record/patient, all
  seven per-class supports, normalized two-way sweep/reperfusion
  errors, and unchanged singleton predictions.

If the directional gain fails on the locked test or on verified
continuous clinical records, the conclusion is a useful **dataset
order-bias finding**, not a clinical-workflow recognition advance.
No wording should imply that v3 observes the needle or patient
movement merely because sequence decoding changed a label.
