# Within-clip FSN motion: feasibility gate before model design

Status: **preliminary visual audit, not an error analysis or model result**.
Private source-video frames and clip-level identifiers are intentionally not
included in this document.

## Why this direction is worth testing

FSN-v3 uses the order of annotated clips in a media record.  User-supplied
provenance clarifies that online videos may be edited compilations, so the
order need not reflect one continuous treatment.  A model based on evidence
inside each annotated clip would not need that cross-clip assumption.

The current seven-class label policy treats sweeping and reperfusion activity
as mutually exclusive.  However, an FSN clinical trial describes patient
resisted knee extension while the clinician continues sweeping
[primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC11124628/).
Therefore, simultaneous activity is plausible and must be checked against
our annotation rules before claiming that a visual model can cleanly separate
the two labels.  RGB video can at best identify visible patient activity; it
cannot measure blood reperfusion itself.

## Local data and visual check

The local pre-materialization snapshot contains 8,215 supervised clips,
including 3,646 labeled sweeping and 1,060 labeled reperfusion.  Original
media paths exist for 4,703 of these 4,706 focus clips.  This snapshot is
not byte-identical to the server's frozen formal manifest and its counts must
not be substituted for the formal experiment.

We decoded and privately inspected twelve *label-selected*, not
model-error-selected, clips across FSN, outpatient, Douyin, Bilibili, and
YouTube sources.  These examples suggest that:

- Patient movement is clearly visible in some reperfusion-labeled clips
  (for example, a leg lift or arm movement), while the clinician's small
  needle/hand movement is visible in some sweeping-labeled clips.
- In other reperfusion-labeled clips, sparse still frames show little or no
  patient displacement.  This may reflect isometric contraction, motion
  outside the crop, an unobserved part of the interval, or label/annotation
  semantics; the examples do not distinguish these explanations.
- Clinician needle/hand activity may remain visible during a
  reperfusion-labeled interval.  This is consistent with possible activity
  overlap, but requires full-video and clinician review before coding a
  concurrent label.
- Online videos sometimes contain subtitles or on-screen action text, a
  potential shortcut for an RGB classifier.  A text-masked control is
  necessary before claiming motion-based discrimination.

These are qualitative feasibility observations only.  The three per-clip
FSN-v3 prediction files are not available locally, so none of these twelve
clips can be presented as a confirmed FSN-v3 error example.

## Temporal sampling question

`experiments.full_data` spreads 36 cached frames uniformly over each clip,
so its nominal sample rate is `36 / duration_seconds`.  In this local
snapshot, 1,003/3,646 sweeping and 429/1,060 reperfusion clips are long
enough that the nominal rate is below 3.3 samples/s.  This does **not** prove
aliasing, but it makes a controlled high-rate, multi-window ablation more
urgent than adding a large network.  Keep the total frame budget equal:
compare existing uniform 36-frame sampling with three local windows of
twelve frames each.  Select window positions from clip time without using
the label or evaluation results.  Preserve the same visual backbone,
pretraining, optimizer, and training budget.

## Private observability audit (go/no-go)

Before training a role-specific branch, draw a deterministic stratified
sample of at least 50-100 clips per focus class across source and duration
bands.  Two reviewers should independently mark:

1. Practitioner hand/needle sweeping **visible / not visible / uncertain**.
2. Patient target-limb or muscle activity **visible / not visible / uncertain**.
3. Both activities simultaneously observable; neither observable.
4. The target region outside the frame, occluded, or visually stationary
   despite the label.
5. Action-explicit subtitles, narration-only cues, and internal video cuts.
6. Whether the existing single primary label is unambiguous under written
   annotation rules.

Report agreement and examples by source, duration, and class without
publishing identifiable frames.  If patient activity is frequently invisible
or overlap cannot be resolved with current labels, first revise the
observability/annotation protocol; a two-branch network cannot recover
information absent from RGB.

## Minimum experiment if the audit passes

1. Original Uni-AdaFocus with its current 36-frame uniform cache.
2. The same model with equal-budget dense local windows, **without** a new
   branch.  This isolates temporal sampling.
3. Equal-sampling whole-frame RGB plus a lightweight motion baseline.
4. A two-role candidate with one practitioner/needle region stream, one
   patient target-region stream, explicit visibility masks, and a small
   residual fusion head initialized to preserve the original predictions.
   Start with manually checked regions as a feasibility upper bound; then
   test automatic regions separately.
5. Region-agnostic two-stream, each-role-only, time-shuffled-role,
   background-only, and text-masked controls.  These distinguish actual
   role evidence from more parameters, scene background, or subtitles.

Every method must use the same split, visual pretraining, frame budget where
applicable, and three seeds.  Primary endpoints are seven-class macro-F1,
both directional sweeping/reperfusion error rates and class-specific
precision/recall/F1; report visible and invisible subsets separately.
An improvement only on subtitle-bearing or visibly easy clips would not
support the proposed mechanism.

## Contribution claim, only after evidence

Hand/pose fusion itself is not new in surgical video
[prior work](https://arxiv.org/abs/2211.07021).  A defensible FSN method
claim would require showing that **role-separated, within-clip temporal
evidence** handles edited media and overlapping practitioner/patient
activities more reliably than equal-budget RGB, generic motion fusion,
and the static sequence prior.  Until those experiments pass, this is a
testable hypothesis, not an established innovation.
