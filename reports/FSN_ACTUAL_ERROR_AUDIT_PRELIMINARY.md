# Preliminary visual review of real FSN classification errors

Status: **qualitative diagnostic, not a representative error-rate estimate**.
No source frames, patient identifiers, media paths, or clip identifiers are
published here.

## Scope

The three saved FSN-v3 prediction files contain 823 common evaluation clips.
Across three visual-model seeds, there are 86 reperfusion-to-sweeping and
114 sweeping-to-reperfusion errors before sequence decoding.  We selected
seven original-video intervals from these actual errors across clinical and
online sources for a first private visual check.  Selection deliberately
included both directions and multiple sources; it was not random and cannot
estimate prevalence.

## What the sampled videos show

- Some reperfusion-labeled intervals contain clearly visible patient limb
  motion; in others, sparse frames show little patient displacement while
  the clinician's hand remains near the needle.  A visible-motion-only
  classifier cannot be assumed to identify every reperfusion label.
- One sweeping-labeled clinical interval shows substantial patient leg
  movement while clinician activity near the needle continues.  This is
  consistent with physically concurrent activities, not proof that the
  annotation is wrong.  The dataset's primary-label rule needs clinical
  review before deriving two auxiliary activity labels from the seven-class
  target.
- Online examples include action-related subtitles, and some camera views
  make the needle or target muscle difficult to see.  Text and framing are
  plausible shortcut or observability variables.
- The interval-level distinction cannot be diagnosed from a few still
  frames alone.  Full-speed review and additional independently annotated
  examples are required before attributing errors to motion frequency,
  action boundaries, or label ambiguity.

An FSN clinical trial describes patient resisted knee movement while needle
sweeping continues [primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC11124628/).
This supports checking concurrent activity in the dataset; it does not
resolve any individual annotation.

## Consequences for the next model

1. Complete the running equal-budget uniform-versus-three-window experiment.
   Compare both sweeping/reperfusion error directions and long/short clips;
   a gain would establish sampling value, not role attribution.
2. Privately audit a source- and duration-stratified set of at least 50-100
   clips per focus class.  Independently record whether clinician needle/hand
   motion and patient target-region activity are visible, simultaneous,
   occluded, outside the frame, or represented only in narration/subtitles.
   Record the seven-class primary-label rule and reviewer agreement.
3. Use matched frame budgets and the same visual backbone to compare RGB,
   dense-window RGB, generic whole-frame motion, and text-masked controls.
   Text masking must be applied consistently during training and evaluation.
4. Only if two role signals are observable and independently labeled should
   a role-separated motion branch be trained.  Its candidate benefit must
   exceed equal-budget whole-frame motion and remain after subtitle masking.
   Report both directional confusion rates and the visibly observable subset.

No new role-specific architecture or superiority claim follows from this
initial review.  Private clips and review notes remain outside GitHub.
