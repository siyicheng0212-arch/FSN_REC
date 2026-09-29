# Directional evidence inside one clip: prepared candidate

Status: **code and CPU contracts prepared; no formal server training or performance claim**.

## Rationale and scope

The v3 control audit found that directed relations between annotated clips,
not simply label frequencies or smoothing, explain most of its measured gain.
Those relations may reflect editing/annotation conventions and are unavailable
when recognizing a single web clip. This candidate tests whether directed
*within-clip* evidence has value without using another clip's label or order.
It cannot be assumed to reproduce v3's gain.

The Original Uni-AdaFocus path and released SSv2 checkpoint mapping are
unchanged. The candidate reads the same uniformly sampled 36 RGB frames.
One small encoder processes downsampled RGB and signed adjacent-frame
differences. Two independent learned spatial queries produce two **latent**
evidence streams in nine four-frame windows. Their within-window activity
probabilities can coexist. A summary retains activity extent, co-occurrence,
both directed adjacent-window products, and start-to-end change. A zero-
initialized head returns a signed margin for sweeping class 3 versus
reperfusion class 4; the other five logits are not directly corrected.
The correction is attenuated when frames repeat, an exceptionally large
image jump occurs, or the Original already strongly separates the pair.
The data-quality gate is a heuristic, **not** a learned proof of visibility
or a validated shot-cut detector.

The new module has 15,139 parameters (checked with the formal model's
optimizer); its two convolutions contribute about 0.12 GMAC per clip,
excluding interpolation and attention. It initializes to exactly the
Original logits. Unlike the previous IEA pilot, its full new module joins
the five-epoch head warm-up; it is then jointly fine-tuned. This is a
predeclared training design, not a measured fix for IEA's small residual.

**Neither latent stream may currently be named clinician-hand or patient-body
evidence.** The existing mutually exclusive seven-class clip labels do not
supervise those two activities, and a clip can contain both. A defensible
role/visibility claim requires separate audited annotations and matching
spatial/visibility supervision. Until then this is a directional latent-
evidence hypothesis, not the proposed clinical-role model fully validated.

## Frozen next-run pairing

`configs/directional_evidence_protocol.json` fixes paired seeds **42, 2026,
1217** (not 123), the 7,372/823 manifests and hashes, Original versus the
opt-in `directional_evidence` variant, the same SSv2 initialization and
optimizer protocol, and no independent-test call. The 823 clips are a
development validation set because each epoch uses them to select a best
checkpoint. Do not start this pairing until the currently running Original/
IEA jobs have ended; deploy this branch in a separate server worktree.

The CLI entry point is `python -m experiments.train_adafocus --variant
directional_evidence`. `--evidence-relation-mode unordered` keeps the same
parameter count but erases relation direction for the key negative control.
`--disable-evidence-quality-gate` tests the fallback. The optional
`--disable-evidence-ambiguity-gate` tests whether the visual-uncertainty
gate merely suppresses a useful correction. The optional
`--pairwise-loss-weight` defaults to zero; if enabled in a later experiment,
apply the **same** value to Original and the candidate so loss changes are
not mistaken for architecture gains.

Report paired three-seed overall and per-class metrics, both *normalized*
sweeping/reperfusion error directions, source/duration slices, and the
correction magnitude and changed/corrected/harmed predictions. Analyze
short clips with repeated frames and edited clips separately. A one-direction
gain with the reverse direction harmed does not validate the proposed
mechanism. Any clinician/patient interpretation also requires private
visibility/co-occurrence annotation rather than seven-class pseudo-labels.
