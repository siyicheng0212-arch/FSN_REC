# Paper experiments must follow the claim, not the length of the model list

Working FSN question: **Can a temporal model retain useful cross-clip correction
while reducing harmful correction when neighbors are unhelpful?** Continuous
clinical recordings and edited teaching clips motivate the question; they do
not prove that every clinical edge is reliable or every teaching edge harmful.
Do not claim the current negative gate results already solve this problem.

## Priority 1: the mechanism's incremental value

The proposed method must be compared with the **same frozen A, input, native
TCN, chains, budget and supervision**:

| Control/evidence | What it rules out | Existing code/status |
|---|---|---|
| A-only | gains require cross-clip correction at all | sealed A scores; no retraining |
| ordinary matched TCN | just attaching TCN is sufficient | `logits` / `visual`, separately by input |
| ordinary TCN plus singleton-A protection | benefit is only an isolated-clip rule | same-checkpoint reported diagnostic; not a new primary classifier |
| ordinary TCN plus identical neighbor augmentation | benefit is merely data augmentation | `logits_aug` / `visual_aug` |
| shared neighbor scalar | weakening all neighbors suffices | existing anchor-reference `scalar_aug`; retain completed negative/positive results |
| proposed target-driven neighbor reference | actual proposed mechanism | existing `reference` / `reference_aug`; not trained again here |
| g=1/g=0 on that same learned checkpoint | coefficient use vs trained weight changes | existing anchor-reference audits; not independent trained baselines |
| grouped coefficient permutation with changed positions >0 | selected locations matter, not coefficient distribution alone | existing audit; zero changed positions is an invalid contrast |

Do not compare a proposed raw-score model with a masked-score ordinary model
and attribute the difference only to the gate. Current native benchmark cells
are **comparators/diagnostics**, not a replacement innovation. Existing
anchor-reference raw-score experiments can be compared only after their full
input, budget, sampling and selected-checkpoint seals match. A prior historical
residual TCN has a different output/training recipe and remains a reference.

For a target-driven module, learned vs shared-scalar comparisons do not exclude
extra parameter capacity. A matched-capacity nongating control remains a gap
before claiming mechanism-specific improvement; it is not silently invented or
added to this queue. If the module does not beat simple controls, publish the
negative result in the development record and do not rename it to imply a win.

Primary outcomes are paired correction **retention** and reduction of harms,
plus full7 Macro-F1/Accuracy, not F1 alone. Report actual harms/fixes/changed-wrong,
both sweep/reperfusion directions and the contribution of the six-sample class.
If fewer harms come with losing most fixes, the stated goal was not achieved.

## Priority 2: does it use neighbor content and order?

For every eligible clip, compare original, no-neighbor, shuffled and
cross-clinical-record replaced views under a fixed plan. Current clips/labels
are unchanged; valid donor vectors and A predictions move together. Report
native and changed scores on the **same eligible clips**, plus all823 scores,
and actual changed logits/predictions. Clinical-to-clinical replacements avoid
explaining a response by source appearance alone. These interventions support
claims only about their construction, not natural cut-boundary detection.

The center-only `clip_logits`/`clip_visual` controls answer whether a trained
single-clip head suffices. They do not force A fallback. Raw-vs-zero score
factorial controls test an easy-score-shortcut hypothesis, not novelty.
Training diagnostics must establish whether anchor augmentation actually
creates a learning signal before designing another gate.

## Priority 3: recognized temporal competitors

Within each frozen A, compare ordinary TCN, MS-TCN++, ASFormer and the chosen
proposed method. This is a model comparison, not its mechanism ablation. The
canonical mature heads use multi-stage supervision/smoothing; paired
final-CE variants expose that recipe dependence but do not isolate smoothing
from intermediate supervision. Report resource and architecture differences.
Do not call these clip-chain adapters dense surgical-phase reproductions.

The mature models already use temporal structure: [MS-TCN++](https://arxiv.org/abs/2006.09220)
separates prediction generation/refinement and uses dual dilation;
[ASFormer](https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/0183.pdf)
uses hierarchical local attention. Merely combining a visual extractor with a
TCN or giving it an FSN name is not enough to distinguish a new mechanism.

## Priority 4: backbone generality, not absolute-score attribution

Repeat the main matched comparisons on existing VideoMAE and AdaFocus exports.
Report each A-only reference and the proposed/ordinary increment **within that
A**; compare increments descriptively across A. [TUNeS](https://arxiv.org/abs/2307.09997)
shows why feature-extractor training context can materially change downstream
temporal scores. Do not claim different A pretraining/views/optimizers were
equal-budget or that a strong A score is caused by the proposed TCN.

This package explicitly registers12 comparator cells per backbone, not a
winner-search whose largest validation score becomes the paper's method.
Prior module evaluations remain part of the record and are not erased by the
new table. Further visual-model training/exports require a separate fixed
protocol and authorization, not spare GPU availability.

## Not supplied by this development queue

val823 remains repeatedly used development validation, not an independent
test even if the user calls it a test. Whole-system multi-seed replication,
unseen clinical/teaching recordings and actual patient/operation independence
are not established by this single-seed frozen-head suite. Record-cluster
intervals describe conditional development uncertainty only. No prospective
clinical benefit, teaching efficacy, natural boundary detection, statistical
significance or paper acceptance is claimed.
