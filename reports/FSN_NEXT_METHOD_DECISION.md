# FSN method decision: evidence gate before a new model

Status: **hypothesis, not a validated method or performance claim**.

## What is established

The original Uni-AdaFocus visual model independently predicts seven actions.
FSN-v3 leaves its visual network unchanged and adds a Laplace-smoothed,
first-order action prior learned from training annotations, followed by
Viterbi decoding within each chronologically ordered source record.  This is
a *soft* prior: sweeping may continue, move to reperfusion, or move directly
to needle removal.  It does not recognize why a particular patient changes
action, and it does not enforce a fixed clinical script.

Across three seeds, v3 improved mean macro-F1 from 0.7760 to 0.8098 over
independent visual predictions.  Sweeping-to-reperfusion errors fell from
114 to 79, while reperfusion-to-sweeping errors rose from 86 to 93.  The
mean reperfusion precision improved from 0.507 to 0.593 while recall fell
from 0.583 to 0.569.  The mechanism and the trade-off must both be reported.

This combination is not a novel sequence algorithm: prior surgical workflow
research has already combined visual predictions, train-counted transition
matrices, and Viterbi decoding [Cadène et al., 2016](https://arxiv.org/pdf/1610.05541).
Generic boundary prediction is also established [ASRF, WACV 2021](https://openaccess.thecvf.com/content/WACV2021/html/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.html).

## Why a fixed flow prior may fail

FSN action order can branch.  A clinical case report describes patient
reperfusion activity while the clinician sweeps the needle
[source](https://pmc.ncbi.nlm.nih.gov/articles/PMC11666207/); this is a
reason to verify whether the dataset's single-label annotations contain
physically overlapping actions, not proof that any particular clip is
mislabeled.  The local pre-materialization 823-clip manifest also contains
many gaps between adjacent annotated clips.  Therefore, two adjacent labels
need not expose the actual transition instant in their endpoint frames.
The locally held manifest has matching class supports but a different byte
hash from the server's derived manifest, so these gap counts are diagnostic
only until the exact frozen clip IDs are checked.

## Required visual error audit

The three saved `validation_predictions.jsonl` files are currently only on
the unavailable data server.  Aggregate confusion matrices cannot identify
which original videos to inspect.  `experiments.audit_sequence_errors`
prepares a private, reproducible review queue, selecting clips where v3
corrects or creates each sweep/reperfusion error direction across seeds.
For each selected original interval, inspect the source video and record:

1. Whether needle/practitioner-hand motion and patient-limb motion are both
   visible, only one is visible, or neither is reliably visible.
2. Whether the annotated target is an onset, offset, concurrent-action
   interval, ordinary interior clip, or disputed boundary.
3. Whether the visual model was confident and overruled by the static prior.
4. Whether an unannotated gap separates the surrounding clips, and whether
   the action between them is observable.
5. Whether the clip appears to have a label or synchronization problem.

Do not build a boundary-conditioned model until this review demonstrates a
repeatable, visually observable error mechanism.  Keep clip identities,
media paths, predictions, frames, and review notes off GitHub.

## Candidate only if the audit supports it

If the relevant roles are visible and labels are consistent, test a small
**role-conditioned, gap-aware transition residual** on top of the v3 prior.
The two candidate evidence streams are practitioner/needle sweeping motion
and patient reperfusion motion.  Their onset/offset evidence would adjust,
not forbid, each allowed transition.  For long unobserved gaps, the model
must not pretend to have seen the boundary and should fall back to the
static prior or independent visual evidence.  Initialize the residual to
zero so the candidate is exactly equivalent to v3 before training.

This is not automatically novel: surgical gesture models, generic boundary
heads, and conditional sequence models already exist.  The specific
contribution would have to be the empirically justified FSN role evidence,
its handling of branching/co-occurrence and unobserved gaps, and a controlled
benefit beyond those generic alternatives.

## Fair comparison required for a method claim

Use the **same per-clip visual evidence**, source-record grouping, and
training budget wherever applicable:

- Original independent visual predictions.
- Fixed v3 HMM/Viterbi and stronger generic transition/duration baselines.
- A generic temporal learner and a generic boundary-aware learner.
- The candidate with both role streams, each stream removed, and each stream
  time-shuffled as a negative control; also compare a gap-only gate.

Predeclare overall macro-F1 plus both normalized directional errors
`C[sweep,reperfusion]/N_sweep` and
`C[reperfusion,sweep]/N_reperfusion`, class-specific precision/recall/F1,
action-onset/offset performance, long/short and source slices, and
record-cluster uncertainty.  A higher overall score that worsens the
reverse direction does not establish that the proposed visual mechanism
resolved the stated problem.  If the role cues are absent or the annotation
ontology collapses simultaneous actions, address the data/label definition
before adding model complexity.

## Honest present-tense contribution statement

At present, the defensible contribution is a carefully curated FSN
fine-grained recognition task/benchmark, a transparent procedure-prior
baseline, and controlled analysis of when sequence context helps and harms.
The role-conditioned model and superiority over generic temporal methods
remain **unproven** until the private error review and matched experiments
are completed.
