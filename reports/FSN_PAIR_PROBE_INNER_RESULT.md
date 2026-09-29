# Train-internal motion-pair probe: a useful negative gate

Status: **one-seed diagnostic completed, not a full FSN model or paper
performance result**. All three arms exited normally and used the same
video-group-disjoint split of the *original training set*. There were 3,770
training and 466 inner-validation clips labeled sweeping/reperfusion;
the latter contained 357/109 clips. The repeatedly examined 823-clip
development validation set was not used to select the probe, and no
independent test was run. Frozen manifest hashes and sanitized curves are in the
[aggregate JSON](../public_results/fsn/pair_probe/seed42_internal_comparison.json).

| Seed 42 pair-only system | Best inner Macro-F1 | Accuracy | Sweep F1 | Reperfusion F1 | Sweep→reperfusion | Reperfusion→sweep |
|---|---:|---:|---:|---:|---:|---:|
| Small RGB + signed frame difference | 0.6031 | 0.7146 | 0.8135 | 0.3927 | 67 | 66 |
| Same-capacity RGB only | 0.4338 | 0.7661 | 0.8676 | 0.0000 | 0 | 109 |
| Same-capacity shuffled-time RGB + difference | 0.4338 | 0.7661 | 0.8676 | 0.0000 | 0 | 109 |
| Existing pretrained Original, conditional pair logits | **0.7522** | **0.8348** | **0.8952** | **0.6091** | **28** | **49** |

The two small controls predicted *every* clip as sweeping. The correctly
ordered difference stream did learn something the controls did not under
this optimization protocol, but its 67/66 two-way pair errors were much
worse than Original's 28/49. The Original comparison is **not** equal
capacity or equal pretraining; it is the existing practical standard the
new stream would need to complement, not a fair claim that motion as a
modality is intrinsically worse.

This pilot does not justify another full-data seven-class architecture
run. Its motion stream is a generic pixel-level signal and cannot be
called clinician-hand or patient-activity evidence. A private,
error-enriched review of 12-frame contact sheets for 24 clips across all seven sources found
obscured needle/contact regions, uncertain or visually stationary
patient activity, action-explicit subtitles, and editing/cut risk; those
observations from a purposive static-frame sample do not establish activity
co-occurrence or its absence, do not estimate dataset prevalence, and cannot
become role training labels. The existing label policy maps one primary class per
clip but has no precedence rule for simultaneous sweeping/reperfusion.

For a defensible task-specific role method, first annotate an independent
training-only sample with clinician/needle motion, patient target-region
activity, each role's visibility/unknown status, overlap, and ROI tracks;
two clinical reviewers must resolve the primary-label rule and report
agreement. Then test an oracle-ROI upper bound **and** a deployable
automatic-ROI model on train-internal held-out videos against Original,
equal-budget whole-frame, and text-masked controls. If the patient action
is not visible reliably, no RGB-only role architecture can establish that
mechanism. FSN-v3 remains a strong *annotated-segment-order* diagnostic,
not a single-clip or verified clinical-workflow method.
