# Directly supervised within-clip pair probe: training-internal diagnostic

Status: code and protocol prepared; no claim of seven-class or clinical-role
performance. This is a go/no-go experiment before any further full-data FSN
architecture search.

The previous IEA and directional residual heads were not sufficiently active:
on their best checkpoints, their corrections changed almost no 823-clip
development-validation predictions. Increasing a residual gate without
new evidence could simply exchange the two sweep/reperfusion error
directions. FSN-v3 obtained a larger gain from *other clips' annotated
order*, which edited online footage cannot be assumed to preserve.

Private review of 24 deliberately selected sweep/reperfusion errors across
seven sources and three duration bands found heterogeneous observability:
hands often appear, but the needle/contact region is frequently obscured;
patient movement is visible in some clips but absent or indeterminate in
many sampled stills; action-explicit on-screen text and internal cuts also
occur. These 24 error-enriched clips do **not** estimate population rates
or provide clinician/patient ground-truth labels. The private contact
sheets, clip IDs, paths and per-example notes are not in this repository.

`PairEvidenceProbe` compares two **equal-capacity** small classifiers of
the existing primary labels `扫散` versus `再灌注` on the *same* 36 uniformly
sampled 224px RGB frames. One receives downsampled RGB plus signed adjacent
frame differences; the other receives RGB plus zero difference channels.
Both have the same spatial attention, temporal encoder, optimizer, class
weights, seed, and early stopping. Unlike the zero-initialized residual,
each model has a direct two-class cross-entropy loss, so its evidence path
must learn or fail in a measurable way. A fixed time-shuffled, independently
trained arm is provided for a negative control if the motion input shows a
positive train-internal signal.

All structure decisions use the existing *group-disjoint split of the
original training set alone* (6,586/786 clips before filtering to the pair).
The original 823-clip development validation set is excluded from this
probe; no independent test is accessed. The frozen manifest hashes,
pair counts, and hyperparameters are in
`configs/fsn_pair_probe_protocol.json`. The first run compares two modes
at seed 42 on separate GPUs. A single-seed lead is not a success claim:
require confirmation with more train-internal seeds, two-directional error
rates, source/duration slices, and action-text controls before promoting
anything to a full seven-class model.

This probe does **not** infer a clinician hand or patient activity state.
For that mechanism, training-only expert annotations must separately mark
both activities as potentially simultaneous, each role's visibility/unknown
state, and usable ROIs. A visually ambiguous or off-frame activity cannot
be recovered just by increasing model size. An oracle ROI experiment, if
used, must be distinguished from a deployable automatic-ROI result.
