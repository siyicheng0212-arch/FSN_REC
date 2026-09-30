# FSN-v3 validation report

## Scope and protocol

FSN-v3 keeps the official Uni-AdaFocus visual network unchanged and adds an
empirical annotated-segment-order decoder over clips from the same source record. The decoder
fits a Laplace-smoothed seven-state transition prior using **training labels
only** and combines it with frozen visual log probabilities using Viterbi
decoding.  The transition weight is fixed at `1.0`; singleton records retain
their independent visual prediction.

The derived protocol contains 7,372 training clips and 823 validation clips.
It combines the former train and validation splits for training and treats the
former test split as validation.  Consequently, this experiment has **no
independent test set**, and every result records `test_metrics: null`.  All
splits remain group-disjoint.

## Evidence-led design

Three hypotheses were checked before implementing FSN-v3:

1. **Late local/global feature interaction.** FSN-v1/v2 did not improve the
   original network.  Checkpoint audits showed their additional branches made
   negligible logit contributions and changed no predictions in the audited
   validation subset.
2. **Missing temporal phases.** Rejected.  The original deterministic sampler
   selected 12 unique frames for every validation clip, with no sample having
   fewer than two frames in any early/middle/late third.
3. **A drifting target leaving the selected patch.** Rejected.  On the
   stratified long-training audit, the original patch covered 98.5% of frame
   area on average and 99.3% of measured motion energy.  A trajectory module
   would therefore not address the observed error.

The training annotations instead showed a strong procedural signal.  For
example, the most common next actions after disinfection, insertion, and
manipulation occurred with rates of approximately 75%, 81%, and 89%,
respectively.  This motivated the train-only transition decoder.

## Three-seed results

All values below are validation metrics from the same frozen manifests and
training protocol.

| Seed | Independent visual macro-F1 | FSN-v3 macro-F1 | Delta | Accuracy delta |
|---:|---:|---:|---:|---:|
| 42 | 0.79593 | 0.80709 | +0.01116 | +0.01458 |
| 123 | 0.75744 | 0.82369 | +0.06626 | +0.04860 |
| 2026 | 0.77449 | 0.79871 | +0.02422 | +0.02430 |

- Mean visual macro-F1: **0.77595** (sample SD 0.01929)
- Mean FSN-v3 macro-F1: **0.80983** (sample SD 0.01272)
- Mean paired macro-F1 gain: **+0.03388** (sample SD 0.02879)
- Mean accuracy gain: **+0.02916** (sample SD 0.01752)
- Every seed improved; no seed regressed.

With only three seeds, the descriptive Student-t interval for the macro-F1
gain is wide and crosses zero.  These results therefore support FSN-v3 as the
formal candidate but do not establish classical statistical significance.

## Error behavior

The number of sweep-to-reperfusion errors changed as follows:

- seed 42: 30 → 24
- seed 123: 43 → 24
- seed 2026: 41 → 31

The reverse reperfusion-to-sweep direction increased slightly in each seed,
so both directional errors and per-class metrics must be reported rather than
only their sum.  Support-aware duration metrics are also retained because the
long-duration slice does not contain all seven classes; its fixed seven-class
macro-F1 alone is misleading.

## Reproducibility and audit artifacts

The evaluator records:

- training and validation manifest SHA-256 hashes;
- official and fine-tuned checkpoint SHA-256 hashes;
- transition matrix, smoothing, and fixed transition weight;
- baseline and sequence-aware metrics from the same visual logits;
- clip-level visual logits and both predictions;
- source and duration slices, including present-class macro-F1;
- explicit null independent-test metrics.

Independent checkpoint re-evaluation differed from the checkpoint-selection
metric by at most 0.00082 macro-F1 across the three seeds.  Server regression
tests passed, as did a balanced 28-training/14-validation end-to-end smoke.

Reproduce one seed with:

```bash
python -m experiments.evaluate_sequence \
  --manifest-dir processed_server/manifests_trainval \
  --cache-dir /path/to/full_cache_36f224_trainval \
  --official-checkpoint /path/to/sthv2_p128_8and12.pth.tar \
  --visual-checkpoint /path/to/original/seed_42/best.pt \
  --output-dir /path/to/sequence_seed42 \
  --transition-weight 1.0 \
  --smoothing 1.0 \
  --seed 42
```

Aggregate formal seeds with `python -m
experiments.aggregate_sequence_results ...`.  The aggregator rejects mismatched
manifests, mismatched transition settings, or any non-null independent-test
metrics.

## Interpretation and limitation

The improvement comes from modeling the observed order of annotated clips in
each media file, not from adding another visual attention block. This order
must **not** be called a validated clinical treatment sequence: some online
media are edited compilations, and the file's neighboring segments need not
be neighboring clinical actions. The frozen-logit controls in
`reports/FSN_V3_MECHANISM_AUDIT.md` show that directed adjacency, rather than
only class frequency or generic same-class smoothing, accounts for most of the
observed gain. Isolated clips are unchanged. Deployment on unsegmented
continuous video would additionally require a boundary proposal or temporal
localization component, which is not evaluated here.
