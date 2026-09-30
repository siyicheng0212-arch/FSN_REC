# Three-model FSN comparison

This directory contains a model-neutral real-video pilot and the unified
evaluation path for:

1. `adafocus_original` — official Uni-AdaFocus-TSM computation path.
2. `adafocus_fsn` — the same initialized backbone plus the FSN adapter and
   local-context interaction.
3. `mvit_v2_s_reference` — a pure-PyTorch local runtime reference.

The formal recent-model comparison is **VideoMamba-Ti 8f/224 (ECCV 2024)**.
It is not used on this Intel Mac because its official selective-scan stack is
CUDA-only.  MViT-V2-S exists here solely to prove that a third independent
video architecture can use the same data/evaluator before moving the run to a
Linux NVIDIA machine.

## What has been run locally

- Full external-drive inventory and annotation/video linkage.
- Balanced real-data pilot: 4 clips per class per split, 84 clips total.
- Fail-fast decode and opaque frame caches: 8 RGB frames at 224×224.
- CPU forward/backward for all three models.
- Exact baseline equivalence before training (`max_abs_logit_difference = 0`).
- One native training step plus full validation/test.
- One full balanced training epoch plus full validation/test.
- Ten-step one-real-clip overfit sanity check.

These are implementation checks, not publishable performance estimates: the
pilot is tiny and models use random initialization.

## Reproduce the local pilot

Create the environment dependencies, build/cache the local-only pilot as
described in `PILOT_DATA.md`, then run:

```bash
.venv/bin/python -m experiments.run_three_models \
  --epochs 1 --batch-size 1 --cpu-threads 6 \
  --output-dir experiments/results/pilot_one_epoch

.venv/bin/python -m experiments.overfit_sanity --steps 10
```

Generated manifests contain absolute private video paths and caches contain
decoded frames. `experiments/.gitignore` excludes manifests, caches, results,
and Python caches from publication.

## Formal GPU protocol

See `configs/three_model_protocol.json`.  The two AdaFocus models must start
from the exact same official checkpoint.  Validation macro-F1 chooses the
checkpoint; test is evaluated once.  Save clip-level logits and report class,
source, and duration slices from the same prediction files.

For full AdaFocus training, first create the shared 36-frame cache with
`python -m experiments.full_data cache`, then launch
`python -m experiments.train_adafocus --variant original ...` and
`--variant fsn ...` on separate GPUs. Both commands must use identical
manifests, checkpoint, seed, batch size, accumulation, and optimizer settings.

Formal manifests retain all positive-duration, canonically labelled actions.
Clips shorter than 0.1 seconds are QC-flagged rather than automatically
discarded. When a clip contains fewer than 36 source frames, FFmpeg
deterministically repeats the nearest decoded frames to keep the input shape
fixed. Cache metadata records `pixel_unique_frames` and
`pixel_repeat_fraction`; primary results must be accompanied by a sensitivity
analysis with these very short clips excluded.

## Procedure-aware sequence decoder

`experiments.evaluate_sequence` adds a train-only seven-state transition prior
to a frozen original AdaFocus checkpoint.  Clips are grouped by `record_id`,
ordered by their annotated timestamps, and decoded with Viterbi.  Singleton
records keep the independent visual prediction.  The transition weight is
fixed at `1.0`, so validation labels are never used to fit or tune the prior.

```bash
python -m experiments.evaluate_sequence \
  --manifest-dir processed_server/manifests_trainval \
  --cache-dir /root/autodl-tmp/full_cache_36f224_trainval \
  --official-checkpoint /root/autodl-tmp/checkpoints/sthv2_p128_8and12.pth.tar \
  --visual-checkpoint /path/to/original/seed_42/best.pt \
  --output-dir /root/autodl-tmp/formal_trainval_results_fsn_v3/seed_42
```

The output records manifest/checkpoint hashes, the fitted transition matrix,
independent and sequence-aware metrics, support-aware duration slices, and
clip-level predictions.  Under the derived train-plus-validation protocol,
the 823 held-out clips are called **validation** and `test_metrics` remains
`null`; this run does not create an independent test set.

After all formal seeds finish, aggregate matched baseline/sequence results:

```bash
python -m experiments.aggregate_sequence_results \
  /path/to/sequence_seed42/result.json \
  /path/to/sequence_seed123/result.json \
  /path/to/sequence_seed2026/result.json \
  --output /path/to/three_seed_summary.json
```

The aggregator fails closed if manifests or transition settings differ, or if
any result contains independent-test metrics.  With three seeds it reports a
Student-t interval as descriptive evidence and explicitly warns that the
interval is unstable at this sample size.

## FSN-v4 structured-decoder comparison and v3 ablations

FSN-v4 is a **candidate**, not a validated replacement for v3.  It reuses
exactly the same frozen original visual logits and train-only labels.  The
first-order v3 decoder is compared with a second-order transition decoder
(`P(y_t | y_{t-2}, y_{t-1})`) and a semi-Markov decoder that models same-class
run lengths.  All use the predeclared weight `1.0` and Laplace smoothing
`1.0`; there is no new visual backbone or additional visual training.

First run `experiments.ablate_sequence_prior` on the three saved v3 prediction
files and their matching result files.  It compares visual-only classification,
uniform/start-only/transition-only/full first-order priors, two deterministic
sequence-order shuffles, and per-record mean-logit pooling.  The order shuffles
are negative controls; no validation label is used to fit a prior.  Then run
`experiments.compare_structured_decoders` on the same files to compare v3
bigram, trigram, and semi-Markov decoding.  Both commands require
`--train-manifest`, `--validation-manifest`, three `--prediction SEED=PATH`
arguments, three matching `--sequence-result SEED=PATH` arguments, and
`--output`.  The structured comparison additionally accepts fixed `--weight`
and `--smoothing`.  Input manifest hashes, seed IDs, labels, stored v3
decisions, and `test_metrics: null` are checked before any result is written.

Before designing another visual/transition module, run
`experiments.audit_sequence_errors` with the same train/validation manifests,
three `--prediction SEED=PATH` files, and three matching
`--sequence-result SEED=PATH` files.  Set `--output-dir` to a **private path on
the data server**.  The script verifies frozen inputs, reproduces v3 decisions,
and writes `summary.json` plus `private_review_queue.jsonl` containing clip IDs,
original media paths, timestamps, visual confidence, adjacent labels, and
inter-clip gaps.  Review the selected source-video intervals manually before
asserting a cause.  Never upload the queue, video frames, or annotations to
GitHub.  In particular, a large gap between adjacent annotated clips is not
evidence of an observed action boundary.

These comparisons use the 823-clip **validation** split already examined for
v3.  They are exploratory diagnostics, not independent-test results or an
unbiased architecture-selection estimate.  A later definitive v4 claim would
need a train-only internal selection protocol and untouched external test data.

## Probability calibration and record-level uncertainty

`experiments.audit_calibration` evaluates the same frozen FSN-v3 predictions
without fitting any calibration parameter on validation.  Original visual
probabilities come from softmax logits.  The sequence probabilities are
computed with the forward-backward algorithm using the transition prior fitted
from training labels.  Final sequence decisions remain Viterbi labels, so
top-label calibration uses the marginal probability assigned to the selected
Viterbi label.  NLL and multiclass Brier score evaluate the complete marginal
distribution.

The audit reports ten-bin expected calibration error, reliability bins,
confidence, entropy, high-confidence mistakes, and selective error rates.
It also resamples **source records**, preserving the dependence among clips
from the same video.  Its percentile interval describes held-out-record
variation conditional on the three fitted seeds.  It does not include new
training-seed variation, external-site variation, or the effect of choosing
checkpoints on this validation set.  See
[Guo et al. (ICML 2017)](https://proceedings.mlr.press/v70/guo17a.html) for
the calibration motivation and
[Rabiner (1989)](https://www.cs.cmu.edu/~guestrin/Class/10701-S06/Handouts/Readings/hmms-rabiner.pdf)
for forward-backward inference.

```bash
python -m experiments.audit_calibration \
  --train-manifest /path/to/manifests_trainval/train.jsonl \
  --validation-manifest /path/to/manifests_trainval/val.jsonl \
  --prediction 42=/path/to/sequence_seed42/validation_predictions.jsonl \
  --prediction 123=/path/to/sequence_seed123/validation_predictions.jsonl \
  --prediction 2026=/path/to/sequence_seed2026/validation_predictions.jsonl \
  --sequence-result 42=/path/to/sequence_seed42/result.json \
  --sequence-result 123=/path/to/sequence_seed123/result.json \
  --sequence-result 2026=/path/to/sequence_seed2026/result.json \
  --output /path/to/probability_and_cluster_audit.json
```
