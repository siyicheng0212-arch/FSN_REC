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
