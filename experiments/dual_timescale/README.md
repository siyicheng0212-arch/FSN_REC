# Dual-timescale FSN action recognition (six-cell ablation)

**Status: code prepared, not trained.** This branch adds an executable, sealed
frozen-feature experiment for the two research questions:

1. **Within one clip:** does pooling its eight sampled VideoMAE tubelet features
   at 1/2/4 temporal scales preserve useful fine-grained evidence?
2. **Between clips:** does an offline TCN use candidate-neighbor context to
   improve classification, and does adding the within-clip branch provide
   additional value?

This is a clip-level seven-class task. It is not temporal action localization
or dense frame-wise segmentation. The VideoMAE backbone and its logits stay
frozen; only lightweight heads are trained. Adaptive frame sampling is
intentionally deferred until the fixed-eight-position multiscale comparison
shows a signal, so this experiment isolates one change at a time.

## Core four-cell ablation plus matched controls

| Cell | Within-clip multiscale branch | Inter-clip TCN | Meaning |
|---|---|---|---|
| M0 | off | off | Original frozen VideoMAE A logits |
| M1S | single-scale mean control | off | Same trainable clip head, but every pooling bin receives the same whole-clip mean |
| M1 | 1/2/4-scale | off | Multi-scale clip-internal representation only |
| M2 | off | on | Inter-clip TCN on mean-pooled global/local evidence and A logits |
| M2S | single-scale mean control | on | Mean-control clip head plus inter-clip TCN |
| M3 | 1/2/4-scale | on | Multi-scale clip branch plus inter-clip TCN |

M1/M3 pool the same eight sampled positions into 1, 2, and 4 equal-width
normalized-time bins (seven pooled bins total). A shared token projection is
applied before pooling, followed by a compact fusion MLP. The clip classifier
starts with a zero correction, so its initial output equals A. M2 uses the
historical 3-block residual TCN dimensions and default dilations [1, 2, 4],
whose theoretical receptive field is 15 candidate clips. M3 lets the inter-clip
TCN use the fixed mean-pooled evidence together with the learned clip embedding.

The source-protocol structural cuts are reused exactly. The TCN never crosses
an ineligible adjacency or a source/recording cut. Source kind is not a model
input and is used only for post-hoc slices. Source-policy adjacency labels
remain weak metadata-derived labels, not verified natural edit-boundary truth.

M1S and M2S are important parameter-matched controls: they use the same
projection, fusion MLP, and classifier dimensions as M1/M3, but repeat the
whole-clip mean descriptor into all seven bins instead of preserving temporal
subdivisions. This lets M1 vs M1S and M3 vs M2S test whether multi-scale
temporal evidence helps beyond adding a trainable head. The six cells support
factorial contrasts: context effect on A (M2−M0), context effect with a mean
head (M2S−M1S), context effect with a multiscale head (M3−M1), multiscale
effect without context (M1−M1S), and multiscale effect with context (M3−M2S).

## Data and evaluation contract

- Exact existing split: train 7372 / development val 823.
- Reuse the sealed VideoMAE frozen features and the already validated source
  protocol. The script checks index hashes, manifests, frozen A replay, source
  seals, per-clip feature files, and chain coverage before creating output.
- No test split is read, generated or evaluated. val823 is a repeatedly used
  development set, not an independent test.
- All trained variants use the same seed, AdamW settings, batch size, epoch
  cap, early-stopping rule and full seven-class validation Macro-F1 checkpoint
  selection. Ties preserve the earliest epoch.
- The output directory must be fresh, private, and outside the repository,
  input feature cache and source-protocol directory. It contains checkpoints,
  per-clip prediction rows and paths; do not commit or publish it.
- The runner is a direct training entry point, not a GPU-ownership guard.
  Inspect active jobs and GPU usage before running it on a shared server.

## Run tests first

From the repository root, in the same Python environment used for the existing
VideoMAE evidence experiments:

~~~bash
python -m unittest tests.test_dual_timescale -v
~~~

## Run the six cells serially

Replace the example paths with the existing sealed paths on your server. Keep
output outside the repository and choose a path that does not already exist.

~~~bash
python -m experiments.dual_timescale.run \
  --index /absolute/path/to/videomae/features/index.jsonl \
  --source-protocol-dir /absolute/path/to/sealed/source_protocol \
  --output /absolute/path/to/new_private_output/dual_timescale_seed42 \
  --device cuda \
  --variants M0,M1S,M1,M2,M2S,M3 \
  --seed 42 --epochs 30 --patience 5 \
  --batch-size-segments 16 --embedding-dim 128 \
  --hidden-dim 64 --blocks 3 --dropout 0.1 \
  --dilations 1,2,4 --lr 0.001 --weight-decay 0.01 \
  --grad-clip-norm 5
~~~

Run all six cells on one GPU serially; do not launch duplicate jobs. The output files
are public_safe_summary.json (aggregate metrics), private_protocol.json,
status.json, and per-variant folders with best.pt, history.json,
public_safe_summary.json and private_val_predictions.jsonl. Keep all private
artifacts local.

## How to interpret the result

- M1 vs M1S: multi-scale pooling gain beyond a parameter-matched mean-pooling head.
- M3 vs M2S: multi-scale pooling gain when inter-clip context is enabled.
- M2 vs M0: inter-clip TCN gain on the original A evidence.
- M2S vs M1S: context gain with the single-scale mean-control head.
- M3 vs M1: context gain with the multi-scale clip head.
- Do not claim multi-scale pooling helps unless M1 beats the matched M1S control
  and/or M3 beats the matched M2S control. Do not claim context helps unless
  the relevant context-on/off pairs improve. If only M3 beats M0, report the
  combined interaction rather than attributing the gain to both modules.
- Use the duration and source slices only as diagnostics. Groups may omit one
  or more action classes, so compare present_class_macro_f1 as well as the
  fixed seven-class Macro-F1 and always report slice support.
- Repeat prespecified seeds (for example 42, 10, 1217) before making robustness
  claims. This initial implementation is a single-seed exploration.
- No performance claim is made by this code commit. It has not started training.
