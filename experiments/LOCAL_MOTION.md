# Early local correspondence experiments

This branch starts from main `1752592`, not from the old three-path difference
branch. No training result or improvement claim is included. The module is an
independently implemented MotionSqueeze-inspired prototype, not a reproduction
of MotionSqueeze, an optical-flow estimator, or a new proven paper contribution.

## Model and controls

The existing MobileNetV2+TSM global branch sees 8 frames. Its temporal policy
selects 12 frames from the same clip's uniform 36-frame cache. Its spatial policy
chooses a clip-shared rectangle, resized to 128. The local ResNet50+TSM remains
intact, with the optional residual module inserted after `layer2` (512 channels)
and before `layer3`/`layer4`. Existing final and auxiliary heads/losses remain.
Clip ordering across videos is never used.

| Trainer variant | Evidence | Context | Added parameters | Trainable during full finetuning |
|---|---|---|---:|---:|
| `original` | existing Original | existing branches | 0 | 27,208,867 |
| `local_appearance` | per-frame appearance control | none | 68,416 | 27,277,283 |
| `local_motion` | local correspondence | none | 68,416 | 27,277,283 |
| `local_motion_context` | local correspondence | same-clip global grid | 175,104 | 27,383,971 |

Counts use seven classes, bottleneck 64, matching window 3, context grid 2.
They are measured **after `model.train()`**, when official partial BatchNorm
freezes 87,040 parameters. Total parameter counts before that freeze are
27,295,907 / 27,364,323 / 27,364,323 / 27,471,011 respectively. Default first
5 epochs train only the 55,587 classification-head parameters. Full finetuning
then restores the existing trainable backbone/policy parameters and the module.
`--module-warmup-epochs` defaults to zero; any added adaptation epochs must also
be budget-matched for Original and the appearance control.

The matching branch projects normalized 512-channel grids to 64 channels.
For each adjacent pair of **selected** frames it compares a 3x3 displacement
neighborhood using normalized feature correlation. Invalid border candidates
are masked. Softmax/soft-argmax gives horizontal and vertical correspondence
displacements and entropy-based concentration, still at every spatial cell.
These three maps feed a small spatial convolution and a zero-initialized output
projection; its residual enters the remaining pretrained local backbone.
Matching softmax runs in float32 under bf16 autocast. No outer zero multiplier
blocks the projection's first gradient. Earlier branch gradients emerge once
that projection updates; a zero first-step gradient there is expected.

The equal-parameter appearance control replaces correspondence maps with three
means of disjoint groups of projected channels. It uses all projected channels
and the same downstream layers. It is a capacity control, not a FLOPs-matched
control: matching adds nonparametric computation.

The context variant projects a same-clip global feature grid, retains a learned
2x2 spatial descriptor, and averages only over its 8 global frames. Combined
with each local spatial feature it calibrates the residual with `1+tanh(gate)`.
It does not assume that the 8 global frames align with the 12 local frames.
Global context can be disabled independently for auditing. This descriptor
does not establish identities of doctor/patient or instrument roles.

Shared tensors, classifier initialization and the post-construction RNG state
match Original at a common seed. New variants forbid the legacy adapter and
late interaction. The repository model tree takes priority over `third_party`
to avoid importing old server-side copies.

## Four-GPU run

Use **four independent single-GPU runs**, not DDP of one variant. They each
have batch 4 and accumulation 16 (effective batch 64); do not multiply that by
four. Select four idle bf16-capable GPUs and an already working CUDA Python.
The launcher checks unique GPU IDs, input existence and previous run files.
It never silently overwrites a `best.pt`/`result.json`.

```bash
export FSN_PYTHON=/absolute/path/to/working/environment/bin/python
export FSN_MANIFEST_DIR=/absolute/path/to/existing/manifests
export FSN_CACHE_DIR=/absolute/path/to/full_cache_36f224
export FSN_CHECKPOINT=/absolute/path/to/official_ssv2_checkpoint.pth.tar
export FSN_OUTPUT_DIR=/absolute/path/to/new/local_motion_results/seed42
export FSN_GPU_IDS=0,1,2,3
export FSN_SEED=42
export FSN_EPOCHS=100
export FSN_PATIENCE=10
export FSN_BATCH_SIZE=4
export FSN_ACCUMULATION_STEPS=16
export FSN_WORKERS=4
export FSN_LR=0.002
FSN_DRY_RUN=1 bash scripts/run_local_motion_4gpu.sh
mkdir -p "$FSN_OUTPUT_DIR"
nohup bash scripts/run_local_motion_4gpu.sh > "$FSN_OUTPUT_DIR/launcher.log" 2>&1 &
```

`100` is the maximum finetuning budget after the same 5 head-warmup epochs.
Early stopping uses validation macro-F1, patience 10. The common optimizer is
SGD, momentum .9, weight decay .0005, cosine schedule; global learning rate
ratio .5, spatial and temporal policy ratios .2. These match the earlier
100-epoch comparison's LR=.002, not the unrelated old LR=.005 pilot.
Use the **same** manifest checksums, cache, SSv2 weights, class weighting,
augmentation, batch, accumulation, seed and stopping rule for all variants.
The launcher exposes environment overrides only for common settings.

Prefer an existing group-disjoint internal train/validation split for method
development. If reproducing the previously used 7,372/823 protocol, call its
823 clips a development validation set. Do not claim an independent test on
that set. No test is evaluated by default. The trainer validates clip/group
separation and cache request digests; it does not create or change splits.
Do not merge unrelated branches or change crops/sampling in this first test.

Before launch, run a real-data forward/backward smoke check at the formal
36/8/12/128 shape for each variant, and inspect official checkpoint loading.
All shared nonhead tensors must load; random initialization is not an accepted
formal fallback. Check finite losses and positive module `up.weight` gradient.
After one update, inspect earlier module gradients as well. The context gate
has another identity initialization, so context projection gradients can emerge
one update later. This behavior has CPU unit coverage; CUDA bf16 and GPU memory
must be checked on the actual server. If OOM occurs, reduce batch and increase
accumulation identically for all four runs, keeping effective batch 64.

## Outputs and interpretation

Each run writes `<output>/<variant>/seed_<seed>/best.pt`, `history.json`,
`result.json`, `val_predictions.jsonl`, and `load_report.json`. New variants
also automatically write `val_module_audit.json` and paired predictions;
the context variant additionally writes `val_context_audit.json`. The launcher
writes log/PID/exit-code files under `launcher_logs/`.

History records one optimizer-step gradient snapshot per finetuning epoch and
residual/input magnitude diagnostics. Zero initialization is not evidence of
a dead module: inspect later epochs and trained checkpoint residuals. Gradient
snapshots are taken before clipping; a nonzero gradient alone does not prove
useful evidence. The full-checkpoint on/off audit restores torch CPU/CUDA RNG
before each paired forward, avoiding false differences from Monte-Carlo draws.
It reports logits, corrected/harmed cases, class/source/duration metrics and
focus-frame indices/gaps. On/off compares the same trained checkpoint; it is
**not** the independently trained Original and does not prove causality by
itself. Always also compare Original and the appearance control.

```bash
"$FSN_PYTHON" -m experiments.audit_local_motion \
  --checkpoint "$FSN_OUTPUT_DIR/local_motion/seed_42/best.pt" \
  --manifest-dir "$FSN_MANIFEST_DIR" --cache-dir "$FSN_CACHE_DIR" \
  --split val --output-dir local_motion_audits/motion42 --device cuda
```

Matching across the selected frames is not necessarily short-time motion:
nonuniform gaps, duplicate decoded frames, camera shake, occlusion and video
cuts can dominate correspondence. Audit outputs report **requested uniform
bin-center times**, not verified decoder PTS or object velocities. This first
branch leaves decoding/crop policy unchanged to isolate the operator. Passing
unit tests establishes implementation behavior, not FSN accuracy gains.

Do not immediately sweep many hyperparameters. First decide whether the motion
variant beats both controls and its active residual helps more predictions
than it hurts. If both motion variants lose, use learning curves, gradients,
on/off cases and selected-frame gaps to diagnose the outcome before further
training. One seed is a screen, not a paper result. A promising candidate must
be repeated with paired Original/control runs at seeds 123 and 2026 under the
same split/protocol, and evaluated on a genuinely untouched test if available.

## Validation available in this branch

CPU checks cover exact initialization identity, shared state/RNG preservation,
known translation and borders, clip isolation, temporal sensitivity, staged
gradients, context intervention, optimizer coverage and checkpoint buffers.
Run relevant tests using:

```bash
for test in test_local_motion.py test_model_wrappers.py test_local_motion_training.py; do
  "$FSN_PYTHON" -m unittest discover -s tests -p "$test" -v || exit
done
```

Main `1752592` has a pre-existing unrelated discovery error:
`tests/test_dedupe_videos.py` imports absent `scripts.dedupe_videos` (the real
utility is under `data_tools`). This branch does not rewrite that data utility.
No private videos, pretrained weights or trained checkpoints are committed.

References: [MotionSqueeze (ECCV 2020)](https://arxiv.org/abs/2007.09933),
[official implementation](https://github.com/arunos728/MotionSqueeze).
