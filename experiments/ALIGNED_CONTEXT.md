# Aligned global/local context: a four-GPU development experiment

This experiment trains new encoder modules directly. It does **not** run the
frozen-feature complementarity probes or the larger CVM backbone matrix.
The candidate is a local ResNet `layer2` residual interaction that retains
global frame/grid tokens. A plain interaction, an interaction with known
sampling/crop correspondence, and a local-only capacity control are compared
against Original. There is no promise of improvement or established novelty.

## Comparisons and scheduling

| GPU | Independent parallel task |
| --- | --- |
| 0 | `original` |
| 1 | `context_plain` |
| 2 | `context_aligned` |
| 3 | `local_capacity` |

Each task is an independent single-GPU training run, not four-GPU DDP. All
four tasks start in parallel; the capacity control does not wait for Original.
Each task's process exit code and actual result artifacts are checked. A failed
task preserves its output; other independent tasks may finish. No restart,
continuation, multi-seed wave, test evaluation,
or automatic follow-up is performed.

All four models start from the **same official SSv2 checkpoint**, with the same
seed and seven-class head initialization. A previously finetuned Original
checkpoint is not used to initialize only the new variants. Global sampling,
local sampling, cropping and Original's stop-gradient classification strategy
remain the same across variants; the existing local auxiliary classification
loss supplies training gradients to the new local module.

`context_plain` and `context_aligned` use the same trainable interaction
structure; the aligned version adds the declared time/geometry bias. Sampling
index correspondence is not verified source PTS or physical velocity.
`local_capacity` removes global input and controls added capacity. Report its
actual parameter and compute differences; capacity matching is not compute
matching. All new branches use a zero-initialized residual output projection.

## What the module computes

The local layer-2 tensor has 512 channels. GroupNorm and a 1x1 projection
reduce it to 64 channels; adaptive 3x3 pooling produces nine local query
tokens per selected frame. The eight global MobileNet feature maps keep their
frame dimension, are pooled to 3x3, and project from 1,280 to 64 channels.
Content attention lets each local token read the 72 global tokens. The
attended features project back to 512 channels, resize to the layer-2 map,
and add to the original features before layer3/layer4.

The aligned version adds two **fixed, soft** distance penalties to attention
logits: normalized selected-frame distance (scale .25) and spatial distance
between global grid centers and crop-mapped local grid centers (scale 1).
Neither prior adds parameters or excludes tokens. Disabling both priors
reproduces the plain module with the same weights. This is a correspondence
hypothesis, not a measured motion model. Coordinates are nominal feature-grid
centroids obtained from Original's crop affine transform; the global CNN's
large receptive fields prevent interpreting them as exact anatomical ROI
locations.

All three new modules add **164,864 active parameters**. The capacity control
uses within-frame local attention and replaces the global projection with a
64-to-640-to-64 local MLP of the same parameter count. For 36/8/12 frames and
128-pixel patches, Original has 27,295,907 total parameters and 27,208,867
finetune-trainable parameters after its partial-BN freeze; each candidate has
27,460,771 total and 27,373,731 finetune-trainable parameters. Classifier-only
warmup has fewer trainable parameters.

## Fixed protocol

Seed 42; five classifier-head warmup epochs at LR .001; at most 100 finetune
epochs, patience 10; batch 4, accumulation 16 (effective batch 64); workers 4;
SGD momentum .9, base LR .002, weight decay .0005; global/spatial/temporal LR
ratios .5/.2/.2; context LR ratio 1; square-root inverse seven-class weights;
gradient clipping 20; CUDA bf16; context dimension 64, grid 3, time scale .25,
spatial scale 1. Input/global/local frames are 36/8/12 and patch size is 128.

The launcher fixes this protocol. Do not silently alter batch size or just one
variant after an OOM. Preserve the failed run and report it. If a new protocol
is necessary, all four comparisons need a fresh, clearly named suite.

## Inputs are reused, never rebuilt by this launcher

Confirm existing inputs and export these variables locally on the server.
Actual private filesystem paths are never included in the public repository:

```text
FSN_PYTHON: verified CUDA Python executable
FSN_WORKTREE: isolated source worktree
FSN_MANIFEST_DIR: existing internal train/val manifest directory
FSN_CACHE_DIR: existing 36f224 cache directory
FSN_CHECKPOINT: existing official SSv2 initialization file
FSN_OUTPUT_DIR: fresh formal output outside the worktree
FSN_SMOKE_DIR: fresh private smoke/report/launcher-log directory
```

The launcher only accepts the existing internal protocol:

| Input | Count | SHA256 |
| --- | ---: | --- |
| train manifest | 6,586 | `f0fc0ace561cd59c471c86d6ec674144de0d5ede6acf447a5a021787538ebf0a` |
| val manifest | 786 | `d6449196c3798a233a593db19da9045c92e02ba62bb6fa5a55d41bf118ee6580` |
| official initialization | — | `2dea5c15ce23b3549aeab977774649f0ce8dcbc5637d5d1d1319efc019896fc3` |

All seven labels and real group identifiers are required. Train/val clip and
group intersections must be empty. The 36f224 cache is validated for every
record. If an internal validation record is stored in the original train cache,
the resolver records its storage split privately without rewriting either
manifest. An existing invalid role-path cache is an error; it is not silently
substituted. A uniquely valid alternative storage path may be used only when
the role-path cache is absent.

The protocol records the original manifest SHA, cache storage mapping SHA, and
an inventory SHA over **all cache metadata hashes and array sizes/mtimes**.
The inventory is not a full-pixel-content checksum. It also records commit,
runtime source hashes, official checkpoint SHA, actual hardware and smoke
report SHA. These files contain private server paths and stay on the server.

## Actual smoke precedes training

Local validation used torch 2.8.0+cpu / torchvision .23.0+cpu: 40 new tests
and 34 affected model/training/metric regression tests passed. At full
36/8/12/128 dimensions with a synthetic CPU batch of one, all three candidates
matched Original at initialization and passed three native-loss steps with
gradient clipping 20. These checks used random initialization, not the private
official checkpoint; tiny random MobileNet features caused residual-RMS
underflow and do not establish useful learning. The inherited broad discovery
also contains `test_dedupe_videos.py`, whose imported script is absent from
the parent branch; that unrelated data-maintenance test is outside this run's
targeted checks.

Run the relevant CPU tests first:

```bash
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$FSN_PYTHON" -m unittest discover \
  -s tests -p 'test_aligned*.py' -v
```

Next run the dedicated smoke CLI on actual
cached data and all four CUDA cards:

```bash
cd "$FSN_WORKTREE"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$FSN_PYTHON" -u -m experiments.train_aligned_context \
  --smoke-only \
  --manifest-dir "$FSN_MANIFEST_DIR" \
  --cache-dir "$FSN_CACHE_DIR" \
  --checkpoint "$FSN_CHECKPOINT" \
  --smoke-report "$FSN_SMOKE_DIR/smoke.json"
```

Use a fresh smoke location. The smoke executes three real bf16 optimizer steps
for **each** variant on its assigned GPU, checks finite losses, staged module
gradients and initialization equality against Original. Report generation is
not itself evidence of success: all four entries must pass. Changes to runtime
source or inputs invalidate a previous smoke report. The formal launcher also
checks idle physical GPUs 0, 1, 2 and 3, actual RTX3090 names and bf16 support.

This repository's local tests do not establish that private-data CUDA smoke or
formal training has passed. Verify those on the actual server before reporting
them as completed.

## Read-only plan, then explicit execution

The shell wrapper requires `FSN_PYTHON` to identify the verified CUDA Python.
Required paths are CLI
arguments, not inferred silently:

```bash
bash scripts/run_aligned_context_4gpu.sh \
  --manifest-dir "$FSN_MANIFEST_DIR" \
  --cache-dir "$FSN_CACHE_DIR" \
  --checkpoint "$FSN_CHECKPOINT" \
  --output-dir "$FSN_OUTPUT_DIR"
```

This prints the checked plan without creating the output. After smoke success,
append `--execute --smoke-report "$FSN_SMOKE_DIR/smoke.json"`. A clean, committed,
isolated worktree is required for execution. The entire formal output must be
absent, including an empty directory. Atomic directory creation and
`.launch_once` prevent two launchers from claiming it. Never run `git pull`,
change code or update packages while the suite runs.

If an older three-GPU suite has already started, preserve its processes,
protocol, outputs and status. Inspect it read-only before planning anything
else; do not automatically interrupt, duplicate or convert that running suite.

Files saved at the suite root include `run_protocol.json`, `run_commit.txt`,
`.launch_once`, `suite_result.json`, and `launcher_logs/{pids,exit_codes}.tsv`.
Per-model output is `<suite>/<variant>/seed_42/`. `pids.tsv` stores actual child
PIDs when tasks really start, rather than guessed initial PIDs. Monitor all
four GPU assignments, child processes, per-task logs and histories. Exit 0
without valid result artifacts does not count as training completion.

## Analysis and stopping rule

Wait for four valid results. Summarize seven-class Macro-F1 and accuracy,
precision/recall/F1/support, the five fine-action classes, best epoch, time,
parameters/resources, source/duration slices and the two directed
扫散↔再灌注 errors. Inspect module residual and gradient diagnostics and the
same-checkpoint module on/off effects. Distinguish the independently trained
Original from turning off a module in a trained candidate.

The same 786 internal-validation clips select checkpoints. These are
development results, not an independent test or a direct comparison with the
old 7,372/823 protocol. One seed or a higher score does not establish novelty,
a clinical mechanism or a paper conclusion. Three comparisons are needed:
aligned vs Original, aligned vs plain interaction, and aligned vs capacity
control. Ordinary cross-attention already has prior work; correspondence is a
hypothesis under test. Perturbation audits only establish sensitivity, not
clinical causality.

After the seed42 report, stop. Do not restart old experiments, merge main, or
upload video, raw annotations, weights, caches, features, per-clip predictions
or original logs. Only deidentified aggregate reports may later be shared.
