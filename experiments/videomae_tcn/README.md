# VideoMAE V2 + ordinary residual TCN

This is a **code-ready baseline, not a completed training result**. Reuse the
existing sealed VideoMAE V2 ViT-B/16 frozen feature export. Do not train or load
the visual encoder, export new features, resume a historical TCN, or read test.

## What is implemented

The model class and segment builder are imported directly from the historical
`experiments/relation/tcn_baseline.py`. No R network, source gate, selective
correction gate, transition decoder, or new reference TCN is substituted.

For candidate clip t, the input is:

```
x_t = concat(mean(global_tokens_t), mean(local_tokens_t), A_logits_t)
z_t = A_logits_t + TCN(x_chain)_t
```

The existing VideoMAE exporter calls its last-block spatial **mean** tokens
`global_tokens` and spatial **max** tokens `local_tokens`. Both come from the
same ViT; these names do not denote separate global/local backbones. Each is
8 x 768; temporal pooling produces 768 + 768 + 7 = 1543 input channels. The
seven A inputs are raw logits, not probabilities. There is no additional L2
normalization before the stem.

The preserved architecture is:

1. Linear(1543, 64), LayerNorm, GELU.
2. Three residual blocks: Conv1d(64,64,kernel=3,dilation=1/2/4), symmetric
   padding=1/2/4, GELU, dropout=0.1, Conv1d(64,64,kernel=1), residual add.
   Mask padded positions after each block.
3. Zero-initialized Conv1d(64,7,kernel=1) predicts a correction added to A logits.

There are **148,935 trainable parameters**, all in the TCN. A has zero trained
parameters. The receptive field is at most 15 candidate clips, not 15 frames
or a fixed number of seconds. Symmetric padding means **offline, bidirectional**
recognition, not causal/online recognition. A singleton still runs the legacy
forward and can be corrected after training; it does **not** strictly fall
back to A. This behavior is tested and reported separately.

At initialization the correction is zero, so output equals A. On the first
update only the classifier receives nonzero gradients; gradients must reach
the stem and temporal blocks on later updates. The CUDA precheck verifies
three disposable updates and never inherits their weights/optimizer into
formal training.

## Fixed evidence and protocol

- Full train7372 / val823, seed42; val823 is repeatedly used for development
  and checkpoint selection, **not an independent test set**.
- Train manifest SHA256:
  `093d0adf08dc1d382c0d2e1863a2f4724cb24ebd5b3d0af070e0c76ca989f1d3`.
- Val manifest SHA256:
  `ad785ec18b14b63616582a143384fac649e69e27d142dfcf87c6852f9d3c6f02`.
- Require the existing VideoMAE export schema, complete/frozen status, numeric
  feature SHA for every clip, index/checkpoint/source-audit SHA, original
  manifest SHA, no-test flag, eight-tubelet feature shapes, and the export's
  per-clip A replay assertion. Recompute A accuracy, full seven-class F1 and
  confusion from the index, rather than trusting its headline F1 alone.
- Reuse the sealed source-protocol chain partition and eligible masks exactly.
  All clips remain; every false adjacency separates segments. Both clinical
  C edges and teaching D edges that satisfy the structural rules remain
  connected for this ordinary TCN. C/D labels are not a training target.
- Validate no train/val group, source-video or recording-key overlap; validate
  complete clip/edge coverage, source identity, time base, ordering,
  nonoverlap, and the fixed <=5 second candidate interval cutoff.
  A recording key is **not** a verified patient identity.
- Structural candidate adjacency and source policy do not establish real
  continuity or verified editing boundaries. Cache tubelet positions are not
  independently verified original-video PTS.

The reused segment builder validates the sealed layout; this program does not
rebuild candidate links, reclassify sources, or randomly reshuffle the split.
The source artifacts must already include the confirmed policy in which FSN
is a network/teaching source.

## Training and evaluation

Fixed settings: AdamW lr=0.001, weight_decay=0.01; batch=16 **variable-length
segments**, not 16 individual clips; hidden64/three blocks/dropout0.1;
unweighted valid-clip seven-class cross-entropy; clip_grad_norm=5; up to 30
epochs, patience5. Every train clip is visited once per epoch. Select best by
full-seven-class val Macro-F1; ties keep the earlier checkpoint. Accuracy is
reevaluated from that **same best-F1 checkpoint**. This adopts the historical
ordinary TCN recipe, but does not claim equal visual pretraining/compute budget
between VideoMAE and the old A. Different input dimensions change TCN capacity.

Formal TCN and its smoke use float32, as the historical TCN did. The earlier
VideoMAE visual export may have used bf16; it is not rerun here. Inputs and
outputs, losses and gradients are checked for finite values. Reuse the export
without requiring timm or loading visual weights.

The default launcher only reads/seal-checks inputs and prints a plan. It does
not probe GPUs or create output. `--execute` requires a clean independent
worktree at the full expected commit, a new output outside code/input trees,
and one specified idle physical RTX3090 whose UUID matches the inventory.
Driver bookkeeping <=64 MiB is permitted; nonzero GPU utilization or any
compute PID blocks launch. Inventory is rechecked before each GPU stage.
Between stages the launcher waits at most 30 seconds for two consecutive
strictly idle observations. This handles the utilization sampling window
after a finished smoke worker, without relaxing the PID/utilization/memory
requirements. Any compute PID fails immediately; persistent telemetry fails
on timeout. Other tasks are not stopped. The selected UUID is mapped to child cuda:0;
this is **one independent experiment, not DDP**. The other GPU is not used.

Stages are CPU unit tests -> real frozen-feature CUDA three-step smoke ->
fresh TCN training -> strict best-checkpoint evaluation. A failure stops
subsequent stages, preserves output, and is not automatically retried.
The caller records the launcher's PID; `launcher_logs/pids.tsv` records stage
PIDs and real GPU mapping; `exit_codes.tsv` records stage exit codes. Success
requires all exits0 and complete result files, not an idle GPU or launch message.

Evaluation reports A and A+TCN overall accuracy/Macro-F1, seven-class
precision/recall/F1/support, confusion, changed/improved/worsened/wrong-to-wrong
counts, both sweep<->reperfusion directions, clinical/network, fixed duration
bins, singleton/neighbor slices, identity-free recording aggregates, per-class
F1 contributions, and time/parameter/VRAM statistics. Small-support classes
are not discarded to improve Macro-F1. This baseline does not add synthetic
neighbor perturbation, new learned gates, or multi-seed training.

## Commands

Read [SERVER_PROMPT.md](SERVER_PROMPT.md) to identify actual assets first.
Replace the following variables with verified absolute paths/identities; do
not reuse another visual export or invent protocol sidecars.

```bash
export FSN_TCN_PYTHON=/absolute/path/to/cuda/python
export FSN_TCN_INDEX=/absolute/path/to/existing/videomae/features/index.jsonl
export FSN_TCN_SOURCE=/absolute/path/to/sealed/source_protocol
export FSN_TCN_OUTPUT=/absolute/path/to/new/private/output
export FSN_TCN_COMMIT=<full-fixed-commit-of-this-independent-worktree>
export FSN_TCN_GPU_UUID=<full-physical-RTX3090-UUID>

# Local CPU checks; torch and numpy are required, pytest is not required.
"$FSN_TCN_PYTHON" -m unittest tests.test_relation_tcn_baseline tests.test_videomae_tcn -v

# Read-only plan: no GPU probing, output creation or training.
env -u CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1 \
  "$FSN_TCN_PYTHON" -m experiments.videomae_tcn.run \
  --index "$FSN_TCN_INDEX" --source-protocol-dir "$FSN_TCN_SOURCE" \
  --output "$FSN_TCN_OUTPUT" --expected-commit "$FSN_TCN_COMMIT" \
  --gpu 0 --gpu-uuid "$FSN_TCN_GPU_UUID" --python "$FSN_TCN_PYTHON"
```

Only when formal training is authorized and the plan is accepted, add
`--execute` to the same command. For nohup, choose a **new** outer log and use
shell noclobber so a second invocation cannot truncate it:

```bash
export FSN_TCN_LAUNCH_LOG=/absolute/path/to/new/private/launcher.log
(
  set -o noclobber
  nohup env -u CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1 \
    "$FSN_TCN_PYTHON" -m experiments.videomae_tcn.run \
    --index "$FSN_TCN_INDEX" --source-protocol-dir "$FSN_TCN_SOURCE" \
    --output "$FSN_TCN_OUTPUT" --expected-commit "$FSN_TCN_COMMIT" \
    --gpu 0 --gpu-uuid "$FSN_TCN_GPU_UUID" --python "$FSN_TCN_PYTHON" \
    --execute > "$FSN_TCN_LAUNCH_LOG" 2>&1 < /dev/null &
  echo "launcher PID: $!"
)
```

Do not modify the checkout during execution. Never restore or overwrite a
failed output. Inspect stage PID/log/exit and status, not just GPU occupancy.

## Output privacy and interpretation

`run_protocol.json`, logs, checkpoints and `private_val_predictions.jsonl`
remain server-private. Numeric features, videos, annotations, IDs, weights,
and private-path logs must not be committed/pushed. Only explicitly inspected
`evaluation/public_safe_summary.json` and identity-free figures/docs may be
exported to a separate non-main report branch; do not merge main automatically.

Results from this implementation are unavailable until execution completes.
Do not carry forward historical TCN F1 as a new VideoMAE+TCN result. A higher
VideoMAE score alone does not identify an architectural cause: visual
pretraining, views and optimizers differ. To attribute gains specifically to
cross-clip context, also compare a matched clip-only learned correction head
and inspect singleton/neighbor slices; that additional trained control is not
included or automatically launched by this baseline. Naming a method FSN-TCN
does not establish architectural novelty or reliable-context selection.

## Checks completed during code preparation

At code-preparation time, 47 relevant CPU tests passed (18 old/new TCN tests
plus feature/source/export regressions). The local Python had no available
CUDA; no actual server smoke or formal run was claimed by those tests.
CPU tests use tiny synthetic **numeric feature** fixtures and temporary
protocols. They are not full-data runs or real server CUDA smoke results.
Coverage includes exact A initialization, preserved singleton/padding behavior,
finite three-step gradients and updates, strict feature/input SHA and protocol
checks, split/recording separation, read-only planning, busy-GPU protection,
fresh-output protection, a short synthetic train/save/replay pipeline, and
safe aggregate metrics. The actual sealed full-data/CUDA checks are performed
by the server launcher before any authorized formal training.
