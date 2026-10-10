# Hospital-protocol T/T′ experiments

This is code preparation and a preregistered experiment, not a positive result.
Train7307 / Menzhen-development888: the owner uses Menzhen to select checkpoints
and models. It is **not an independent test**. Group/recording separation is
not verified patient separation. R+D is disabled. Old train7372 includes799 of
the new888, so never reuse its FSN-trained visual/temporal weights or features
as hospital-held-out evidence.

## Research problem and scope

Teaching retrieval and operation review motivate recognition. This experiment
does not prove teaching is exclusively manual or robots require video-language
models. Keep applications separate from the measured seven-class recognition.

- A′ hypothesis: improve subtle/brief within-clip evidence. No A′ is implemented
  here. Sixteen uniformly sampled frames cannot guarantee recovery of a brief
  important event. Global/local fusion already occurs in AdaFocus; renaming it
  is not a novelty claim.
- T′ hypothesis: model helpful across-clip changes and context spans as residual
  corrections while protecting singleton clips.
- Clinical/network source is not actual continuity/discontinuity ground truth.
  Source names/identities never enter the model. Structural adjacency is not
  an independently verified editing boundary.

The former residual TCN already reads visual features **and raw seven logits**,
not just probabilities. New A is VideoMAE V2 freshly trained from official
pretraining under the corrected hospital protocol. Complete its105-epoch
pipeline and selected-checkpoint replay before frozen export or T training.
Do not change/restart its runtime checkout or output. Only the token-capture
helper is reused; no R/D training or decoder is executed.

## Registered models

| Variant | Change | Trainable parameters | Maximum field |
| --- | --- | ---: | ---: |
| plain_tcn | Existing residual TCN, singleton unprotected |148935|15 clips|
| clip_only | Pointwise correction, no neighbors, width74 |148229|1 clip|
| protected_tcn | Ordinary TCN, singleton strictly A |148935|15 clips|
| multispan_tcn | Same kernel at d and2d, averaged |148935|29 clips|
| relative_tcn | Valid neighbor contribution GELU(h_j−h_t) |148935|15 clips|
| multispan_relative_tcn | Factorial combination |148935|29 clips|

A-only is an untrained reference. Also evaluate the **same plain checkpoint**
with isolated outputs reset to A; that inference protection is distinct from
the independently trained protected model. The clip-only control matches
parameters within0.5%, not exactly. Equal parameters do not mean equal compute.

Input: temporal-mean spatial-mean768 + temporal-mean spatial-max768 from the
same late VideoMAE layer + raw logits7. These are not two independent visual
backbones or annotated operation boxes; no extra input normalization.
Keep Linear/LayerNorm/GELU stem, three residual blocks with dilation1/2/4,
GELU/dropout0.1/pointwise mix, and zero-initialized seven-class correction.
Initial outputs exactly equal A; A is immutable and never in the optimizer.
Symmetric convolution is offline, not causal/online.

Relative models preserve the center and bias; absent neighbors contribute zero.
GELU on the difference is deliberate: a purely linear h_j−h_t is largely an
ordinary-kernel reparameterization in the interior, not a new representation.
This operation does not guarantee rejection of bad neighbors or strict boundary
isolation. Singleton protection does not solve errors in connected chains.

Small/large dilated spans are established prior art:
[MS-TCN++](https://arxiv.org/abs/2006.09220).
Local/hierarchical context is established in
[ASFormer](https://arxiv.org/abs/2110.08568).
These prototypes are not official reproductions. Neither naming FSN-TCN nor
adding multiple scales proves innovation.

Spans count candidate clips, **not seconds or clip-internal action duration**.
Menzhen maximum chain length is10:29 is a theoretical field, not29 real neighbors
for every clip. Report short-chain/boundary and duration-slice results.

## Data and fixed budget

CPU audit: train7307 /4245 edges /5955 neighbor clips /1352 isolated,
Menzhen888 /451 edges /656 neighbor clips /232 isolated.
Train3062 segments, Menzhen437. Preserve every clip once.

Same split/recording/group/video/time base; sort verified timestamps; unique
starts; no overlaps including nested coverage; gap≤5sec. Each exclusion cuts
the chain, without bridging or deleting clips. No source/action gate.

config.json fixes seed42,100epochs,16segments/batch,AdamW lr0.001/wd0.01,
unweighted valid-clip CE,clip-grad5,no early stopping or training augmentation.
Six models share one sampler:7307 clip visits and192 updates per epoch,
19200 updates per model. Dropout seeds match common-shape variants; the width74
control cannot have identical masks. This never changes A's training budget.
Select strictly by Menzhen-development Macro-F1; Acc is from that same checkpoint.
No extra seeds or automatic hyperparameter search.

## Commands

Variables are actual absolute paths. Use clean fixed independent code and
fresh private output outside all inputs/code. No data/IDs/paths go into Git.

```sh
PYTHONDONTWRITEBYTECODE=1 "$FSN_PYTHON" -m pytest -q \
  tests/test_hospital_temporal.py tests/test_hospital_temporal_models.py \
  tests/test_hospital_temporal_pipeline.py tests/test_metrics.py -p no:cacheprovider

# CPU-only chain audit; no weight/cache loading or GPU probing.
PYTHONDONTWRITEBYTECODE=1 "$FSN_PYTHON" \
  -m experiments.hospital_temporal.prepare --protocol-dir "$FSN_NEW_HOSPITAL_PROTOCOL"

# Default read-only: no GPU probe, output creation, or training.
env -u CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1 "$FSN_PYTHON" \
  -m experiments.hospital_temporal.run \
  --a-run "$FSN_COMPLETED_NEW_A_RUN" \
  --official-source "$FSN_OFFICIAL_VIDEO_SOURCE" --pretrained "$FSN_PRETRAINED" \
  --python "$FSN_PYTHON" --output "$FSN_NEW_PRIVATE_TEMPORAL_OUTPUT" \
  --expected-commit "$FSN_TEMPORAL_COMMIT" --gpu-uuid "$FSN_PHYSICAL_GPU_UUID"
```

Inspect paths/SHA/budget. Append --execute to the same command only once A
completes. For an explicitly chosen CPU dependency queue append both
--wait-for-A --execute. Waiting reserves no GPU while A prepares/trains.
All A stage exits, completion, checkpoint seal and replay must pass first.
Failed/missing live A or a96-hour deadline stops the queue, preserving A.
This optional capability is **not a claim a queue has been started**.

Execution requires one idle physical RTX3090 with matching UUID and no compute
processes. Driver-only≤64MiB may be idle; do not kill/take over other jobs.
Independent CUDA_VISIBLE_DEVICES UUID mapping gives each child cuda:0; not DDP.
Stages: CPU tests → exact newly exported A → every model's real-feature CUDA
three-step smoke → six models sequentially → unified replay/paired evaluation.

New output/exclusive markers/logs, actual PID/UUID and per-stage exits.
No overwrite/resume/retry/protocol weakening. Any failed stage stops later
models. Only full exit0, valid result/history/best and replay mean completion.

## Interpretation and deliverables

Report seven-class P/R/F1/support, confusion matrices, F1/Acc, corrected/harmed/
wrong-to-wrong versus A/plain/protected TCN, each sweep↔reperfusion direction,
isolated/neighbor/source/duration slices, recording-average accuracy (not
patients), each class's F1/7 contribution, best epoch, steps/time/peak memory
and parameters. Menzhen fixed class has17 samples: report that denominator.
Menzhen-only validation cannot measure held-out network-video generalization.

Diagnostics: original context, same-checkpoint center-only, fixed shuffle,
fixed cross-record neighbors. Anchor feature/logits/label stay unchanged.
Donor visual features and A logits move together. Challenges use only declared
Menzhen-development clips; no extra test and no training against challenge
labels. Report actually changed positions: shuffling a single neighbor in a
two-clip chain changes nothing. Measure retained plain-TCN corrections and
recovered harms; beating A alone does not prove beating simple protection.
Synthetic challenges are not natural editing-boundary assessment.

Once A′ exists, use A+T, A′+T, A+T′, A′+T′ plus A-only/A′-only under the same
protocol. Mature temporal benchmarks belong in a separate properly adapted
comparison, not mislabeled versions of these prototypes. Positive outcomes
are hypotheses, not prewritten conclusions.

private_features.pt, sampler plans, weights, IDs, raw logs/input paths remain
private. Only code/config/tests/docs and aggregate public_safe_summary.json
can be published to non-main. No main merge.
