# VideoMAE V2: train7307 / Menzhen-dev888

This is a new protocol, not a continuation of an old model or the 7372/823
experiments. R+D is disabled. All former non-Menzhen train and dev clips are
merged into training; Menzhen is excluded from optimizer updates.

The user explicitly changed the evaluation rule: Menzhen888 is used for
checkpoint selection and model comparison. Consequently it is **development
validation**, not independent test. An institution holdout does not establish
patient/operator independence. Historical runs also used Menzhen records.

## Data and preprocessing

- Read only the SHA-fixed former `train.jsonl` and `val.jsonl`; never open an
  additional old test manifest. Expected measured counts are 7372+823 before
  deriving train7307 / Menzhen-dev888. Seven classes must remain present.
- Preserve original clip/event IDs, labels and full-event membership. Reparse
  original annotations, require their sealed SHA, check video size/duration,
  count and clip/record/group/path overlap, then hash raw video bytes to reject
  exact video copies across the new partition. No new random split.
  Positive historical file sizes must match. A historical zero-size/no-moov
  sentinel is explicitly marked missing, requires a successful fresh duration
  and geometry probe, and is sealed using the current bytes. It is not claimed
  to match an unavailable historical checksum. Historical hashes, when present,
  must match. No duration fallback to stale metadata is allowed.
- Bare numeric timestamps have no inferred unit. TSV clocks independently
  verify numeric columns. Other units require private, SHA-keyed evidence.
  The owner confirmed the decimal-second/integer-millisecond export convention
  on 2026-10-10. `unit_evidence` applies this explicit convention, not an integer
  or maximum-value threshold; paired EAFs must replay when available. Ambiguous
  files, changed counts, changed labels or invalid boundaries stop the run.
  A read-only audit found exactly four terminal overruns (maximum0.437 seconds).
  The owner explicitly authorized clipping these four ends on2026-10-10.
  `--clip-terminal-overrun` requires exactly four corrections, decodes each
  final frame, and caps at the smaller container/decoded video end; original
  annotation boundaries and per-clip correction provenance remain private.
  It rejects large overruns, empty remaining intervals or a changed count;
  without the flag any overrun remains a blocker. No source file is modified.
- Decode original RGB video, sample 16 frames by nearest actual PTS centres
  inside each interval, resize the short side to 256 with aspect ratio retained,
  centre-crop224, and normalize with ImageNet mean/std. Store the actual frame
  indices, timestamps and repeated-frame counts privately. Old square caches
  are not reused or upscaled. Lack of an actual frame in an interval is an error.
- The spatial transform is numerically checked against the pinned upstream
  OpenGVLab validation transform. FSN uniform temporal sampling, single view
  and flip-only training are declared differences: this is not a reproduction
  of upstream K710 random-augmentation or multiview benchmark results.

## Fixed training

Official VideoMAEv2 ViT-B/16 K710 distilled checkpoint, SHA
`8141a6955e0700d11bf15928fe6d61e5cfe482606fed8cfdddb1b922c0fd88ec`,
source commit `29eab1e8a588d1b3ec0cdec7b03a86cca491b74b`. Load all160
non-head state tensors; replace only the710-class head with seven classes.

Seed42; batch4 / accumulation16 (effective64); workers4; weighted CE with
sqrt-inverse training-only class counts; clip-grad20; flip probability0.5.
AdamW lr0.000175, weight decay0.05, layer decay0.75. Five head-only warmup
epochs at lr0.001, then100 full-model finetune epochs with cosine decay.
No early stopping: all registered variants can later have the same budget.
7307 examples are visited once each epoch:115 updates/epoch,12075 total.

Each epoch evaluates Menzhen888. `best.pt` is selected by strict improvement
in **finetune development Macro-F1**; Accuracy is reported from that same
checkpoint. `last.pt` preserves progress; `final.pt` preserves the unselected
last-epoch snapshot. No automatic resume/retry. OOM/NaN/missing gradients
preserve files and stop instead of silently changing batch or protocol.

## Commands

All variables below must be replaced with real absolute paths. Unit maps,
manifests, cache, logs and model weights remain outside Git. Use a clean,
independent checkout at `$FSN_COMMIT`. The selected Python must have compatible
CUDA torch/torchvision, timm0.4.12, NumPy1.26.4, OpenCV4.11.0.86, decord0.6.0
and pytest. Do not replace an existing environment during a run.

First create private evidence after owner confirmation:

```sh
"$FSN_PYTHON" -m experiments.hospital_holdout.unit_evidence \
  --source-manifest-dir "$FSN_OLD_MANIFESTS" --records "$FSN_RECORDS" \
  --data-root "$FSN_DATA_ROOT" --confirmed-decimal-seconds-integer-ms \
  --output "$FSN_PRIVATE_UNIT_MAP"
# Review the aggregate plan; add --write to create the new private map once.
```

Read-only run plan (no GPU probe, no output creation):

```sh
env -u CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1 "$FSN_PYTHON" \
  -m experiments.hospital_holdout.run \
  --source-manifest-dir "$FSN_OLD_MANIFESTS" \
  --records "$FSN_RECORDS" --events "$FSN_EVENTS" \
  --data-root "$FSN_DATA_ROOT" --unit-map "$FSN_PRIVATE_UNIT_MAP" \
  --clip-terminal-overrun \
  --official-source "$FSN_OFFICIAL_SOURCE" --pretrained "$FSN_PRETRAINED" \
  --python "$FSN_PYTHON" --expected-commit "$FSN_COMMIT" \
  --output "$FSN_NEW_PRIVATE_OUTPUT"
```

After data copying has finished and the plan verifies, use **the same command**
with `--execute` under `nohup` and a new private external log. It reserves a new
output, records launcher/stage PIDs, physical GPU UUID, logs and exit codes;
then runs CPU tests, data sealing, new train and dev caches, actual batch4 CUDA
bf16 three-step updates, a fresh official initialization for training, and
best-checkpoint metric replay. Physical GPU0 maps by UUID to internal cuda:0;
this is not DDP. The launcher rechecks occupancy after CPU caching. Driver-only
idle memory up to64MiB is allowed only if compute-apps are empty and utilization
is at most5%; it never stops another job.

```sh
PYTHONDONTWRITEBYTECODE=1 FSN_OFFICIAL_VIDEO_SOURCE="$FSN_OFFICIAL_SOURCE" \
  "$FSN_PYTHON" -m pytest -q tests/test_hospital_holdout.py \
  tests/test_build_dataset_duplicates.py tests/test_videomae2_baseline.py \
  tests/test_metrics.py -p no:cacheprovider
```

Inspect `status.json`, `launcher_logs/pids.tsv`, `exit_codes.tsv`,
`smoke/cuda_smoke.json`, and `train/history.json`. A GPU becoming idle is not
completion. Only exit0 + complete suite/training/evaluation results confirm it.
The final aggregate has seven-class P/R/F1/support, confusion matrix, duration
and source slices, and both sweep/reperfusion error directions. Private IDs,
per-clip outputs, original annotations/videos, cache, logs and weights must
not be uploaded.

## Further A / T comparisons (not launched by this command)

Reinitialize each visual baseline from its official pretraining on this new
partition. Do not reuse old FSN backbones or frozen features that trained on
Menzhen. First compare unmodified VideoMAE and AdaFocus under fully reported
inputs/pretraining/budgets; then isolate one visually motivated modification.

For a proposed A′ and T′, report A-only and A′-only plus the factorial comparison
A+TCN, A′+TCN, A+T′, A′+T′. Under each same frozen A, compare ordinary TCN,
MS-TCN++ and ASFormer adaptations, and protect isolated clips with an explicit
baseline. Check original/no/shuffled/cross-record neighbours. Separate the
clip-internal visual time scale from the across-clip context time scale.

The visual hypothesis is better discrimination of brief/local operation
evidence; the temporal hypothesis is useful multi-span contextual correction
without sacrificing the current-clip prediction. Neither a multiscale branch
nor a gate is automatically novel. Previously ineffective gates are not proof
that a larger gate will work. Pre-register variants before scoring Menzhen,
report negative results and actual compute, and avoid treating development
selection as an independent generalization result.

### Foundation-backbone screening plan (not implemented by this runner)

Keep the existing MViT-V2-S, Swin3D-T and R(2+1)D-18 supervised K400
alternatives, and prioritize two additional frozen representation probes:

| Candidate | Representation route | Initial comparison |
| --- | --- | --- |
| VideoMAE V2 ViT-B/16 | Masked-video lineage plus K710 classification/distillation checkpoint | Reinitialized FSN fine-tuning; also an official-pretrained frozen probe if comparing probes |
| MViT-V2-S / Swin3D-T / R(2+1)D-18 | Supervised K400 video models | Separate frozen-probe and fine-tuning tables |
| V-JEPA 2 ViT-L/16 | Self-supervised latent prediction | Frozen encoder and common seven-class probe |
| InternVideo-Next Base | Two-stage Encoder-Predictor-Decoder pretraining without video-text supervision | Frozen encoder and common seven-class probe |
| VideoPrism | General-purpose encoder with image/video/text pretraining | Optional later work; separate JAX/Flax environment |

All candidates must use the newly sealed train7307 / Menzhen-dev888 manifests;
none may reuse old FSN weights or frozen evidence trained on Menzhen. Existing
foundation adapters on the other branch require migration: their 36-frame
224-square cache cannot recover original aspect ratio or a larger field of
view by upscaling. Decode original frames for each pinned official processor,
save normalization/resize/crop/frame/view settings and input/weight/source
SHA, and test processor and encoder outputs against that exact upstream code.
Do not change the running VideoMAE checkout or its registered budget.

The official V-JEPA 2 SSv2 scores are **attentive-probe** results; the headline
77.3% belongs to the larger ViT-g/384 configuration, not ViT-L with mean pooling
and a linear head. A common mean-pool/linear-head FSN screening experiment
must be named as such, not presented as reproduction of the official protocol.
Require `eval()` on every frozen encoder (including after `probe.train()`),
zero encoder gradients/updates, strict checkpoint loading and finite real-batch
feature checks. Small-sample screening is not proof that full fine-tuning will
work or that its memory budget is supported. Single-GPU jobs run sequentially.

For InternVideo-Next, the officially linked Base weights are hosted under
`revliter`, not an OpenGVLab account. Report the actual owner, pin the model
revision and inspect the custom model code before enabling remote-code loading.
Follow its processor; do not substitute V-JEPA or VideoMAE normalization.
Pretraining data, parameter counts and native input views differ, so do not
attribute all score differences to architecture or claim equal pretraining.

These comparisons establish the visual baseline; they are not the paper's
method contribution. Separately isolate A′ and T′ with A-only/A′-only and the
four combinations above, then test context ablations under the same frozen A.
R+D and Markov/Viterbi decoding are excluded from this new experiment plan.

Primary sources checked on2026-10-10:

- [V-JEPA 2 official code and model/probe table](https://github.com/facebookresearch/vjepa2)
- [V-JEPA 2 ViT-L SSv2 inference configuration](https://github.com/facebookresearch/vjepa2/blob/main/configs/inference/vitl/ssv2.yaml)
- [InternVideo-Next paper](https://arxiv.org/abs/2512.01342)
- [Officially linked InternVideo-Next Base model](https://huggingface.co/revliter/internvideo_next_base_p14_res224_f16)
- [VideoPrism official JAX/Flax implementation](https://github.com/google-deepmind/videoprism)
