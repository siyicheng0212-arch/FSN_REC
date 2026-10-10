# FSN frozen-backbone input and temporal benchmark

This package prepares bounded **comparison and diagnostic** experiments. It is
not a new FSN model, proof of improvement, a completed CUDA check or a completed
server experiment. No training starts on import or by default. Its scientific
priorities are in [CLAIM_MATRIX.md](CLAIM_MATRIX.md), not a leaderboard search.

## Why compare feature extractors separately?

Use one fixed frozen A to compare temporal methods, then repeat the same design
on a second A. A better absolute score after changing A is not evidence that a
temporal module improved. [TUNeS](https://arxiv.org/abs/2307.09997) studies both
feature-extractor context and temporal design and reports the importance of
contextualized features. [MS-TCN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.pdf)
compares different input features. These support separating backbone effects
from temporal-head effects; they do not establish novelty for applying TCN to
FSN.

The explicitly supported existing exports are:

| Frozen backbone | Existing visual vector | Seven A score slots | New visual training |
|---|---|---|---|
| VideoMAE V2 | two 768-dimensional poolings of the same ViT tokens | 7 raw logits | none |
| AdaFocus A | mean global1280 + mean local2048 | 7 raw logits | none |

Do not call VideoMAE's two poolings independent global/local backbones. Each
backbone has its own genuinely verified index, A seal and historical TCN run.
They share the original full train7372/val823 protocol and label space. Their
pretraining, views and visual training budgets differ. This is head-level
replication under two fixed A models, not whole-system multi-seed stability.
Other extractors can be a future explicit experiment; this launcher does not
automatically train/export VideoMAE, UniFormer, I3D, ResNet or another A.

## Finite matrix: twelve cells per requested backbone

| Task | Head | A score input | Anchor view | Loss |
|---|---|---|---|---|
| `logits` | ordinary native TCN | raw | original | final CE |
| `logits_aug` | same | raw | replaced neighbors | final CE |
| `visual` | same | all7 zero | original | final CE |
| `visual_aug` | same | all7 zero | identical replacements | final CE |
| `clip_logits` | same TCN with all noncenter taps disabled | raw | original | final CE |
| `clip_visual` | same center-only control | all7 zero | original | final CE |
| `mstcnpp_logits` | MS-TCN++ | raw | original | 4-stage CE + smoothing |
| `mstcnpp_visual` | MS-TCN++ | all7 zero | original | same |
| `asformer_logits` | ASFormer | raw | original | 4-stage CE + smoothing |
| `asformer_visual` | ASFormer | all7 zero | original | same |
| `mstcnpp_logits_final_ce` | same MS-TCN++ | raw | original | final CE only |
| `asformer_logits_final_ce` | same ASFormer | raw | original | final CE only |

One explicitly requested backbone registers12 cells; both register24. They do
not imply24 fresh training runs: supported completed identical native cells may
be explicitly verified and reused. No seeds or architectures are added based
on observed results. Do not rerun completed anchor-reference gates just to
inflate this table; their existing matched results remain relevant, including
negative results. There is no new gate in this package.

The native TCN reuses `input_evidence`: width64, 3 blocks, k3,
dilation1/2/4, symmetric padding, GELU, pointwise mix, residuals and dropout.1.
It predicts native class logits, **not** A plus a correction. Historical
residual TCN is replayed as a reference, not substituted for this matched head
or used to initialize it. Center-only is a separately trained per-clip
classifier; it is not an A fallback. Allocated unused neighbor weights remain
and are reported. An inference-only singleton-A protection diagnostic is also
reported without silently changing primary predictions.

MS-TCN++ uses11 prediction-generation layers,10 refinement layers and3
refinements, width64. ASFormer uses10 layers,3 decoders and width64. Their pinned
author-derived code, upstream provenance and MIT licenses are retained under
`experiments/temporal_heads/vendor/`. The first is a dual-dilation multi-stage
TCN ([paper](https://arxiv.org/abs/2006.09220)); the second uses hierarchical
local attention ([paper](https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/0183.pdf)).
Rows of different exact lengths are forwarded separately so padding does not
alter attention/instance normalization. Existing short-sequence/singleton
adaptations are retained and tested against the pinned author implementation.
These are **FSN clip-chain adaptations**, not reproductions of the papers'
dense frame-segmentation protocols or reported segmentation F1/Edit scores.

## What is matched, and what is not

Seed42; exactly60 epochs/no early stopping; batch16 candidate segments; fresh
AdamW lr.001/wd.01; cosine once per epoch to .00001; gradclip5; float32 numerical
features. Shared independent model initialization and private sampling plan;
same normal/anchor view counts and optimizer-step budget. Same-family input and
loss variants start with exactly identical weights. Different architectures
have different parameters, dropout and computation: record actual resources,
not equal-compute claims.

Every task uses .5 normal loss + .5 preserved-anchor loss. All tasks use the
same clinical anchor pool. The two augmented native cells use the exact same
train-only replacement plan: another clinical recording, same collection
preferred, no label-based donor choice. Donor feature and A scores stay paired;
only the unchanged anchor is supervised. Donor labels are ignored. Unaugmented
tasks get the same number of anchor views with the original neighbors.

Mature-head canonical loss sums CE over four stages and adds .15 times capped
log-probability difference smoothing over valid **normal-chain** adjacent
positions. No smoothing or donor classification is applied to anchor views.
The final-CE control removes intermediate CE **and** smoothing together; it
does not isolate the smoothing term alone. Common anchor supervision and
budget are FSN adaptations rather than author-paper training recipes.

Zero-score models mask all7 slots inside every forward, including donor inputs
and validation. They have no direct A residual. Exact raw feature/score pairing
and finiteness are checked before masking. No in-place evidence modification
is permitted. Removing scores does not make the remaining frozen visual train
features out-of-fold. At epochs1/10/30/60 record native and anchor errors,
confidence/disagreements/corrections; if training is again trivial, report it
without changing the protocol. No val errors are used as training labels.

## Private assets and safe reuse

Create an **external private** JSON list (do not commit it). Example placeholders,
not real server paths:

```json
[
  {
    "backbone": "videomae2",
    "index": "/PRIVATE/videomae_features/index.jsonl",
    "source_protocol_dir": "/PRIVATE/sealed_full_source_protocol",
    "historical_run": "/PRIVATE/completed_ordinary_videomae_tcn",
    "reuse": {
      "logits": {"run": "/PRIVATE/completed_anchor_reference", "variant": "plain"},
      "logits_aug": {"run": "/PRIVATE/completed_anchor_reference", "variant": "plain_aug"}
    }
  },
  {
    "backbone": "adafocus",
    "index": "/PRIVATE/adafocus_features/index.jsonl",
    "source_protocol_dir": "/PRIVATE/sealed_full_source_protocol",
    "historical_run": "/PRIVATE/genuine_old_residual_tcn",
    "reuse": {}
  }
]
```

Remove a backbone row to request only one; do not pretend missing features
exist. Do not fill a template instead of verified old assets. The genuine old
AdaFocus adapter and VideoMAE verifier enforce source/index/manifests/A seals,
7372/823 coverage, legal time/group/record chains, history and strict weights.
The old823 individual predictions must replay before CUDA/training begins.
No test, old6586/786 split, new export, raw video or frame cache is used.

Reuse is allowed ONLY for identical completed native4 cells from
`fsn-input-evidence-native-v1`, or raw `plain`/`plain_aug` cells from
`fsn-anchor-reference-native-v1`. The plan verifies complete suite exits,
exact60-epoch budget, common recipe, code/input/A SHAs, strict weights and
selected best-F1 checkpoint/Accuracy. Execution regenerates the same plans and
requires sampling/replacement/challenge hashes and actual replay metrics to
match. Different output semantics, early-stopped budgets or an equal rounded
F1 are not proof of equivalence. A failed reuse check stops the entire plan;
never silently retrain or reinterpret a requested reused cell. Reused weights
and histories remain in their old locations, read-only, and are not copied.

## Commands: default read-only, then explicit execute

See [SERVER_PROMPT.md](SERVER_PROMPT.md) for asset discovery and boot checks.
Set privately: `FSN_BENCH_PYTHON`, `FSN_BENCH_ASSETS`, `FSN_BENCH_OUTPUT`,
`FSN_BENCH_COMMIT`, `FSN_BENCH_GPUS`, `FSN_BENCH_UUIDS`. Require a clean
independent worktree at the exact full40-character published commit, an actual
CUDA Python with torch/numpy and a brand-new repository-external output whose
parent exists. Obtain actual indices/UUIDs after this boot; never paste an old
mapping. The same numeric frozen tensors are reused, not re-exported.

```sh
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$FSN_BENCH_PYTHON" -m unittest \
  tests.test_input_benchmark tests.test_input_evidence \
  tests.test_temporal_heads tests.test_anchor_reference -q

env -u CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1 "$FSN_BENCH_PYTHON" \
  -m experiments.input_benchmark.run_suite --assets "$FSN_BENCH_ASSETS" \
  --output "$FSN_BENCH_OUTPUT" --expected-commit "$FSN_BENCH_COMMIT" \
  --gpus "$FSN_BENCH_GPUS" --gpu-uuids "$FSN_BENCH_UUIDS" \
  --python "$FSN_BENCH_PYTHON"
```

The plan reads/hashes/verifies assets and prints parameters/finite task/wave
mapping. It does not probe GPUs, create output, replay or train. Do not upload
the printed plan. Only after explicit run authorization append `--execute` to
the identical command. For persistence, in a shell with noclobber enabled:

```sh
set -C
nohup env -u CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$FSN_BENCH_PYTHON" -m experiments.input_benchmark.run_suite \
  --assets "$FSN_BENCH_ASSETS" --output "$FSN_BENCH_OUTPUT" \
  --expected-commit "$FSN_BENCH_COMMIT" --gpus "$FSN_BENCH_GPUS" \
  --gpu-uuids "$FSN_BENCH_UUIDS" --python "$FSN_BENCH_PYTHON" --execute \
  > "$FSN_BENCH_OUTPUT.launcher.log" 2>&1 < /dev/null &
FSN_BENCH_LAUNCH_PID=$!
printf '%s\n' "$FSN_BENCH_LAUNCH_PID"
```

The queue supports1/2/3/4 free cards. Within the fixed task ordering above,
backbones are interleaved; reused cells are removed from the training queue.
Each wave fills selected GPUs in order; the next starts only after all peers
exit normally and sealed complete artifacts validate. With both backbones and
three cards, initial smoke wave is VideoMAE/logits on card0, AdaFocus/logits on
card1, VideoMAE/logits_aug on card2. Training mapping depends on explicit reuse;
read the actual plan. Each worker uses a single UUID as cuda:0, never DDP.

Execution checks actual RTX3090 inventory, UUID/driver and compute owners;
never kills, resets or steals a card. All registered12/24 real batch4 CUDA
three-update smokes pass **before any formal training**. They check singleton/
padding isolation, score masking/sensitivity, finite gradients/updates and
unchanged evidence. Smoke weights are discarded. Any failure preserves the
scene, waits already-launched peers naturally and stops later waves/evaluation;
there is no auto-retry/resume/threshold/batch change.

## Evaluation and deliverables

Same best-F1 checkpoint supplies Accuracy. Full7 P/R/F1/support/confusion,
paired fixes/harms/different-wrong vs A and native TCN, per-class F1 change/7,
sweep-to-reperfusion and reverse, singleton/neighbor, clinical/network,
duration and identity-free recording aggregates. Parameters/unused native
weights, steps, elapsed time and peak allocated/reserved memory are recorded.

All checkpoints receive the same original/no-neighbor/shuffle/cross-clinical
interventions, with both native and changed metrics on exactly the eligible
cohort plus overall823. No-neighbor uses the actual model on isolated clips;
it is not forced back to A. The source field constructs plans/strata only, not
an inference answer or model input. Structural adjacency and source policy are
not manually verified boundaries; synthetic challenges do not establish
natural editing-boundary detection.

Paired recording-cluster bootstrap1000 replicates reports conditional95%
development intervals vs the matched native input control. Keep all7 classes
when rare classes disappear in resamples. This is not patient independence,
an independent test or confirmatory significance after repeated val use.
The six-sample class is retained; expose its contribution rather than dropping
it. Single seed and many exploratory cells cannot support a paper conclusion
without further honest validation. No automatic independent-test substitution.

Private output: plans/evidence/cache, audits, checkpoints/history/results,
individual predictions, stage logs, pids.tsv (actual physical index/UUID),
exit_codes.tsv, status/suite results. Exit0 or an idle GPU alone is insufficient.
Public-safe outputs: `public_safe_summary.json`, aggregate/per-class CSVs,
four SVG F1 panels per backbone and `REPORT.md`. Privacy-check these before
export. Only code/config/tests/docs and reviewed identity-free aggregates may
go to a nonmain branch. No raw logs/private paths/IDs, plans, features, labels,
video, cache or weights are uploaded. No main merge, further training or
automatic monitoring is enabled by this package.

CPU tests exercise synthetic real model forwards/three-step updates and
complete12-cell checkpoint/evaluation/report fixtures. They are **not** actual
server feature/CUDA checks or new experiment results. Runtime ETA must be
estimated from real early epoch times and requested/reused jobs, not promised
as an overnight guarantee.

Code-handoff check:145 CPU tests passed across `test_input_benchmark` (41 tests),
`test_input_evidence`, `test_anchor_reference`, `test_temporal_heads` and
`test_relation_tcn_baseline`. Local environment: Python3.12, torch2.2.2,
numpy1.26.4, no CUDA. Launcher `--help` and staged whitespace checks passed.
Real server assets, current GPUs, historical runtime replay and CUDA smoke
remain to be checked when an execution is authorized. No formal training was
started for this code delivery.
