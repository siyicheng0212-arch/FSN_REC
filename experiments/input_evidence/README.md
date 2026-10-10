# FSN input evidence × neighbor augmentation diagnostic

This is a bounded diagnostic, **not a new gate model**, a completed server run,
or a claim of improvement. No training starts on import or by default.

## Question and four cells

The previous native temporal heads fitted the strong backbone's original train
clips extremely quickly. The current hypothesis is that raw A logits provide
an easy shortcut; it is **not an established explanation**. Test the input
choice separately from the already registered train-only neighbor replacement.

| Cell | Seven A-logit input slots | Preserved-anchor view |
|---|---|---|
| `logits` | original raw scores | original neighbors |
| `logits_aug` | original raw scores | replaced neighbors |
| `visual` | always zero | original neighbors |
| `visual_aug` | always zero | identical replaced neighbors |

The visual vectors remain the exact mean-global, mean-local frozen vectors.
Mask **all seven slots for every clip, including donors**, inside every model
forward during training, validation and challenges. Do not zero them only at
evaluation, mutate the shared export, change normalization or inject A scores
through another argument/output residual. Original paired scores are still
validated before masking and retained privately for A-only comparisons.

All four cells have identical allocated weights, initialization, three blocks,
width64, kernel3, dilation1/2/4, padding, GELU, pointwise mix, residuals and
dropout.1. They reuse the previous **plain native classifier**, not a newly
invented TCN or a dynamic reference gate. The seven stem columns remain
allocated in the masked cells but receive zero data gradients; AdamW decay can
still change those unused weights. Report that effective input capacity differs
even though allocated parameter counts match. Source/identity/action truth is
never an input. The model is bidirectional/offline.

Outputs are native seven-class logits, **not A logits plus a correction**.
Singletons retain this same classifier behavior. A fallback is an explicitly
reported inference diagnostic only, never silently applied to the primary
results. Historical residual TCN results are references, not matched-budget
architecture evidence; historical weights are replayed, not used to initialize
the fresh native head.

## Fixed recipe and original data

Reuse the sealed full train7372/val823 source protocol and matching 8195-entry
feature index, genuine historical run, manifest SHAs, recording/group/time
chain checks and original class definition. The verifier rejects old6586/786
assets, incompatible checkpoints and cross-split/cross-record chains. Neither
raw video/frame caches nor A weights are loaded for training. A is frozen; no
new export, repartition, test read, extra seeds or follow-up training occurs.

Seed42, exactly60 epochs/no early stop, batch16 segments, AdamW .001/wd.01,
cosine to .00001, gradclip5, float32 on numeric frozen evidence. Normal clip CE
and preserved-anchor CE have equal .5 weight; donor positions are ignored.
The same independent initialization RNG, dropout seeds, sampling orders, view
counts, optimizer steps and replacement plan are shared across all cells.
The two augmented cells use byte-identical plans. Donors come only from train,
another clinical recording, same collection preferred, without action-based
donor selection. Feature vectors and raw A scores travel together.

This does **not** create honest out-of-fold evidence: the frozen visual model
has already seen train7372, even when its seven logits are masked. If fitting
remains trivial, report it and stop. Do not silently corrupt labels/scores,
retrain A, switch to cross-fitting, change budget or start a new gate.

## CPU checks, private variables and default read-only plan

In a clean independent worktree at the intended full commit:

```sh
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$FSN_INPUT_PYTHON" -m unittest \
  tests.test_input_evidence tests.test_anchor_reference tests.test_relation_tcn_baseline -q
```

Fill these variables privately with verified absolute server paths. Do not
commit a filled environment file, plan or terminal log:

- `FSN_INPUT_PYTHON`: CUDA Python with torch/numpy and repository dependencies.
- `FSN_INPUT_INDEX`: the existing sealed feature `index.jsonl`.
- `FSN_INPUT_SOURCE`: the corresponding source-protocol directory.
- `FSN_INPUT_HISTORICAL`: genuine completed ordinary-TCN run for the same A.
- `FSN_INPUT_OUTPUT`: **new** private repository-external directory, with an
  existing parent; cannot overlap features, source, historical run or code.
- `FSN_INPUT_COMMIT`: full40-character fixed code commit.
- `FSN_INPUT_GPUS`: actual physical GPU indices, for example `0,1,2`.
- `FSN_INPUT_UUIDS`: matching physical UUIDs measured after the current boot,
  in the same order. Never reuse old hardware mappings.

```sh
env -u CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1 "$FSN_INPUT_PYTHON" \
  -m experiments.input_evidence.run_suite --backbone videomae2 \
  --index "$FSN_INPUT_INDEX" --source-protocol-dir "$FSN_INPUT_SOURCE" \
  --historical-run "$FSN_INPUT_HISTORICAL" --output "$FSN_INPUT_OUTPUT" \
  --expected-commit "$FSN_INPUT_COMMIT" --gpus "$FSN_INPUT_GPUS" \
  --gpu-uuids "$FSN_INPUT_UUIDS" --python "$FSN_INPUT_PYTHON"
```

The plan hashes/validates inputs and code, checks counts, lists four policies,
parameters and GPU mapping. It does **not** probe GPUs, create outputs, replay
predictions, run smoke or train. The genuine historical823 prediction replay
is an execution-stage prerequisite, not falsely claimed by the plan.

The separate old AdaFocus adapter is supported using `--backbone adafocus`,
its own verified index/history and a different output. It is not automatically
queued. First diagnose VideoMAE; another backbone requires its own execution.

## Explicit execution after authorization

Inspect actual `nvidia-smi` inventory/compute owners first. A broken driver,
wrong UUID, unavailable CUDA or occupied card blocks the run; never reset a
GPU, stop another process, bypass the guard or infer success from idle memory.
Append **`--execute` to the exact same plan command** only when authorized.
Use `nohup env -u CUDA_VISIBLE_DEVICES ... --execute` for persistence, with a
new private external log, shell noclobber protection, and capture the actual
launcher PID. The launcher never resumes/retries or overwrites an output.

| Available cards | Wave1 | Next waves, only after peers all finish normally |
|---|---|---|
| one: `0` | logits/GPU0 | logits_aug → visual → visual_aug, all GPU0 |
| two: `0,1` | logits/GPU0, logits_aug/GPU1 | visual/GPU0, visual_aug/GPU1 |
| three: `0,1,2` | logits/GPU0, logits_aug/GPU1, visual/GPU2 | visual_aug/GPU0 |
| four: `0,1,2,3` | all four cells on GPU0/1/2/3 respectively | none |

These are independent single-GPU experiments, **not DDP**. Each child sees one
mapped UUID as cuda:0. All four real-feature CUDA batch4 three-update smokes
must pass before **any** formal training. Smoke verifies native-plain numerical
parity, consistent masking, no raw-score path/gradient in masked cells, finite
TCN stem/convolution/classifier gradients and real parameter updates, unchanged
input tensors and unchanged singleton behavior. Smoke weights are discarded.
Float32 matches this frozen-feature head, not raw-video bf16 training.

On failure, started peers finish naturally and later waves/evaluation stop.
Exit0 alone is insufficient: best/result/history, exact60 epochs and complete
step counts are checked and all selected checkpoints strictly replayed.

## Outputs and interpretation

Private output contains commit/input/config seals, shared plans, genuine replay,
per-variant history/best/result, CUDA smoke, PID/physical-GPU/UUID logs and exit
codes. Training at epochs1/10/30/60 records native and preserved-anchor accuracy,
A/model disagreements, corrections/harm, CE, confidence margin and entropy.
This distinguishes native saturation from replacement-view saturation.

Primary evaluation keeps all seven classes, same-best-F1 Accuracy, class P/R/F1/
support/confusion, two sweep↔reperfusion directions, paired fixes/harm/different
wrong, per-class F1 contributions versus all matched cells, clinical/network,
duration, singleton/neighbor and identity-free recording aggregates. Report
actual parameters, steps, time and peak memory, not equal compute by assertion.

Same-checkpoint challenges: center-only/no-neighbor, deranged neighbors, different
clinical-record replacements. Every method uses the same fixed val-only plan;
no val donor enters training. Both **native and changed metrics on the identical
eligible cohort** are reported, plus complete823 metrics, paired changes and
logit differences. Empty cohorts are explicitly unidentifiable. Source is used
for plan construction/reporting only, not as an inference answer. These are
synthetic interventions, not proof of detecting natural editing boundaries.

The report exposes four descriptive F1 contrasts and their difference-in-
differences interaction. Single seed/repeated val development is **not** a
statistical significance test. Evaluate the native train signal and actual
neighbor sensitivity alongside scores; no cell is predeclared the winner.
The six-sample fixed class remains in Macro-F1 and is flagged; do not drop it.
Record/group disjointness is not independently verified patient isolation.

Only privacy-reviewed `evaluation/public_safe_summary.json` may be exported.
Code/config/tests/docs may go to a nonmain GitHub branch. No data, source labels,
IDs, private plans/audits, clip predictions, raw/private-path logs, features,
frame caches, videos or weights are committed/uploaded. Never merge main.

## Checks at code handoff

CPU test results are reported with the delivered commit. The test suite includes
synthetic three-step updates, complete four-cell CPU checkpoint replay/evaluation,
masking invariance, donor-label safety, correct eligible-cohort comparison and
queue failure barriers. These are **not server CUDA checks or completed formal
training**. A runtime CUDA smoke and genuine historical replay remain required.

This diagnostic is not presented as a novel learning mechanism. Upstream
training predictions being easier than inference evidence is already discussed
in [Not End-to-End](https://arxiv.org/abs/2107.04810). Neither dropping logits nor
adding synthetic neighbors is promised to improve FSN recognition.
