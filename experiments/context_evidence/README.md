# Center-preserving incremental context: exploratory frozen-VideoMAE study

This follows an already observed developmental diagnostic, not a preregistered
independent confirmation. The historical TCN can change a clip even when no
neighbor exists. Earlier score-adoption gates had no native training corrections
or harmful changes to learn from because both A and TCN fit train7372 perfectly.
We therefore separate center-only classification from incremental context.

Let `T(x_chain)_t` be the actual ordinary TCN and `T(x_t)_t` its same-weight
single-clip forward, including the original stem, biases, nonlinearities and
residuals. Define `delta_context = T(x_chain)_t - T(x_t)_t` and
`z_t = z_A_t + g_t * delta_context`. This is a computational counterfactual,
not a claim to uniquely identify a semantic causal effect.

The two streams share ALL TCN parameters and each block's dropout mask. The
center stream applies only the central temporal kernel tap. The final classifier
bias cancels and its seven unused parameters are explicitly frozen. The isolated-clip contextual delta is exactly zero, so these new
context variants preserve A on singleton clips; the legacy controls retain their
original singleton behavior. Candidate boundaries stay the existing sealed
recording/group/time/split boundaries. This does not detect true edits or cut
unreliable eligible neighbors. The method is offline/bidirectional.

The optional seven-parameter coefficient uses six detached confidence statistics
of the current A scores and proposed context-corrected scores. A linear layer
and bias give `g = 2*sigmoid(s)`, zero initialized to one. The range is (0,2),
not a probability of a reliable connection. No source, identity, path or true
action is an inference input. Confidence normalization uses train only.

## Six one-GPU serial controls

| Variant | Initialization / forward | Second unchanged-anchor view |
| --- | --- | --- |
| clip_only_matched | fresh 3-block kernel1 width74 head; 148229 parameters | original neighbors (ignored by kernel1) |
| legacy_continue | same completed ordinary-TCN checkpoint; full correction | original neighbors |
| legacy_aug | identical old checkpoint; full correction | replaced neighbor |
| context_delta_continue | identical old checkpoint; contextual increment only | original neighbors |
| context_delta_aug | identical old checkpoint; contextual increment only | replaced neighbor |
| context_delta_gate_aug | identical old checkpoint plus g=1 coefficient | replaced neighbor |

The clip-only head is within -0.47% of the ordinary head's 148935 allocated
parameters. It is NOT identical in functional capacity or initialization. Do
not infer the module's effect solely from that comparison. The five ordinary-
width models start from the exact same checkpoint and a fresh optimizer.
Primary comparisons are context_delta_continue vs legacy_continue,
context_delta_aug vs legacy_aug, and context_delta_gate_aug vs context_delta_aug.
The historical TCN result is a reference, not a same-budget training control.

A is frozen and never instantiated in training. Train7372 / development val823,
seed42, same sealed numeric evidence; no test, new split or A re-export. Fixed
30 actual epochs, no early stopping, batch16 normal segments, AdamW .001/wd.01,
gradclip5. Same private sampling plans and optimizer-step counts for six models.
Each epoch visits all native training clips and all eligible clinical unchanged
anchors. Native CE and anchor CE have equal weight; only the unchanged anchor
label is supervised in a replaced view. Donor visual features AND A logits are
replaced together. Same-collection donors are preferred, different recordings
required, and all training donors come from train. Sampling never uses actions.
The three augmented groups get exactly the same fixed replacements. Other
groups get corresponding unchanged views and the same view/step counts.

The perfect native train predictions remain a limitation. Synthetic replacements
need not cause any errors; if not, report that, not that a gate learned to reject
bad neighbors. This is not honest out-of-fold A evidence. Do not train on val
misclassifications or silently perturb scores to imitate held-out predictions.

## Plan then explicit execution

Create a clean independent worktree at a fixed full commit. Set actual private
absolute paths; use a fresh output outside the repository and old runs.

```bash
env -u CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1 \
  "$FSN_CONTEXT_PYTHON" -m experiments.context_evidence.run \
  --index "$FSN_CONTEXT_INDEX" --source-protocol-dir "$FSN_CONTEXT_SOURCE" \
  --base-run "$FSN_CONTEXT_COMPLETED_TCN_RUN" --output "$FSN_CONTEXT_OUTPUT" \
  --expected-commit "$FSN_CONTEXT_COMMIT" --gpu 0 \
  --gpu-uuid "$FSN_CONTEXT_GPU_UUID" --python "$FSN_CONTEXT_PYTHON"
# After plan approval, append --execute. This user authorized this round.
```

Default plan verifies all feature files, seals and complete historical TCN
artifacts, probes no GPU, and writes nothing. Explicit execution atomically
reserves output, validates one idle RTX3090 by UUID, runs CPU tests, per-clip
historical replay, six real CUDA three-update smokes, six serial trainings,
then strict best-checkpoint evaluation. Never DDP; no resume or auto retry.
Each stage records PID, GPU mapping, log and exit code. Any failure stops later
stages and preserves output. Idle driver bookkeeping <=64MiB is permitted only
with zero utilization and no compute PID, using the existing release guard.

## Metrics and audit

All results use full seven-class Macro-F1; accuracy comes from that same best-
F1 checkpoint. Report P/R/F1/support, confusion, both sweep/reperfusion errors,
paired changes vs A/historical TCN/same-budget legacy_aug, original fixes retained
and harm prevented, clinical/network/duration/singleton and identity-free record
aggregates, actual steps/time/peak memory/parameters. Fixed synthetic replacement
challenge evaluates unchanged anchors; non-challenged native clips remain in the
overall summary. Report the synthetic-anchor count separately.

The gated checkpoint also has g=1 and grouped coefficient-permutation audits.
Permutation preserves the distribution within source-kind (audit only) and
predicted-A-class strata and reports moved indices, changed numerical coefficients
and changed nonzero-context positions. Zero actual changes are NOT a valid control.
The zero gate returns A by construction and is not a separately trained baseline.
The diagnostic singleton restriction can lose real single-clip corrections;
report these tradeoffs, not only fewer mistakes. The six-sample class matters
disproportionately to Macro-F1. Record-disjoint is not patient-disjoint.

Private files include sampling plans, derived evidence, predictions, weights and
raw logs. Only privacy-reviewed aggregate `evaluation/public_safe_summary.json`
can be exported. Git may include code/tests/this document only. No raw private
paths or IDs in GitHub; no main merge.

## Literature and remaining paper comparisons

- [Not End-to-End (2021 preprint)](https://arxiv.org/abs/2107.04810): already studies
  nearly perfect upstream training predictions and disturbed/cross-validated
  evidence for downstream refinement. That observation/training idea is not new.
- [TUNeS](https://arxiv.org/abs/2307.09997): compares temporal methods with common
  frozen feature sets and studies representation/context choices separately.
- [MS-TCN++](https://arxiv.org/abs/2006.09220) and
  [ASFormer](https://arxiv.org/abs/2110.08568): appropriate mature temporal controls
  to adapt later, not automatically trained by this launcher. Their dense temporal
  segmentation metrics do not equal our seven-class clip Macro-F1.

After this bounded study: compare a mature temporal head on the identical A and
legal chains; validate on independent records; repeat prespecified head seeds
with paired record-cluster uncertainty. Frozen-A head seeds are not full-system
stability. This launcher starts no extra structures, seeds, visual training or
hyperparameter search after completion. No novelty or improvement is promised.
