# Server task: four-cell frozen input diagnostic

Default: read and print a private plan. **Do not train until the user explicitly
authorizes execution.** This handoff is code preparation, not an instruction to
resume old queues or claim new results.

1. Create/reuse a clean independent worktree at the delivered fixed commit of
   `codex/fsn-input-evidence-v1`. Read `experiments/input_evidence/README.md` and
   its config completely. Never edit running code/main, git pull into an active
   checkout, touch old output, backups or processed data, or start extra seeds.
2. Read actual current processes and GPU inventory. Hardware may have changed
   after reboot; measure indices and UUIDs now. Do not kill other tasks or alter
   drivers. If the same protocol is already running/completed, inspect it and
   do not duplicate/resume. Each cell is single-GPU, not DDP.
3. Locate the existing full train7372/val823 sealed source protocol and matching
   VideoMAE frozen8195 feature index. Verify actual fields/counts, per-file SHA,
   original train/val seals, A checkpoint identity and completed ordinary-TCN
   historical assets. Do not use internal6586/786, template assets, a new TCN in
   place of historical replay, new splits, new A exports, raw video or test.
4. Fill private absolute paths in `FSN_INPUT_PYTHON`, `FSN_INPUT_INDEX`,
   `FSN_INPUT_SOURCE`, `FSN_INPUT_HISTORICAL`, `FSN_INPUT_OUTPUT`,
   `FSN_INPUT_COMMIT`, `FSN_INPUT_GPUS` and `FSN_INPUT_UUIDS`. Output must be new,
   outside repositories/inputs/old runs, with an existing parent. Python must
   actually provide torch CUDA, numpy and this code's dependencies.
5. Run CPU tests:

```sh
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$FSN_INPUT_PYTHON" -m unittest \
  tests.test_input_evidence tests.test_anchor_reference tests.test_relation_tcn_baseline -q
```

6. Run the default plan, retaining it privately:

```sh
env -u CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1 "$FSN_INPUT_PYTHON" \
  -m experiments.input_evidence.run_suite --backbone videomae2 \
  --index "$FSN_INPUT_INDEX" --source-protocol-dir "$FSN_INPUT_SOURCE" \
  --historical-run "$FSN_INPUT_HISTORICAL" --output "$FSN_INPUT_OUTPUT" \
  --expected-commit "$FSN_INPUT_COMMIT" --gpus "$FSN_INPUT_GPUS" \
  --gpu-uuids "$FSN_INPUT_UUIDS" --python "$FSN_INPUT_PYTHON"
```

7. Tell the user the verified private asset paths, counts/SHA, fixed commit,
   Python, current hardware and four-cell plan. The plan has not queried GPUs,
   created output, replayed predictions, smoked CUDA or trained. Only after
   execution authorization use the **same command plus `--execute`** under
   nohup with a new non-clobbering private log and record the real launcher PID.
8. The queue first tests, strictly replays historical823 predictions, seals one
   train-only replacement/sampling plan, runs all four real-feature CUDA batch4
   three-update smokes, then trains and evaluates. Three GPUs: logits/GPU0,
   logits_aug/GPU1, visual/GPU2; after all three exit0/complete, visual_aug/GPU0.
   Two GPUs: [logits/GPU0, logits_aug/GPU1], then [visual/GPU0, visual_aug/GPU1].
   One GPU serial; four GPUs all four independent. No silent GPU3 use.
9. Keep seed42,60 epochs and every registered setting unchanged. Only two
   factors differ: all seven A-logit slots raw versus always zero, and original
   versus paired replaced neighbors. A stays frozen; no gates/new losses/OOF.
   Raw donor features and scores stay paired. Donor labels are never supervised;
   preserved current evidence/truth cannot change. Both augmented cells share
   exactly the same private plan. First full epoch: inspect history, gradients,
   native saturation and replaced-anchor saturation; do not stop/reconfigure
   simply because the observed hypothesis is unsupported.
10. Missing asset, mismatch, bad driver/UUID, occupied GPU, OOM/NaN, missing
    gradient/update or wrong replay blocks execution/later stages. Preserve
    evidence, let already running peers finish naturally, do not retry, restore,
    lower checks or modify parameters. No completion claim from GPU idle.
11. When all four have exit0 and valid best/history/result plus evaluation,
    report A/historical-reference/four-cell F1 and same-checkpoint Accuracy,
    all-class metrics, two directional errors, correction retention/harm,
    singleton protection diagnostic, source/duration/recording aggregates,
    train signal, actual steps/time/parameters/memory and matched-cell F1
    contrasts. Challenges must include native and changed scores for the same
    eligible anchors, not a comparison of different cohorts.
12. Label val823 repeatedly used development, not independent test. Masking
    scores does not untrain A or make features out-of-fold. Keep the six-sample
    class; report instability, not a post-hoc exclusion. Synthetic donor tests
    are not natural-boundary labels. No automatic further backbone, model,
    source-rule oracle, tuning, seeds or cross-fitting after this bounded run.

Only inspected aggregate metrics may be published on nonmain. Never publish
features, IDs, source labels, raw manifests, private audits/plans/paths/logs,
per-clip predictions, videos, caches or weights. Deliver the report and stop.
