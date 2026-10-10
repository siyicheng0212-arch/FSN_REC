# Server task: bounded frozen-A temporal comparison, one boot

Read README.md and CLAIM_MATRIX.md completely. The default task is asset
discovery plus a **read-only plan**. Do not start training until the user opens
the server and explicitly asks to run this fixed commit. Do not change code
while running or add tasks because a GPU is idle.

1. Create/reuse a clean independent worktree of `codex/fsn-input-benchmark-v1`
   at the exact full published commit supplied with delivery. Preserve main,
   old worktrees, untracked backups, processed data and all prior outputs.
   Never git pull or patch a running checkout.
2. Inspect actual GPU/driver and processes after this boot, not historical
   UUIDs. Only free RTX3090s1–4 may be used. User's current server may have
   three cards; choose only those actually present. No DDP or stopping/resetting
   another task. A torch/numpy-capable CUDA Python is required. Use the existing
   authorized SSH method if remote access is needed; do not request secrets.
3. Identify the sealed **full train7372/val823** source protocol and existing
   matching8195-entry indexes for VideoMAE and AdaFocus, and genuine completed
   ordinary/residual TCN histories/checkpoints/823 individual predictions.
   Inspect fields/counts/SHA, not expected numbers alone. No test, inner6586/786,
   redivision, new A training/export, raw video or frame-cache loading. A
   template or another TCN cannot replace missing genuine assets.
4. Inspect completed earlier input-evidence/anchor-reference outputs first.
   Explicitly list reusable native cells only if supported, complete, same
   source/index/A seals, recipe,60-epoch history, steps and strict checkpoint.
   Raw `plain` maps to `logits`, `plain_aug` to `logits_aug`; earlier input-evidence
   cells map by exact name. Reference/scalar gates are not reused as ordinary
   heads. Failed reuse is a blocker, not permission to silently train again.
   Optional backbone rows must be chosen before execution; missing assets must
   be reported, not manufactured or replaced by an additional visual export.
5. Make an external mode600 private assets JSON using README's schema, with
   actual absolute paths and explicit reuse. Never put this JSON, printed
   plan, environment values, input IDs or private audit into Git. Select a
   brand-new private external output, parent already exists, and a new external
   launcher log. Do not reuse a completed/partial output as a resume target.
6. Set `FSN_BENCH_PYTHON`, `FSN_BENCH_ASSETS`, `FSN_BENCH_OUTPUT`,
   `FSN_BENCH_COMMIT`, `FSN_BENCH_GPUS`, `FSN_BENCH_UUIDS` privately. Run README's
   CPU command, then its default `run_suite` command with actual values. It
   validates code and asset SHAs, counts, finite tasks, parameters, reuse and
   queue mapping without GPU probes/output creation. Report actual paths,
   task/new/reused counts and plan status to the user; do not call it CUDA
   preflight or completed training.
7. On explicit run authorization use the **same** command plus `--execute`
   under nohup with outer `env -u CUDA_VISIBLE_DEVICES`, noclobber and a new
   private log. Record the real launcher PID. Worker pids.tsv records stage,
   backbone, task, PID, physical index and UUID. Every worker is one GPU/cuda:0.
   CPU tests and genuine823 replay (including reused plans/metrics) precede
   all registered batch4 CUDA three-step smokes. **All smokes pass before any
   formal training.** Float32 is deliberate for numeric frozen-feature heads.
8. Fixed queue interleaves backbones within the registered12-task ordering and
   skips explicit completed reused training. Each wave waits all peers and
   validates sealed complete checkpoints/histories/results before next wave.
   Runtime failure/OOM/NaN/zero gradient: preserve output, wait running peers
   naturally, stop later waves, report the specific blocker. No retries/resume,
   hidden batch/threshold/loss changes, new seed or new model.
9. After first complete epoch check actual history and gradients, losses,
   val823 F1, best and task/GPU PID. Estimate remaining time from measured
   epoch duration and actual remaining jobs; do not promise an overnight ETA.
   Idle GPU is not success. All newly trained cells must have exact60 epochs,
   full step counts, normal7372 visits and anchor-plan coverage, valid best and
   result. Reused cells must replay as old read-only results, not new runs.
10. All registered cells complete before unified evaluation/curves/report.
    Compare within each A before cross-backbone increments; prioritize the
    claim map, ordinary/augmentation/simple protection controls and correction
    retention/harms. Report full7 classes including the six-sample class,
    two-direction errors, clinical/network/singleton/duration/record aggregates,
    native and identical eligible intervention cohorts, input contrasts and
    mature-head joint loss contrast, resources and conditional record intervals.
    Existing target-reference negative results must not be erased or turned
    into evidence of a win. This comparator queue creates no new gate.
11. Only privacy-reviewed top-level public_safe_summary.json, CSVs, SVG curves
    and REPORT.md may be exported. Never upload weights, feature indexes/data,
    labels, plans, private audit/cache, individual predictions or raw/path logs.
    Code/config/tests/docs and safe aggregates can go to an independent
    nonmain branch; never merge main. Report links plus exact commit/status and
    stop. No new experiment/visual training/seed/monitor follows automatically.

This code delivery has CPU checks only. Runtime CUDA checks and actual server
training remain to be executed; neither is implied by a pushed branch.
