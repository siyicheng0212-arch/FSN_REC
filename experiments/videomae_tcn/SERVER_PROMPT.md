# Server handoff: one VideoMAE V2 + ordinary TCN baseline

This document is a run protocol, **not permission to start training merely by
reading it**. Obtain the user's authorization for execution. Do not resume
monitoring or any old training. Read README.md and preserve the old outputs.

1. Use a separate clean worktree on `codex/fsn-videomae-tcn`; pin the exact full
   delivered commit. Never change a running checkout or main. Check process
   inventory and relevant output markers before any launch. If an identical
   experiment already runs, observe it rather than starting a duplicate.
2. Find the previously completed VideoMAE V2 ViT-B frozen export, not the old
   AdaFocus/Original export. Require its `protocol.json`, numeric feature SHA,
   full index8195, correct VideoMAE schema and [8,768] spatial mean/max tokens.
   Locate the original full7372/823 sealed source protocol: audit, metadata,
   edges and both chain files. Do not read test or re-export A. The launcher
   recomputes A accuracy, F1 and confusion from the frozen logits and requires
   equality to the export. Do not fabricate an audit or substitute templates.
3. Identify a CUDA Python with torch and numpy. Inspect nvidia-smi's actual
   indices, UUIDs, RTX3090 name and compute PIDs. This ordinary baseline needs
   only **one** GPU, not two-card DDP. Select an idle card without stopping any
   existing task. Record its full physical UUID and index; child cuda:0 means
   this UUID, not necessarily physical GPU0. Do not inherit outer
   CUDA_VISIBLE_DEVICES. Use only the previously authorized SSH alias if a
   remote connection is needed; do not solicit passwords or swap credentials.
4. Set the README's variables to actual absolute paths. Choose new private
   output/log paths outside the repository and sealed input directories.
   First run the default read-only plan. Report actual paths, input/A seals,
   commit, count, dimensions, parameter count and intended GPU mapping.
   The plan intentionally does not establish GPU availability or CUDA success.
5. When execution is authorized and all seals/plan checks pass, use the same
   command plus --execute with nohup as documented. The launcher performs
   CPU tests, real frozen-feature CUDA three-step updates, fresh ordinary TCN
   training, and strict same-best-checkpoint evaluation. Do not bypass a
   failure, change hyperparameters, add a gate/seed, restore a checkpoint,
   modify the split or overwrite old artifacts. Report exact blocker and
   preserve the scene on OOM/NaN/incorrect replay/missing assets. The float32
   head smoke is intentional; do not convert it to a new bf16 training recipe.
6. After launch report the actual launcher/worker PIDs, physical GPU/UUID,
   stage and smoke outcome. After the first full epoch check history coverage,
   finite classifier/stem/blocks gradients and best.pt. Only all stage exits0
   plus complete suite/evaluation result mean training is completed.
7. Deliver A versus VideoMAE+ordinary TCN aggregate metrics and limitations
   from public_safe_summary.json. Show Accuracy from the same best-F1
   checkpoint, full seven-class metrics, both confusion directions,
   corrections/harm, singleton/neighbor/source/duration/record aggregates,
   elapsed time/parameters/VRAM. Do not say candidates are verified clip
   boundaries, that A was retrained, or that val823 is an independent test.
   Do not claim all improvements come from neighbors: the legacy TCN also
   learns a correction for singleton clips. No automatic further experiments.

Do not upload feature files, checkpoints, videos/annotations, IDs, per-clip
predictions, run_protocol.json or raw/private-path logs. Any optional report
export requires inspected deidentified aggregates in a separate non-main
branch. Stop after the requested report; no automatic main merge.
