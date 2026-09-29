# Epoch-boundary resume for future FSN runs

Status: implemented and tested locally; **not retroactive** to the Original
seed-2026 or directional seed-42 processes that started with older code.

Every completed head-warm-up and fine-tuning epoch now writes `last.pt` in
that run's `variant/seed_N` directory. It includes:

- model parameters and buffers;
- the active optimizer, plus the cosine scheduler during fine-tuning;
- phase, completed epoch, history, best epoch/F1, and early-stop stale count;
- Python, NumPy, PyTorch CPU/CUDA, training sampler, worker-generator, and
  augmentation-generator states;
- a fingerprint of the manifest hashes, official checkpoint hash, cache
  location, code commit, PyTorch version, seed, architecture switches, and
  training hyperparameters.

`last.pt` and `best.pt` are written through a temporary file, fsynced, then
atomically renamed. The prior complete checkpoint is kept if a save is
interrupted. The final `result.json` is atomic too. `last.pt` is updated
once **after** validation for each complete epoch, so an interruption during
an epoch loses only that incomplete epoch. If power fails between updating
`last.pt` and `best.pt`, resuming reconstructs the missing best file when
the latest epoch was itself the best.

To continue a new run, launch the **same code commit, same output directory,
same data/cache/checkpoint and every original training argument**, adding:

```text
--resume-from /root/autodl-tmp/<output>/<variant>/seed_<N>/last.pt
```

The trainer rejects a completed `result.json`, changed protocol fingerprint,
missing optimizer/scheduler/RNG state, or an old model-only `best.pt` passed
as `--resume-from`. Do not change `--epochs` to the *remaining* count: it
remains the original total (for example 100), and the scheduler resumes
from its saved position. Resume uses the next epoch after the saved epoch.

This is epoch-boundary continuation, not a promise of bitwise identity across
different PyTorch/CUDA versions or hardware. The cached dataset currently
has no stochastic worker-side transforms; its training sampler uses its own
saved generator so worker recreation cannot change batch order. A Linux
three-epoch check matched the previous trainer's batch order exactly.

The old runs have **only** `best.pt`. They have no optimizer, scheduler,
sampler, or RNG state, and therefore cannot be exactly resumed, regardless
of how many GPUs the restarted instance has. Their checkpoints can be used
for inference or a clearly labeled *new warm-start experiment*; they must
not be described as uninterrupted continuation. The stopped directional
seed-42 run must be restarted as a fresh formal run to enter a fair seed
comparison. Put all `last.pt`, `best.pt`, logs, and manifests on the AutoDL
data disk; never commit them to GitHub.
