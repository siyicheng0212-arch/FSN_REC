# FSN_REC

## CVM 2027 paper revision (opt-in, no performance claim)

The `cvm/` package implements a controlled study of the original shared-video-
backbone clinical hierarchy: flat and capacity controls, auxiliary supervision,
hard/soft decoding, actual-router and fair-oracle diagnostics, fixed random
taxonomies, recording-group paired bootstrap, and strict manifest/weight checks.
It does not modify or resume previous AdaFocus/local-motion experiments.

- [Research questions and experiment design](docs/cvm/EXPERIMENT_DESIGN.md)
- [Modern baselines, comparison and ablation matrix](docs/cvm/EXPERIMENT_MATRIX.md)
- [Literature: hierarchy and attribution](docs/cvm/literature_hierarchy.md)
- [Literature: video and procedural benchmarks](docs/cvm/literature_video.md)
- [Commands and environment](docs/cvm/RUNBOOK.md)
- [Server Codex prompt](prompts/run_cvm_codex.md)
- [Full predeclared development matrix prompt](prompts/run_cvm_full_matrix_codex.md)

Modern baselines are R(2+1)D-18, MViT-V2-S, VideoMAE ViT-B16, and
VideoMamba-Ti16. `formal` declares 13 configurations x 3 seeds = 39 training
jobs; optional `extended` includes sampling/weight/frozen-feature sensitivities
(57 jobs). Each stage is opt-in and dry-run by default. Post-training
`robustness` evaluates fixed checkpoints without training. Source-holdout
protocol derivation, aggregate comparison/ablation tables and actual CUDA
deployment profiling have separate entry points. No medical benchmark score
or CUDA readiness is claimed from unit tests. Stored model parameters include
unused audit heads; active deployment and training counts are reported apart.

This is research infrastructure, not an already validated new algorithm.
Previously reused validation data cannot be relabelled as independent test.
Generated private predictions, video, annotations, identities, logs, features,
caches and weights must not be pushed to this public repository.

```bash
python -m unittest discover -s tests -p 'test_cvm_*.py' -v
python -m cvm.protocol --help
python -m cvm.train --help
python -m cvm.analysis --help
```

Code release for FSN temporal action recognition, including reproducible data
cleaning/splitting tools and an FSN-specific Uni-AdaFocus-TSM candidate model.

## Model comparison

Two complete Python source trees are kept side by side:

- `models/Uni-AdaFocus-TSM-original/`: unchanged official Python baseline from
  `LeapLabTHU/Uni-AdaFocus`, commit
  `8846488310fdd4a18412608006030643e794c36e`.
- `models/Uni-AdaFocus-TSM-FSN/`: the FSN candidate Python implementation.

Upstream experiment READMEs, checkpoint names, and external artifact links are
not mirrored in this public code release. Refer to the official upstream
repository for those materials.

The modified tree is opt-in: `--fsn_local_adapter none --fsn_interaction none`
keeps the original baseline path. See `models/COMPARISON.md` and
`configs/fsn_adafocus_ablation.json` for the exact changes and A–G ablations.

## Data tools

`data_tools/build_dataset.py` normalizes the three observed ELAN TXT export
formats, preserves provenance/QC metadata, and creates group-disjoint 8:1:1
train/validation/test manifests. Raw video, TXT files, patient metadata,
generated clips, frames, checkpoints, and experiment logs are intentionally
excluded from this repository.

The code also detects ELAN rows expressed as integer milliseconds and
normalizes them to seconds.

## Tests

The FSN modules have standalone PyTorch tests:

```bash
python -m unittest tests/test_fsn_modules.py -v
```

## Runnable three-model pilot

`experiments/` now provides a fail-fast video decoder/cache, unified metrics,
model wrappers, baseline-equivalence checks, a three-model train/validation/test
runner, and a one-clip overfit sanity test. Local manifests, decoded frames,
predictions, and result files are gitignored.

The CPU-runnable third architecture is MViT-V2-S for pipeline validation. The
formal recent-model protocol uses VideoMamba-Ti on Linux CUDA; see
`configs/three_model_protocol.json`.

Full end-to-end training follows the official Uni-AdaFocus environment and
requires the complete private video collection. No model-performance claim is
made by this code release before those experiments are run.

## Third-party license

The Uni-AdaFocus source is MIT licensed. Its license is preserved at
`models/UNI_ADAFOCUS_LICENSE`.
