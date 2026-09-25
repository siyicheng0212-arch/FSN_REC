# FSN_REC

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
