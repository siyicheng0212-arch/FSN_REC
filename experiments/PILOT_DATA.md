# FSN pilot data layer

This module provides a model-neutral, fail-fast input pipeline for a small
three-model comparison. It never edits the source manifests or videos.
Generated pilot manifests contain local absolute video paths, and NPZ files
contain decoded research frames. Both output directories are gitignored and
must remain local-only.

Build deterministic, class-balanced manifests (default: 4 clips per class in
each split):

```bash
python -m experiments.pilot_data build \
  --input-dir processed_external/manifests \
  --output-dir experiments/pilot_manifests \
  --per-class 4 \
  --verify-decodable
```

`--verify-decodable` decodes two low-resolution probe frames from each proposed
clip. An undecodable source rejects the whole clip, is counted in `summary.json`,
and is replaced deterministically; it never causes frame substitution.

Decode each split once. Frames are sampled at equal temporal-bin centres,
center-cropped/resized by the binary supplied by `imageio_ffmpeg`, and stored
only as `cache/<split>/<clip_id>.npz`:

```bash
python -m pip install -r experiments/requirements-pilot.txt
for split in train val test; do
  python -m experiments.pilot_data cache \
    --manifest "experiments/pilot_manifests/${split}.jsonl" \
    --cache-dir experiments/pilot_cache \
    --num-frames 8 --crop-size 224
done
```

`PilotClipDataset` returns a dictionary containing:

- `video`: float32 tensor `[T,C,H,W]` in `[0,1]`
- `label`: integer class ID
- `clip_id`: opaque cache/sample ID
- `source`: collection name
- `duration`: annotated clip duration in seconds

No failed decode is replaced. Existing invalid/stale caches are errors rather
than silently rebuilt. CLI errors and progress never print source paths.

Run checks:

```bash
python -m unittest experiments.test_pilot_data
python -m experiments.pilot_data smoke
```
