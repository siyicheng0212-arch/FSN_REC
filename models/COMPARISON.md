# Original vs. FSN model

Both directories contain the complete Uni-AdaFocus-TSM Python source tree so
code review does not depend on reconstructing an upstream version. Upstream
experiment documentation and checkpoint links are intentionally not mirrored.

| Concern | Original | FSN candidate |
|---|---|---|
| Global encoder | MobileNetV2-TSM | unchanged |
| Local encoder | ResNet-50-TSM | unchanged backbone; optional adapter between layer3/layer4 |
| Spatial/temporal selection | Uni-AdaFocus policies | unchanged |
| Training paths | random and policy crops | both retained |
| Policy gradient isolation | official `detach()` sites | unchanged |
| Local feature output | pooled vector | optional layer-4 `3×3` grid plus original pooled branch |
| Fusion | global logits + local logits | optional interaction correction before final output |
| Time alignment | implicit branch sampling | normalized source-frame positions returned by dataset |
| Metrics | accuracy | accuracy plus macro/micro/weighted F1 and per-class support |

## Modified or added files

- `archs/fsn_modules.py` — new residual local adapter and local-context module.
- `archs/uni_adafocus_tsm.py` — opt-in integration; original auxiliary outputs remain.
- `ops/models.py` — exposes local ResNet grids and inserts the adapter.
- `ops/dataset.py` — optionally returns actual sampled source-frame positions.
- `ops/dataset_config.py` — adds a seven-class FSN dataset entry.
- `opts.py` — adds A–G ablation switches and validation selection metric.
- `main.py` — passes frame positions and emits metrics from one prediction stream.

The early-exit model remains unchanged in the first experimental round. It
should only be adapted after fixed-compute ablations establish whether the FSN
modules are useful.
