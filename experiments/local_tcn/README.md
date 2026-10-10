# Local TCN receptive-field ablation (Scheme B)

This branch adds a **configurable dilation schedule** to the existing residual
TCN. It does not alter the default model and does not start training.

## Exact change

| Setting | Dilation | Theoretical receptive field |
|---|---|---:|
| Historical/default TCN | [1, 2, 4] | 15 candidate-clip positions |
| **Scheme B: local TCN** | **[1, 1, 1]** | **7 candidate-clip positions** |
| Optional medium-range control | [1, 1, 2] | 9 candidate-clip positions |

The receptive field is computed as 1 + 2 * sum(dilations) because each of the
three blocks contains one kernel-width-3 temporal convolution. Channel width 64,
three residual blocks, kernel size 3, pointwise mixing, dropout, optimizer,
loss, data split and checkpoint-selection rule are unchanged. Parameter count
is therefore unchanged by the dilation schedule.

The code preserves TCNConfig(... )'s historical default schedule. The existing
sealed VideoMAE evidence schema also omits the new field when the schedule is
implicit, so existing full-context run artifacts remain compatible.

## Run the local TCN baseline

Use the same CUDA Python, frozen feature index, and sealed source-protocol
directory as the prior completed VideoMAE/TCN run. The source protocol's parent
must still contain the original evaluation/private_protocol.json and
evaluation/public_safe_summary.json. Do not use another split or rebuild
the feature export.

```bash
python -m unittest tests.test_local_tcn tests.test_relation_tcn_baseline -v

python -m experiments.relation.tcn_baseline \
  --index /absolute/path/to/videomae/features/index.jsonl \
  --source-protocol-dir /absolute/path/to/sealed/source_protocol \
  --output /absolute/path/to/new_private_output/local_tcn_seed42 \
  --device cuda --seed 42 --epochs 30 --patience 5 \
  --batch-size 16 --hidden-dim 64 --blocks 3 --dropout 0.1 \
  --dilations 1,1,1
```

Choose a fresh output path that does not already exist. The script saves
best.pt, history.json, private_protocol.json,
private_val_predictions.jsonl, and public_safe_summary.json. Check the
summary's dilations=[1,1,1] and receptive_field_clips=7.

The baseline script is a direct training entry point, not the guarded
multi-stage launcher. Before running on a shared server, inspect current GPU
owners and use the server's established process/output safeguards.

## Interpretation and next comparison

This commit provides a runnable **local ordinary-TCN baseline**, not a result.
No GPU training or performance claim is included.

First compare its full seven-class Macro-F1 and accuracy with the historical
[1,2,4] TCN under the same data protocol and seed. To test whether the
center-preserving contextual increment itself helps with the smaller receptive
field, its full and center-only branches must both be instantiated with
dilations=(1,1,1) and initialized from this local TCN's checkpoint. Do not
reuse the [1,2,4] checkpoint as the local model's trained checkpoint or claim
that the existing 90.79% context result transfers to this architecture.

For a clean paper comparison, run paired seeds 42, 10, and 1217 with separate
output directories. Select checkpoints by development Macro-F1; val823 remains
a repeatedly used development set, not an independent test set.
