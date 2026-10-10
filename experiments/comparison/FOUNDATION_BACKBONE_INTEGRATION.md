# Foundation video backbone integration plan

This document defines the controlled first-stage integration of V-JEPA 2 and InternVideo-Next into the FSN visual-backbone comparison. It deliberately does not claim that either adapter is implemented or tested yet.

## Models to add

| Registry name | Official source | First-stage checkpoint | First-stage protocol |
|---|---|---|---|
| `vjepa2_vitl` | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | `facebook/vjepa2-vitl-fpc64-256` | Freeze encoder, train 7-way linear head |
| `internvideo_next_base` | [OpenGVLab/InternVideo-Next](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo-Next) | `revliter/internvideo_next_base_p14_res224_f16` | Freeze encoder, train 7-way linear head |

The InternVideo-Next collection lists a Base checkpoint (91M parameters) and a Large checkpoint (about 0.3B parameters). Base is the preferred first integration target to control GPU memory and engineering cost. V-JEPA 2 official checkpoints include ViT-L, ViT-H and ViT-G; ViT-L is the initial target, not the largest model.

## Required adapter contract

Both adapters must implement the same downstream interface without pretending their original preprocessing is identical:

- Input: one cached clip `uint8 [36, 3, 224, 224]`.
- Temporal selection: use the benchmark's frozen 16-frame index list, preserving clip identity and order.
- Preprocessing: model-specific, explicitly recorded in each run's audit JSON; do not silently reuse Kinetics-400 normalization.
- Encoder: frozen for linear probing; use `eval()` so dropout / normalization state does not drift.
- Feature output: convert the official model output into one finite per-clip embedding. Prefer the model's documented pooled/CLS output; only mean-pool token outputs when that is the supported feature interface.
- Head: a newly initialized 7-class linear layer trained on train7372 only.
- Metrics: the same seven-class Macro-F1 and Accuracy on val823, plus inference throughput, peak CUDA memory, parameter count and checkpoint/model revision.
- Split safety: never read or select checkpoints against an independent test split in this stage; val823 is development validation only.

## Implementation gates

1. Confirm the official model API and checkpoint format from the upstream code/revision.
2. Add one adapter at a time; run a CPU or single-batch GPU smoke test with output-shape and finite-value assertions.
3. Verify that the encoder is frozen and only the 7-class head receives gradients in linear-probe mode.
4. Run the same frozen manifest/cache preflight used by `run_vision_backbone_benchmark.py`.
5. Run a short overfit/sanity test before any full benchmark.
6. Add tests for each adapter and document dependency pins. Do not add a model to the paper-results table until its run has completed successfully.

## Fairness caveats

- This is a representation comparison, not a claim that all models have identical pretraining data or objectives.
- VideoMAE-V2's existing result is an already fine-tuned sealed reference; linear-probe results for new models must be labelled as a separate screening table, not ranked directly against that fine-tuned result.
- For the final paper comparison, either fine-tune all selected backbones under the same protocol or report frozen-feature and fine-tuned comparisons as two separate tables.
- A gain on val823 is exploratory model selection, not an independent generalization result.
