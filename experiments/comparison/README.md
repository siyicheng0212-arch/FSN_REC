# Model comparison benchmarks for FSN action recognition

This folder provides two matched comparison tracks. **Code is committed; neither
benchmark has been trained or tested in the current environment.** Run the unit
tests before any GPU run.

## Literature-grounded choice of baselines

### Visual backbones

The current main reference is the fine-tuned VideoMAE-V2 ViT-B/16 frozen-A
checkpoint. The visual runner now includes this **exact sealed A logit export**
as a reference row on the same val823 clips, alongside three Kinetics-supervised
alternatives and two newer self-supervised foundation-encoder probes:

| Backbone | Model family | Pretraining used by code | Reason to include |
|---|---|---|---|
| Current VideoMAE-V2 ViT-B/16 | Masked-video-pretrained ViT | Existing FSN-finetuned K710-distilled checkpoint | Exact sealed A-only reference row; prior seed-42 val Macro-F1 89.70% |
| MViT-V2-S | Multiscale video Transformer | Torchvision Kinetics-400 supervised weights | Hierarchical spatial-temporal feature scales |
| Video Swin-T | Local-window video Transformer | Torchvision Kinetics-400 supervised weights | Local spatiotemporal attention with shifted windows |
| R(2+1)D-18 | Factorized 3D CNN | Torchvision Kinetics-400 supervised weights | Convolutional control with factorized spatial/temporal operations |
| V-JEPA 2 ViT-L/16 | Self-supervised video foundation encoder | Official V-JEPA 2 ViT-L local checkpoint (target_encoder) | Tests video representations learned through self-supervised feature prediction |
| InternVideo-Next Base (91M) | Self-supervised video foundation encoder | Pinned Hugging Face checkpoint revliter/internvideo_next_base_p14_res224_f16 | Adds a newer video representation without video-text pretraining |

Relevant papers:

- VideoMAE: *Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training*, NeurIPS 2022 — [paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/416f9cb3276121c42eebb86352a4354a-Paper-Conference.pdf).
- VideoMAE V2: *Scaling Video Masked Autoencoders With Dual Masking*, CVPR 2023 — [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_VideoMAE_V2_Scaling_Video_Masked_Autoencoders_With_Dual_Masking_CVPR_2023_paper.html).
- MViT: *Multiscale Vision Transformers*, ICCV 2021 — [paper](https://openaccess.thecvf.com/content/ICCV2021/html/Fan_Multiscale_Vision_Transformers_ICCV_2021_paper.html).
- Video Swin Transformer, ICCV 2021 — [paper](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Video_Swin_Transformer_ICCV_2021_paper.html).
- R(2+1)D: *A Closer Look at Spatiotemporal Convolutions for Action Recognition*, CVPR 2018 — [paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_A_Closer_Look_CVPR_2018_paper.html).
- X3D is another established efficient 3D-CNN family — [CVPR 2020 paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Feichtenhofer_X3D_Expanding_Architectures_for_Efficient_Video_Recognition_CVPR_2020_paper.html). It is not in the default suite because this comparison intentionally uses the pinned torchvision model/weight API without adding PyTorchVideo dependencies.

The existing A row is read from the SHA-sealed frozen feature index and checked
against the exact manifest/cache protocol. MViT, Swin3D and R(2+1)D use
supervised Kinetics-400 weights. V-JEPA 2 and InternVideo-Next are run as
**frozen-encoder linear probes only** in this benchmark: we train a fresh
seven-class linear head over mean-pooled video tokens, rather than claiming to
reproduce the upstream pretraining or full fine-tuning recipes.

V-JEPA 2 loads the official [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
ViT-L frozen-evaluation encoder and a local checkpoint containing
target_encoder weights. It needs the upstream repository and its dependencies
(e.g. the versions required by its environment, commonly including xformers).
InternVideo-Next Base is loaded using Transformers remote code from
[revliter/internvideo_next_base_p14_res224_f16](https://huggingface.co/revliter/internvideo_next_base_p14_res224_f16)
at a pinned revision. Although the model appears in the
[OpenGVLab InternVideo-Next collection](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo-Next),
the actual checkpoint repository owner is revliter; inspect/review this remote
code before enabling trust_remote_code=True in a production environment.

The current VideoMAE-V2 reference uses a different pretraining recipe and has
already been fine-tuned on FSN. A frozen foundation linear probe and a fully
fine-tuned VideoMAE checkpoint are not a pure architecture-only comparison.
Use the new probes to screen representation quality first; only invest in
larger training runs after checking the result and the model's own license,
provenance and compute requirements.

### Temporal heads on identical frozen evidence

The temporal runner evaluates multiple heads on the **same frozen VideoMAE
evidence**, exact same train7372/val823, and identical legal source-protocol
chains:

| Model | What it tests | Status |
|---|---|---|
| a_only | No learned temporal head | Mandatory reference |
| markov_viterbi | Train-only, Laplace-smoothed first-order label transition prior with Viterbi decoding | Simple class-order prior baseline; tests whether a neural head adds value beyond a basic transition model |
| mlp | Trainable per-clip correction without neighbors | Capacity/no-context control |
| bigru | Bidirectional recurrent context | Strong surgical-activity-style sequential baseline |
| tcn | Residual TCN with dilations [1,2,4] (receptive field 15 clips) | Existing temporal baseline |
| tcn_local | Same three-block TCN with dilations [1,1,1] (receptive field 7 clips) | Parameter-matched local-context control |
| mstcnpp_adapted | Three-stage TCN-inspired clip-prediction refinement | Multi-stage temporal refinement baseline |
| local_transformer | Transformer with a local temporal attention mask | ASFormer-inspired local-connectivity baseline |

Relevant papers:

- He et al., *An Empirical Study on Activity Recognition in Long Surgical Videos*, ML4H 2022. It compares visual backbones and temporal models and reports that Swin Transformer + BiGRU performs strongly on some surgical-action benchmarks — [paper](https://proceedings.mlr.press/v193/he22a.html).
- *MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation*, TPAMI — [paper](https://arxiv.org/abs/2006.09220).
- *ASFormer: Transformer for Action Segmentation* — [paper](https://arxiv.org/abs/2110.08568).

**Important task distinction:** MS-TCN++ and ASFormer were developed for dense temporal
action segmentation. Here they are explicitly adapted to sequences of already
cut clips with one class label per clip; the implementations are not claimed to
be official or exact reproductions, and their dense-segmentation paper scores
are not directly comparable to FSN clip-level Macro-F1. The source-protocol
chains have median length 2 and maximum length 10 in the prior audit, so report
chain-length support and don't overstate evidence about very long-range context.
The Markov prior is estimated strictly from train-only labels on these eligible
chains, with Laplace smoothing; it is a simple order-prior control, not verified
clinical workflow truth. The `tcn` and `tcn_local` rows keep depth, width,
kernel and parameter count fixed while changing only the receptive field
(15 versus 7 candidate clips), making this the matched comparison for the
local-TCN experiment.

A crucial limit: under the current source-rule protocol, network-video candidate
adjacencies are generally cut, so network clips mostly become singleton
sequences. These temporal-head comparisons therefore evaluate context on the
eligible clinical chains and preserve independent visual classification for
network clips; **they do not evaluate a learned relation model on potentially
valid network-video adjacencies**. That separate question requires the proposed
R module and verified or carefully constructed relation evaluation pairs.

## Data and fairness contract

Both scripts only use the sealed train7372 / val823 protocol. They never read
the test split. val823 is a repeatedly used development set, **not an
independent test**.

The visual benchmark uses identical cached videos and the same 16 temporal
sample positions for every model, resets the class head to seven labels, and
supports:

- **linear_probe**: freeze the pretrained encoder and train the seven-class head.
  Use this for low-cost screening of pretrained representations.
- **finetune**: train the whole backbone and head. Use this mode for the more
  meaningful fine-tuned backbone comparison to the existing fine-tuned A model.

The Kinetics-supervised wrappers use each torchvision weight's normalization
and resize/center-crop geometry after the existing square cache. MViT and Swin
receive 224-square crops; R(2+1)D receives a 112-square crop after its 128x171
resize. Both foundation probes use their model-specific normalization and
224/256-square input settings. Every model samples the same 16 temporal positions
from the 36-frame cache. Because the original videos were already center-cropped
to 224 when cached, no wrapper can recover the uncropped original frame; each run
records preprocessing and checkpoint provenance in its audit.

**Interpret the linear-probe run only as a screening test.** The current
VideoMAE reference is already fine-tuned on FSN; the Kinetics alternatives freeze
their pretrained encoders during linear probing, and the two foundation models
also freeze their encoders. These scores answer whether the pretrained
representations are useful with a simple classifier; they are not a fair
architecture-only ranking against the fully fine-tuned VideoMAE reference.
For a stronger backbone comparison, full fine-tuning is available for the three
torchvision Kinetics alternatives. The foundation-model adapters deliberately
reject `--mode finetune` until their upstream training/evaluation recipes can
be implemented and validated.

The temporal benchmark uses one fixed set of VideoMAE per-tubelet global/local
features and A logits. It does not re-extract visual features and does not read
source kind or action labels as model inputs. Structural cuts and ineligible
neighbors come from the existing sealed protocol. Source-policy adjacency labels
are weak metadata labels, not verified edit-boundary truth.

Metrics include overall seven-class Macro-F1 and accuracy, per-class results,
paired corrections/harm relative to A, sweep/reperfusion confusion, source and
duration slices (<5s, 5–15s, ≥15s), model size, validation clips/second, elapsed
time and peak CUDA memory where available. Both benchmark runners preserve the
support count for each duration slice and report present-class Macro-F1 alongside
the fixed seven-class Macro-F1. Slice metrics must be interpreted with their
support because some slices may not contain every class.

## 1. Run unit tests first

From the repository root and the same environment used for the existing
VideoMAE experiments:

~~~bash
python -m unittest tests.test_dual_timescale tests.test_comparison_benchmarks -v
~~~

The benchmark code has not been run in the user's GPU environment. Do not skip
this test or run several GPU suites simultaneously.

## 2. Temporal-head comparison (low cost)

Create the output parent first (for example, with mkdir -p /absolute/path/to/new_private_output). Use the same sealed feature index and source-protocol directory as the current
VideoMAE/TCN experiments. Replace the sample paths with their existing absolute
paths. Keep output outside the repository and all input directories.

~~~bash
mkdir -p /absolute/path/to/new_private_output
python -m experiments.comparison.run_temporal_benchmark \
  --index /absolute/path/to/videomae/features/index.jsonl \
  --source-protocol-dir /absolute/path/to/sealed/source_protocol \
  --output /absolute/path/to/new_private_output/temporal_heads_seed42 \
  --device cuda:0 \
  --models a_only,markov_viterbi,mlp,bigru,tcn,tcn_local,mstcnpp_adapted,local_transformer \
  --markov-smoothing 1.0 --markov-strength 0.25 \
  --seed 42 --epochs 30 --patience 5 \
  --batch-size-segments 16 --hidden-dim 64 --dropout 0.1 \
  --tcn-dilations 1,2,4 \
  --transformer-layers 2 --transformer-heads 4 --local-radius 4 \
  --lr 0.001 --weight-decay 0.01
~~~

This is one-seed screening. Repeat the baseline and finalists on paired seeds
42, 10 and 1217 before claiming robustness; don't select the final method
solely from one repeated-use validation split.

## 3. Visual-backbone comparison (first screen)

Pass the same full manifests and 36-frame cache already used by the VideoMAE
V2 run. The script validates the manifest SHA, class coverage, cache mapping, cache metadata, and the existing A export seal, checkpoint hash, source-audit hash and exact clip coverage before creating an output directory. Add --existing-a-index pointing to the same sealed VideoMAE features index.jsonl used by the temporal benchmark and --reference-source-protocol-dir pointing to the matching source-protocol directory whose audit hash is recorded in the export seal. It may need internet access
to fetch the named torchvision Kinetics-400 weights if they are not already in
the local PyTorch weight cache.

Quick frozen-feature/linear-probe screen (first create the output parent):

~~~bash
mkdir -p /absolute/path/to/new_private_output
python -m experiments.comparison.run_vision_backbone_benchmark \
  --manifest-dir /absolute/path/to/frozen/manifests \
  --cache-dir /absolute/path/to/cache36 \
  --existing-a-index /absolute/path/to/videomae/features/index.jsonl \
  --reference-source-protocol-dir /absolute/path/to/sealed/source_protocol \
  --output /absolute/path/to/new_private_output/vision_linear_probe_seed42 \
  --device cuda:0 \
  --models mvit_v2_s,swin3d_t,r2plus1d_18 \
  --mode linear_probe --seed 42 --epochs 5 --patience 2 \
  --batch-size 2 --eval-batch-size 2 --accumulation-steps 4 \
  --workers 4
~~~

### Add the two foundation encoders (linear-probe screen)

V-JEPA 2 uses its upstream checkpoint and repository explicitly; download both
before running and use a fresh output directory. The InternVideo-Next checkpoint
is downloaded by Transformers from the pinned model repository if it is not in
the local cache. This run contains only frozen-encoder probes plus the sealed
VideoMAE A reference; it does not run either model's official fine-tuning recipe.

~~~bash
mkdir -p /absolute/path/to/new_private_output
python -m experiments.comparison.run_vision_backbone_benchmark \
  --manifest-dir /absolute/path/to/frozen/manifests \
  --cache-dir /absolute/path/to/cache36 \
  --existing-a-index /absolute/path/to/videomae/features/index.jsonl \
  --reference-source-protocol-dir /absolute/path/to/sealed/source_protocol \
  --output /absolute/path/to/new_private_output/foundation_linear_probe_seed42 \
  --device cuda:0 \
  --models vjepa2_vitl,internvideo_next_base \
  --mode linear_probe --seed 42 --epochs 5 --patience 2 \
  --batch-size 1 --eval-batch-size 1 --accumulation-steps 4 \
  --vjepa2-repo /absolute/path/to/vjepa2 \
  --vjepa2-checkpoint /absolute/path/to/vitl.pt \
  --workers 4
~~~

If V-JEPA 2 is not ready, run InternVideo-Next separately by setting
`--models internvideo_next_base` and omitting the two V-JEPA-specific flags.
Before using InternVideo-Next, review the pinned remote code. Do not use
`--random-init` for either foundation model.

For the paper-grade comparison among the three torchvision alternatives, run
full fine-tuning on the same protocol. Do not include vjepa2_vitl or
internvideo_next_base in this command; the current adapters intentionally
support linear probing only. This takes materially more GPU time. Use a separate
fresh output path:

~~~bash
mkdir -p /absolute/path/to/new_private_output
python -m experiments.comparison.run_vision_backbone_benchmark \
  --manifest-dir /absolute/path/to/frozen/manifests \
  --cache-dir /absolute/path/to/cache36 \
  --existing-a-index /absolute/path/to/videomae/features/index.jsonl \
  --reference-source-protocol-dir /absolute/path/to/sealed/source_protocol \
  --output /absolute/path/to/new_private_output/vision_finetune_seed42 \
  --device cuda:0 \
  --models mvit_v2_s,swin3d_t,r2plus1d_18 \
  --mode finetune --seed 42 --epochs 30 --patience 5 \
  --batch-size 2 --eval-batch-size 2 --accumulation-steps 4 \
  --backbone-lr 0.0001 --head-lr 0.0005 --weight-decay 0.05 \
  --workers 4
~~~

Use a fresh output directory for every run. The script is a direct launcher,
not a GPU-ownership guard and not an automatic-resume mechanism. Inspect current
GPU owners first. Each output contains a private run protocol, status.json,
per-model best.pt, validation prediction rows and an identifier-free aggregate
summary. Never commit the checkpoints or private per-clip output.

## Recommended publication sequence

1. Complete the temporal comparison on identical frozen VideoMAE features.
2. Screen visual backbones with linear probes; full-fine-tune selected
   alternatives under the same fixed split and checkpoint-selection metric.
3. Once a backbone comparison is settled, export that backbone's temporal
   features and compare the same proposed within-clip/inter-clip architecture
   against the same temporal-head baselines. Otherwise, a paper can only claim
   that the temporal module works on VideoMAE, not that it is backbone-agnostic.
4. Lock the protocol before evaluating any genuinely independent, previously
   untouched test set. Do not call val823 a test set.
5. Report at least three prespecified seeds and paired uncertainty by recording
   cluster where the data support it. Parameter count, GPU memory and
   clips/second should accompany accuracy metrics.

A better score is not guaranteed. These experiments are designed to show
whether gains come from visual representation, temporal head, or their
interaction, even if the result is that a simpler baseline is best.
