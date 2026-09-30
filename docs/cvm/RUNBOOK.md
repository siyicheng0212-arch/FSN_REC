# 运行说明：先核验，再做冻结的 CVM 实验

本分支是新实验代码，不会接续旧 AdaFocus/local-motion 分支。这里给的是可运行入口，不表示已经完成服务器训练。研究问题、反证与论文边界见 [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md)。

## 环境与单元检查

在独立 worktree 使用现有 CUDA Python，记录 Python/PyTorch/torchvision/CUDA/GPU 版本。不要为安装依赖覆盖原服务器环境。`requirements-cvm.txt` 列出直接依赖；若 CUDA torch 已可用，只补缺失的 numpy/torchvision。测试使用标准 unittest。VideoMamba 使用独立的官方定制扩展，不能用通用 mamba-ssm 替代。

```bash
python -m unittest discover -s tests -p 'test_cvm_*.py' -v
python -m cvm.protocol audit --help
python -m cvm.train train --help
python -m cvm.run_suite plan --help
```

本地测试包含 fake feature backbone 与真实私有格式的 synthetic cache，不涉及真实医疗样本。Torchvision 原生模型形状曾在 CPU 检查；正式预训练文件、VideoMamba 实际 CUDA 和服务器数据必须另外核验。

## 冻结本地 train/val

manifest 使用现有字段：`clip_id`、`split`、`label_id`、`normalized_label`、`source_collection`、`group_id`、`video_path`、`clip_start_sec`、`clip_end_sec`、`clip_duration_sec`。七类名字与 ID 必须严格吻合。缺失 group 时不允许拿 clip_id 代替独立录制组。

```bash
python -m cvm.protocol audit \
  --train /private/manifests/train.jsonl \
  --val /private/manifests/val.jsonl \
  --prior-eval-manifest /private/history/previous_validation.jsonl \
  --output /private/cvm_protocol_v1
```

`--prior-eval-manifest` 可多次提供；审查范围要覆盖此前所有用于选模型的评估数据。输出 `protocol.json` 仅含聚合，完整冻结 manifest 在 `private/`。相对视频路径需显式 `--video-root`。默认不提供 test；不能因为当前想投稿，就把旧 validation 改为 test。

缓存固定使用已存在的 36-frame/224 uint8 `.npy` 与其 metadata。**主实验选中 16 个等距缓存索引**，之后做各 backbone 的空间预处理；这与旧稿直接32帧采样不保证一致。缓存路由有差异时，生成全新私有适配 manifest 加 `cache_split`，例如 semantic `split=val`、`cache_split=train`；仅指缓存位置，不改变评估角色、clip、group、路径或标签。不得修改原清单/缓存，不能重用 digest 不匹配的缓存。

## 准备权重

R2+1D/MViT 使用 torchvision 原始 K400 文件：完整严格载入400类权重并校验官方文件哈希，再移除旧分类器。可通过显式 `--allow-download` 的一次本地模型初始化取得官方文件；之后所有 suite 都使用同一个确定路径，不各组自行下载。

```python
from cvm.models import build_model
model = build_model("r2plus1d_18", weights="DEFAULT", allow_download=True, seed=42)
print(model.load_report)
```

VideoMamba：固定 [官方 commit](https://github.com/OpenGVLab/VideoMamba/tree/37355c26d0ae99ca2459f6d4044a5f509031a79f)，按该仓库说明编译 custom causal-conv1d/mamba，取得官方 K40016 Ti checkpoint。详情和来源在 [literature_video.md](literature_video.md)。未经核验的 SSv2 8帧、ImageNet 占位权重、随机初始化均不能替代该参照。

VideoMAE：固定 [官方 commit](https://github.com/MCG-NJU/VideoMAE/tree/14ef8d856287c94ef1f985fe30f958eb4ec2c55d)，使用MODEL_ZOO中**ViT-B/K400/800轮预训练后的400类fine-tuned checkpoint**，不是只有自监督encoder的文件，也不是HF转换版、SSv2或v2K710蒸馏权重。适配器严格载入完整400头再移除，特征为mean pooling+fc_norm768。本地官方源码CPU形状验证使用`timm==0.4.12`；先在隔离环境核验，不改可用CUDA安装。含Namespace的官方训练容器需要支持安全白名单的PyTorch，旧版torch只接受安全tensor-only容器，不以unrestricted pickle绕过。官方未发布完整可信checksum，本地SHA仅证明文件固定；还须记录下载来源。真实权重/CUDA待服务器核验。

在仓库外生成私有 `checkpoints.json`，仅 pilot 时可只含第一项：

```json
{
  "r2plus1d_18": "/private/checkpoints/r2plus1d_18-91a641e6.pth",
  "mvit_v2_s": "/private/checkpoints/mvit_v2_s-ae3be167.pth",
  "videomamba_tiny16": {
    "checkpoint": "/private/checkpoints/videomamba_t16_k400_f16_res224.pth",
    "external_repo": "/private/third_party/VideoMamba"
  },
  "videomae_base16": {
    "checkpoint": "/private/checkpoints/videomae_vitb_k400_800e_finetuned.pth",
    "external_repo": "/private/third_party/VideoMAE"
  }
}
```

示例文件名须与实际 installed torchvision enum URL 校验，不靠手工拼名字。权重路径、安装日志和完整 load report 不上传。

## 四卡 smoke 与 pilot

`plan` 仅生成计划，不训练。每个 output 必须不存在。`launch` 不带 `--execute` 仍是 dry-run；带该参数才实际启动。一个 GPU 同时最多一个独立训练任务，四卡不是 DDP。

```bash
python -m cvm.run_suite plan \
  --stage smoke --protocol /private/cvm_protocol_v1/protocol.json \
  --cache-root /private/full_cache_36f224_trainval \
  --weights-json /private/checkpoints.json \
  --output /private/cvm_smoke_v1 --gpus 0,1,2,3

python -m cvm.run_suite launch \
  --plan /private/cvm_smoke_v1/run_plan.json --execute
```

smoke 默认四个 R2+1D 模式、seed42、无 warm-up、1 个真实完整训练 epoch。它比单 batch 多做了一轮，用来验证载入、cache、优化、checkpoint、预测导出和报告整个路径；不能把 smoke 分数拿来当正式结果。若只需先检查 batch，按冻结 shape 做单 batch CUDA FP32/AMP 前后向，记录梯度/显存后再进入 smoke。

四组 smoke 都成功后，在新 output 生成 `--stage pilot` 并执行：四个 R2+1D 模式，seed42，5轮 head warm-up、最多60轮 finetune、patience10；batch2/accum16，有效32；AdamW backbone1e-5/head1e-4、warm-up head1e-3、wd.05、none class weights、clip5。主 best 仅选 finetune validation 七类 macro-F1。默认设置也明确写入 plan/config。

```bash
python -m cvm.run_suite plan \
  --stage pilot --protocol /private/cvm_protocol_v1/protocol.json \
  --cache-root /private/full_cache_36f224_trainval \
  --weights-json /private/checkpoints.json \
  --output /private/cvm_pilot_seed42_v1 --gpus 0,1,2,3

CVM_PYTHON=/path/to/cuda/python bash scripts/run_cvm_4gpu.sh \
  /private/cvm_pilot_seed42_v1/run_plan.json --execute
```

各 run 写 best/last、history、config、load report、私有 val predictions/metadata 与聚合 val_report；result 标明 completed/early_stopped/interrupted/failed。`launcher_logs/pids.tsv`、`exit_codes.tsv` 与 `launcher_result.json` 决定是否完成，不能用 PID 或 best 文件存在代替完成确认。SIGINT 保留 checkpoint，不会自动续训；一个任务失败后不会继续排队启动任务。

OOM 时保留原目录；统一调整同一比较中的 batch/accum 保持有效 batch，并生成新协议输出。不要覆盖，不修一组配置后混入原比较，不恢复旧 extra_wave。

## 正式矩阵

`--stage formal`默认为13配置×3固定seeds `42/2026/2027`，39次独立训练。四backbone的flat/hierarchy，R2+1D的capacity/aux和3random去重合并。完整矩阵与研究问题见[EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md)。`benchmark/ablation`是子矩阵，不与formal重复启动；`sensitivity/representation`为附加对照；`extended`一次去重生成57次上限。所有权重/依赖在任何job启动前核验。显式`--backbones`子集标记正式矩阵不完整。

```bash
python -m cvm.run_suite plan --stage formal \
  --protocol /private/cvm_protocol_v2/protocol.json \
  --cache-root /private/cache36f224 \
  --weights-json /private/checkpoints.json \
  --output /private/cvm_formal_v2 --gpus 0,1,2,3
# plan only. Do not also launch benchmark/ablation with overlapping configurations.
```

R2输入帧数敏感性显式`--frames 32`，仅卷积R2/R3D支持；位置仍从现有36缓存取等距索引，repeat audit分母随16/32变化，未恢复真实PTS。MViT/MAE/Mamba拒绝偷偷改变其固定16帧位置编码。`--backbone-training frozen`保持主干参数和BN统计不变，history/best标为frozen_features；这是官方特征线性probe对照，不是旧稿FSN分阶段精确复现。

不要仅因 pilot 涨点就宣称贡献，也不要因某个随机/seed 不利而从正式矩阵移除。当前服务器关闭，本代码不会替用户启动任务。正式方案、数据角色和资源确认后，在全新 output 生成并执行 formal 计划；不要自动追加未登记的 seed。

## 诊断与聚合

不需要重训即可对已完成 run 的私有预测生成同 checkpoint 硬/软、真实 router、oracle、七类/五类、来源/时长/重复率的聚合。五类主F1保留完整七类的FP/FN，oracle为真实组提示诊断，不是部署指标。

```bash
python -m cvm.analysis aggregate \
  --predictions /private/run/val_predictions.jsonl \
  --metadata /private/run/prediction_metadata.json \
  --output /private/aggregates/run_val.json

python -m cvm.analysis compare \
  --baseline /private/flat/val_predictions.jsonl \
  --candidate /private/hierarchy/val_predictions.jsonl \
  --baseline-metadata /private/flat/prediction_metadata.json \
  --candidate-metadata /private/hierarchy/prediction_metadata.json \
  --output /private/aggregates/paired_val.json

python -m cvm.report \
  --runs /private/seed42 /private/seed2026 /private/seed2027 \
  --output /private/aggregates/seed_summary.json
```

计划对应的对比/消融表由`cvm.report --plans`产生，缺失seed和中止run保留为空，不能当0分/训练完成。不同采样、类权重和frozen设置属于不同配置，不混成seed重复。

```bash
python -m cvm.report --plans /private/cvm_formal_v2/run_plan.json \
  --output /private/aggregates/formal_matrix.json \
  --markdown-output /private/aggregates/formal_tables.md

python -m cvm.run_suite plan --stage robustness \
  --trained-plan /private/cvm_formal_v2/run_plan.json \
  --output /private/cvm_robustness_v2 --gpus 0,1,2,3
```

robustness只对已完成clinical flat/hierarchy的best做val none/static/shuffle，formal完整时24checkpoint×3=72次推理，0次训练，默认不读test。新的none评估带checkpointSHA，`cvm.analysis compare --same-checkpoint-probe`才允许与static/shuffle配对，其他预处理必须一致。采样敏感性比较用`--frame-sensitivity`，只放行同卷积主干16→32及相应repeat审计变化。

表中的跨seedmean±SD与每seed配对bootstrap是两类统计，后者需要私有逐clip预测，不能从聚合表逆推。`cvm.paired`按冻结plan一次生成**全部预声明配对**（包括有信息预算对称的oracle），不会挑更好对照、拟合或推理：

```bash
python -m cvm.paired \
  --plans /private/cvm_formal_v2/run_plan.json /private/cvm_robustness_v2/run_plan.json \
  --output /private/aggregates/declared_pairs_v2 --replicates 2000 --seed 42
```

其中`--seed`仅为bootstrap随机数，训练seeds仍由plan定义；每个缺失/中止pair保留状态。脚本先通过report的协议/配置/权重/seed检查，再核对完整clipID、真值和group，禁止交集匹配。probe与frames对照只通过各自窄开关放行其预声明变化。输出只有去身份聚合；异常保留failed_or_interrupted状态，不能报全部完成。

来源隔离从**已冻结且历史verified_clean的原test**派生，来源名称显式提供，不按评分自动选：

```bash
python -m cvm.source_holdout derive \
  --protocol /private/cvm_protocol_with_clean_test/protocol.json \
  --heldout-source PREDECLARED_DOMAIN \
  --source-selection-exposure predeclared_before_source_scores \
  --output /private/cvm_source_holdout_v1
```

tool不会把原train/val移test；train/val删除该domain，test仅保留原clean test该domain。缺七类/未知历史就报告data_insufficient并停止该项，不能换成好看的domain。用新协议生成独立R2 flat/hierarchy×3seed子矩阵即可起步；更多主干和domain的预算单列。原始source名/父路径只在private，safe summary仅hash别名。来源切片仍是描述，不能替代留出实验。

每个正式checkpoint的资源另用`cvm.profile`：

```bash
python -m cvm.profile --checkpoint /private/run/best.pt \
  --output /private/aggregates/resource_profile.json \
  --batch-sizes 1,4 --warmup 20 --iterations 100
```

这是固定shape合成输入的真实CUDA推理成本测量，不是医疗性能评估；分别列保存、可训练和实际部署参数。flat/aux/capacity部署只算flat头，hierarchy算group/fine头；hard当前向量化实现也算全部fine heads。若未运行profile，延迟/吞吐填待测，不猜FLOPs。

seed42 pilot 汇总显式 `--expected-seeds 42`。缺失/中止 seed 会列出，不能用最有利的另一次运行补成相同 seed。bootstrap 抽整个原始录制组，CI与训练seed SD分别报告。

冻结 checkpoint 的 static/shuffle 诊断通过 `python -m cvm.train evaluate ... --split val --probe static|shuffle`，每个 probe 新目录；在已选16帧上扰动，shuffle保留这些输入帧的多重集合。它们不是跨 clip 顺序、手针定位或临床因果证明。

## 视觉 taxonomy 扩展

```bash
python -m cvm.train export-features \
  --protocol /private/cvm_protocol_v1/protocol.json \
  --cache-root /private/full_cache_36f224_trainval \
  --checkpoint /private/flat/best.pt --output /private/train_features_v1

python -m cvm.visual_taxonomy \
  --features /private/train_features_v1/features.npz \
  --metadata /private/train_features_v1/features.meta.json \
  --output /private/visual_taxonomy_v1.json
```

分组统计只读 train 示例/标签；encoder 若由 validation 选 best，元数据必须保留这一开发集使用，不能声称整个构组过程完全未用 validation。禁止由最终 test confusion 生成组。helper枚举210个相同组大小分区，最大化类均值特征的组内余弦相似度，这是基线聚类，不是创新。该分组可能改变 singleton，需与固定 singleton 随机对照区分。用 `--taxonomy-file` 显式添加扩展，不静默改变原矩阵。

## 独立 test 与发布

test 必须在训练前的协议中冻结，并具有经负责人核验的完整曝光历史。`cvm.train evaluate --split test` 需要显式 `--allow-test` 和匹配 `--locked-protocol-sha256`；代码将拒绝暴露或历史不明的 test。**代码审查只能验证已提供的历史，不认证用户没有漏报。**没有合格 test 时继续开发验证，不能假称已完成最终评估。

输出只发布安全聚合。视频、原标注、逐 clip 预测、身份映射、原始路径日志、缓存、权重、features.npz和运行配置路径留在私有目录。提交到非main独立分支，不自动合并。匿名稿另用匿名入口；这份实名仓库文档不是匿名投稿链接。
