# 当前片段驱动的上下文参考实验

本实验在**已训练的全接 A＋TCN**上继续训练，比较邻居替换训练和动态参考系数。A 始终冻结，输入来自已封存的 A 特征及七类 logits；不重新训练视觉模型，不读取 test，不改变 train7372/val823。val823 是开发验证集，不是独立测试。

代码支持两个独立基座：

| 基座 | 实际实现与起点 |
|---|---|
| `reference` | 已有新参考 `TemporalResidualTCN`：默认四个空洞卷积块，视觉特征进入 TCN，输出修正 logits 加回 A logits；单片段严格返回 A。必须提供同证据、同结构的 `strategy=all` checkpoint。 |
| `legacy` | 用户实际旧 A＋TCN，由真实 factory 和已核实审计文件加载。`adapters.py` 可以适配视觉特征＋A logits/概率的输入布局，但**没有重建未知的三个卷积块旧模型**。旧源文件、权重或审计缺失时停止，无参考版兜底。 |

历史 79.59% 和新参考约 79.20% 只作历史开发结果，不是本轮代码已取得的结果。两个基座分别启动四组，使用各自的同一个原 checkpoint；新旧共八组，不能把不同基座混在同一消融比较中。

## 模块如何工作

在显式声明的原生时序 `Conv1d` 中，保留原中心项和偏置，只调节非中心卷积项：

`output = original_conv(input) + sum((g - 1) * neighbor_tap_contribution)`

`g = 2 * sigmoid(s)`，范围是 `(0,2)`，表示抑制或增强强度，**不是连接可信概率**。标量组共享一个 `s`；动态组用冻结的当前视觉特征、邻居视觉特征、差异和有符号跨度产生 `s`，不读取来源、身份、动作真值或 A 类别分数。动态门先投影到64维，所有被选择卷积层共享门网络。

门末层初始化为零，使初始 `g=1`。加载实际基座权重后，新增模块初始 eval 输出复现原模型，同时保持门末层梯度路径。初始化不会改变数据采样/dropout随机状态。门前层通常在末层更新后才开始获得梯度。

模块没有构造新的排序，也不推断不存在的临床时间。正常候选链沿用封存的同录像、同记录、合法时间间隔规则，结构断点先切段；**小于1的参考系数不保证严格隔断跨层传播**。参考基座是双向离线模型，使用未来片段；旧版的实际方向和单片段行为由真实 factory 保留。

首版只支持单链 `[T,F]`、同长度且对齐的原生 `Conv1d`、奇数 kernel≥3、stride1、对称零填充。因果/custom padding、时间重采样、带额外 hooks、weight normalization/parametrization、跨调用状态等模型不能直接套用；会报错，不能静默修改旧网络。

实际参数在预检和结果中统计。对于参考配置、输入维度3328、dim64：

| 组别 | 新增参数 | TCN＋模块参数（A不实例化、不训练） |
|---|---:|---:|
| continue / aug | 0 | 279,559 |
| scalar_aug | 1 | 279,560 |
| dynamic_aug | 225,665 | 505,224 |

旧模型参数必须根据真实模型统计，不能沿用上表。

## 四组和训练视图

| variant | 模型 | 第二视图 |
|---|---|---|
| `continue` | 原 TCN 继续训练 | anchor的原邻居 |
| `aug` | 原 TCN | 替换anchor附近邻居 |
| `scalar_aug` | 统一可学习系数 | 同样替换 |
| `dynamic_aug` | 当前片段决定邻居系数 | 同样替换 |

各组从同一个基座 checkpoint 开始，新优化器状态一致；每步使用相同正常片段视图和 anchor 视图，保持视图数量、采样计划、最大预算一致。后三组替换计划完全相同。早停仍可能使实际步数不同，结果必须报告实际步数和资源，不能声称实际算力完全相等。

`augmentation.py` 用独立随机状态、固定seed/epoch生成私有计划：每条原结构段覆盖一次，选择anchor；只从 train7372 中同来源集合、不同 group/record/video 的临床片段取 donor。anchor视觉证据、A预测和标签不变，donor的视觉特征与A logits成对替换。扰动视图**只监督anchor**，不把原邻居标签配到donor上。正常链每片标签仍保留。

单片段和无donor片段的第二视图保持干净，按实际基座语义运行。若整个训练集无法构造替换，正式计划失败，不降低条件或重新划分。计划包含 clip IDs，必须留在私有输出，不上传。首轮普通视图损失＋等权anchor损失，使用显式配置的无权重七类CE；不训练“临床=1、网络=0”的来源BCE。旧版若实际使用其他损失，本版停止兼容，需明确扩展后才能声称保持旧设置。

## 旧模型的适配和审计

`legacy_template.json` 是未就绪模板，不是可运行的旧复现配置。先找到真实旧源码/config/checkpoint，核实预测输入是logits还是概率、归一化、布局、卷积/残差/归一化/dropout、单片段策略、分类输出、损失和优化器。

真实 factory 的契约为：

```python
def build(input_dim, **verified_kwargs):
    # 实例化真实旧模型；必要时用 LegacyPredictionAdapter
    # 返回 forward(features[T,F], a_logits[T,7]) -> logits[T,7]
    return actual_audited_model
```

不得仅凭“3个卷积块”猜结构。`LegacyPredictionAdapter` 不额外加残差或校准；若旧版需要特殊归一化或输出，真实 factory 必须实现原语义。

配置必须显式声明 `factory`、`kwargs`、`temporal_paths`、`checkpoint_state_key`、`checkpoint_target`、所有真实源码 `source_dependencies` 和 `legacy_audit`。审计和源码依赖使用**绝对路径**，避免worker切换cwd后含义变化。时序路径是适配后模型的实际路径，例如只有实际采用helper时才可能带 `model.` 前缀。`checkpoint_target` 指定原权重加载到哪个真实子模块，不自动剥除前缀。加载使用严格state dict和 `weights_only=True`，不自动执行外部pickle对象。

私有审计JSON必须满足：

```json
{
  "schema": "fsn-context-utility-legacy-audit-v1",
  "checkpoint_sha256": "真实权重SHA256",
  "source_sha256": {"真实源码绝对路径": "真实SHA256"},
  "fingerprint": {"完整且完全一致的当前冻结证据fingerprint": "这里不是有效占位数据"},
  "semantics": {
    "input_a_predictions": "已核实logits或probabilities",
    "feature_normalization": "已核实的实际预处理说明",
    "input_layout": "已核实的实际布局",
    "temporal_architecture": {"每个真实temporal_path": "实际Conv统计字典"},
    "singleton": "已核实的单片段行为",
    "loss": "CE",
    "optimizer": "与配置完全一致的AdamW或SGD"
  }
}
```

上述是字段说明，不是可直接运行的审计文件。`source_sha256` 必须与factory及完整依赖hash一致；`temporal_architecture` 必须与真实模型的逐路径Conv统计完全一致，包括类、通道、kernel/stride/padding/dilation/groups/bias/padding_mode。fingerprint由严格 `load_bundle` 取得。helper的预测输入、布局、单片段配置还要与审计相符；不能用写一份字符串说明代替核实。完整可执行的工程测试示例见 `tests/test_context_utility_config.py`。

## 预检和正式入口

本目录只消费已封存的完整7372/823冻结A证据。source protocol、feature index、所有feature文件和原A checkpoint标识必须一致；本轮不重新导出A、不改变source chains。默认来源映射包含 `FSN=network`，仍以实际封存协议为准；如既有协议不同，不允许静默改写。

先将代码固定到独立非main分支和干净提交。正式运行期间不得git pull或修改代码。示例中替换各实际路径：

```bash
FSN_UTILITY_PYTHON=/path/to/cuda_environment/bin/python
FSN_UTILITY_INDEX=/private/frozen_features/index.jsonl
FSN_UTILITY_PROTOCOL=/private/sealed_source_protocol
FSN_UTILITY_BASE=/private/full_open_reference/seed_42/all/best.pt
FSN_UTILITY_COMMIT=$(git rev-parse HEAD)

"$FSN_UTILITY_PYTHON" -m pytest -q -p no:cacheprovider tests/test_context_utility_model.py tests/test_context_utility_augmentation.py tests/test_context_utility_config.py tests/test_context_utility_evaluate.py tests/test_context_utility_launcher.py tests/test_context_utility_training.py

"$FSN_UTILITY_PYTHON" -m experiments.context_utility.run_suite \
  --feature-index "$FSN_UTILITY_INDEX" --source-protocol-dir "$FSN_UTILITY_PROTOCOL" \
  --config experiments/context_utility/reference.json --base-checkpoint "$FSN_UTILITY_BASE" \
  --output /private/context_utility_reference_seed42_v1 --gpus 0,1 \
  --python "$FSN_UTILITY_PYTHON" --expected-commit "$FSN_UTILITY_COMMIT"
```

不加 `--execute` 时只读检查：真实输入、严格基座加载、实际参数、epoch1替换可行性、固定验证扰动、代码/资产hash和命令。不会探测GPU或创建输出。legacy缺资产也会在这一步停止。

确认计划后使用相同命令末尾加 `--execute`。两卡安排如下，第一波所有进程正常退出后才能开始第二波：

| 波次 | physical GPU0 | physical GPU1 |
|---|---|---|
| 1 | continue | aug |
| 2 | scalar_aug | dynamic_aug |

四卡只将 `--gpus 0,1` 改成 `--gpus 0,1,2,3`，四组同波独立运行，不是DDP。每个worker通过物理GPU UUID单独设置 `CUDA_VISIBLE_DEVICES`，内部 `--device cuda:0`。不要把worker的cuda:0解释为所有实验挤在物理GPU0。

旧版用核实后的私有legacy配置、旧checkpoint和**另一个全新输出目录**分别执行同样命令；不要复用reference输出，也不要同时在相同卡上启动两个suite。首轮只使用配置中预先声明的seed42，不自动扩展种子。

正式执行要求GPU空闲、输出在代码仓库外且不存在、HEAD符合固定提交。顺序为：本目录单测→真实冻结证据CUDA三步smoke→四组按波次训练→核实四组exit0、best.pt、非空history和完整val823结果→统一评估。smoke不是重新训练A，也不是训练完成；CPU helper仅用于工程单测，不能代替CUDA正式预检。

运行目录含 `run_protocol.json`、`run_commit.txt`、`.launch_once`、冻结配置、`cuda_smoke.json`，及 `launcher_logs/{pids.tsv,exit_codes.tsv,阶段日志}`。每组保存best/last/history/result和私有配对预测。任何失败保留现场；同波已经运行的任务自然结束，后波停止。无自动retry、resume、kill、阈值调整或静默改batch。各依赖阶段检查配置、源码、提交、checkpoint、旧审计和source seals未变化。

## 评估如何判断模块有用

统一报告总体及七类P/R/F1/support、混淆矩阵、Accuracy、Macro-F1、扫散↔再灌注两个方向、相对A和继续训练旧TCN的改对/改错/错改另一种错、来源/时长/录像级聚合、每类对Macro-F1变化的贡献、参数和资源。

同时评估原始候选链及固定 `clinical_cross_record` / `clinical_shuffle` 合成挑战；动态组增加同checkpoint强制g=1、分组随机置换系数和系数分布/方向/跨度诊断。记录实际被置换位置，避免“随机对照”实际上几乎不变化。查看是否保留旧TCN原有纠错，而不只看比A高多少。

动态组需要与**同增强的aug和scalar_aug**比较，才能归因到当前片段驱动的参考。若只优于A、没有优于这些对照，本轮没有证明模块价值。合成挑战只能支持明确扰动下的表现；真实剪辑稳健性、跨医院泛化和论文结论需未参与开发的独立记录。连续门、TCN、时序增强本身已有先例，本代码不预先保证涨点或CVM创新性。

公开提交只含代码、配置、测试和去身份说明；视频、标注、特征、私有计划/审计、逐clip预测、路径日志和权重留在私有服务器。启动器不自动上传或合并main。
