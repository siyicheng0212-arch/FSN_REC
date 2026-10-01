# 浮针 A–R–D：来源自动监督的完整实验

这是按用户已确认的数据条件实现的新版本：临床连续拍摄允许使用流程，网络视频不允许使用流程。利用清单的已有来源字段自动生成 C/D，不新增逐对人工连接标注。**这些是来源政策标签；网络 D 不代表逐对证实存在剪辑。** 原有七类动作标签保留。

代码包含数据审核、连接生成、Original 特征导出、R 训练、流程转移统计、五组配对评测、真实 CUDA 预检与统一 launcher。目前本地只有 CPU 测试；没有服务器 CUDA 成功报告、真实训练结果或涨点结论。

## 数据怎样划分

**完全保留 train7372 / val823，不再划内部验证。** 不改清单、缓存、动作标签，不读取额外 test。

| 用途 | 数据 | 处理 |
|---|---|---|
| A 视觉识别器 | 已完成的 full7372/823 Original `best.pt` | 严格加载七分类权重并全部冻结，不重新训练 |
| R 关系训练 | 两端都属于 train7372 的有效候选片段对 | 临床=1、网络=0；不把 val 放进训练 |
| D 转移统计 | train7372 内有效临床前后连接及其七类标签 | 网络边不进入转移计数 |
| 选 R checkpoint | 两端都属于 val823 的有效候选对 | 用未加权 C/D BCE；阈值本轮预先固定0.9 |
| 五组动作评测 | 全部 val823 clip，每个出现一次 | 包括单片段、断链片段；报告全部七类与来源切片 |

7372/823 是 clip 数量，**R 的训练样本数是满足条件的片段对数**，由数据审计报告实际生成，不能预先说有7372条边。

固定原始清单 SHA：

- train：`093d0adf08dc1d382c0d2e1863a2f4724cb24ebd5b3d0af070e0c76ca989f1d3`
- val：`ad785ec18b14b63616582a143384fac649e69e27d142dfcf87c6852f9d3c6f02`

代码拒绝数量、SHA 不符的清单。两端不得跨划分；group、原视频路径、存在的 record_id 也不得跨 train/val。当前分组独立性不等于患者独立，患者/转载重复身份仍需核实。823曾用于选择模型，是开发验证，不是独立测试。

## 自动标签怎样生成

`source_map.json` 使用仓库 `data_tools/build_dataset.py` 已有来源约定：

| source_collection | 来源政策 |
|---|---|
| lishui、menzhen | clinical |
| bilibili、dy、ks、youtube | network |

服务器先运行 inventory 核对真实清单。未知来源报错，不凭目录名字猜。无需改原清单；映射表单独固定并记录 SHA。

`source_edges.py` 在同 group、同原视频、同时间基准、同 record_id 内按真实标注的起止时间排序，只考虑相邻两条。默认条件是：非重叠、起点没有歧义、间隔≤5秒。满足时临床记 C，网络记 D，写明 `label_origin=source_rule_v1`。不同临床操作不会连接，也不随机制造负例。

重叠、重复起点、间隔过大时断开；不跳过中间片段后补一条线。所有 clip 保留在评测链中，只有边关闭。没有候选顺序的单 clip 独立分类。

**5秒是本轮固定工程上限，并非临床上已验证的最佳间隔。** 当前 clip 主要由动作标注事件生成，下一条标注不一定是紧接发生的动作，未标注过程可能形成间隔。网络时间表示编辑后的播放位置，不能据此恢复临床顺序；缓存 token 位置也不是已核验帧 PTS。

## 模型与五组对照

- **A**：复用 Original 的36张输入、8张全局、12张局部及128 crop。临时 hook 读取全局1280维与局部2048维特征，不改 forward。严格加载 finetuned Original；官方SSv2初始化权重不能替代它。
- **R-MLP**：两侧全局/局部特征分别压到64维，均值池化后拼接左右、绝对差和逐元素乘积；默认291,521个可训练参数。
- **R-Dual**：相同场景配对，再读前片段末两个、后片段首两个已采样全局 token，以侧别/位置嵌入和一层Transformer融合；默认391,297个可训练参数。这不是精准原视频边界窗口，可能遗漏剪辑画面。
- **D**：hard门控默认q≥0.9接通，起始先验均匀。连接因子为 `lambda * log((1-r)+7*r*T)`。断开处完全中性；全部断开或单 clip 时精确回到同一个 A 的 argmax。

R 只接收冻结画面特征及缓存位置。来源、路径、clip ID、七类标签和七类 logits 不进入 R。来源只用于生成训练目标及规则对照。

所有对照共用**同一个 A、同一套 logits、同一张临床训练转移表、同一组结构候选边**：

| 评测名称 | 连接方式 | 目的 |
|---|---|---|
| A_only | 全部关闭 | 单clip Original |
| all_candidate | 所有有效候选边接通 | 检查忽略来源的流程使用 |
| source_rule | 临床接通、网络关闭 | 直接使用已知来源的必要基线 |
| learned_mlp | MLP预测达到阈值才接通 | 简单学习对照 |
| learned_dual | Dual预测达到阈值才接通 | 检验新增时序分支价值 |

`all_candidate` 使用新的临床限定转移表和均匀起始项，**不是复现旧完整v3，也不能继承旧v3分数**。先验清洗、来源规则和学习R的贡献需分别比较。

本轮R：seed42，AdamW lr0.001、wd0.01、batch16、最多30epoch/patience5；train正例权重=N_D/N_C，仅由训练边统计；val BCE不加权。MLP/Dual相同设置。0.9未校准，尤其加权BCE下不能解释成90%临床连续概率；不自动搜阈值、追加seed或改batch。

## 最简单的运行方式

使用已经能运行 Original 的CUDA Python。四张3090服务器可以运行本实验，但**这里只训练两个R任务，两张卡足够**：先GPU0做真实缓存预检/导出，再GPU0训练MLP、GPU1训练Dual。A不重训，不是四卡DDP；其他卡无需为凑满而启动旧实验。

先只读核对来源（所有路径均为服务器本地占位符）：

```bash
python -m experiments.relation.source_edges --manifest-dir /path/to/full_manifests --inventory
```

统一launcher默认只生成计划，不启动训练：

```bash
python -m experiments.relation.run_suite \
  --manifest-dir /path/to/full_manifests \
  --cache-root /path/to/full_cache_36f224_trainval \
  --original-checkpoint /path/to/completed_original/seed_42/best.pt \
  --output /path/to/fresh_source_relation_seed42_v1 \
  --gpus 0,1
```

同样命令加 `--execute` 才依次完成：

1. 创建来源标签/训练与验证链，固定协议及SHA。
2. 真实缓存CUDA smoke：冻结A的logit一致性、两个R三步梯度、D断链回退。
3. 导出全8195个clip的A证据；固定36/8/12/128。
4. 只统计训练临床C边的七类转移。
5. 两张卡并行训练MLP和Dual，检查退出码及best/metrics。
6. 对完整val823做五组配对评测，再标记suite完成。

命令支持 `--expected-commit <固定完整SHA>`。执行要求干净独立worktree、全新且位于worktree外的输出目录、所选GPU无已有计算进程；失败保留现场，不自动重启或覆盖。运行过程中禁止git pull、改代码/环境/数据。代码SHA、清单、来源映射、A checkpoint、每个特征文件与协议产物均记录/检查。预检失败不会继续导出或训练；一组R失败时另一组可完成，但不汇报整轮成功。

正式A加载还检查checkpoint中保存的full7372/823清单审计和finetune轮次，拒绝内部划分、warmup或缺少来源记录的权重。每个阶段前复查代码与外部输入SHA；导出后再次核对与smoke一致。服务器若设置了非恒等`CUDA_VISIBLE_DEVICES`，先报告并核实；launcher拒绝含糊映射，并记录物理GPU UUID、固定子进程的CUDA顺序。缓存本身没有逐字节哈希清单，不能宣称所有缓存内容都已校验。

`SERVER_PROMPT.md` 是可直接给服务器Codex的完整操作说明。先运行：

```bash
python -m unittest discover -s tests -p 'test_relation*.py'
```

## 单独运行各阶段

也可按顺序调用 `source_edges`、`smoke`、`export`、`fit_transition`、`train`、`evaluate_suite`。用各模块 `--help` 查看全部参数。来源训练/拟合必须传 `--source-protocol-dir`；它检查完整7372/823、产物SHA、与特征index一致，不能把来源标签悄悄当成人工逐对真值。

`infer.py` 保留为低层显式链接口；正式来源实验应使用 `evaluate_suite.py`，获得来源规则对照和全val覆盖检查。原先手工C/D/U输入仍可单独用底层train，但不得与本轮来源标签混成同一实验。

## 输出与论文解释

私有输出：`run_protocol.json`、`.launch_once`、`run_commit.txt`、`cuda_smoke.json`、`suite_result.json`；`launcher_logs/`记录各阶段日志、PID和退出码；`source_protocol/`保存来源标签、候选链、元数据和审计；`features/`保存冻结证据；`mlp/dual`保存best、history、metrics、protocol；`evaluation/`保存逐clip预测和配对报告。

只有 `evaluation/public_safe_summary.json` 是去身份聚合：七类P/R/F1/support、Macro-F1、accuracy、混淆矩阵、来源/时长切片、扫散↔再灌注双向错分及归一化率、改对/改错、网络门控激活率，以及R在来源政策标签上的表现。视频、原标注、特征、权重、逐clip预测、含路径日志和完整协议留在服务器，不上传GitHub。

**本轮能检验的是来源政策监督下R是否有额外价值。** 若只超过全连接、没有超过来源规则，不能据此宣称学会了通用临床连续性或创新成立。单seed开发验证涨点也不是独立泛化结论；新医院/新网络源测试及真实边界证据仍需后续补充。
