# 审稿意见对应的完整实验矩阵

本文件定义要执行的实验，不包含实际医疗训练结果。主问题是：**临床分组在什么视觉表征和数据条件下有效，收益是否确实来自五类细动作，以及硬路由损失有多大。** 网络视频没有跨 clip 顺序，因此所有模型均只输入当前 clip，禁止跨 clip 流程建模。

## 1. 现代视觉模型必须真实训练，不只写进 related work

| 模型 | 本轮作用 | 预训练/输入 | 同主干对比 |
|---|---|---|---|
| R(2+1)D-18 | 旧稿关键 CNN 的新协议参照 | torchvision K400，16×112² | flat / clinical hierarchy |
| MViT-V2-S | 多尺度视频 Transformer | torchvision K400，16×224² | flat / clinical hierarchy |
| VideoMAE ViT-B | 自监督视频表征基线 | 官方 K400 自监督800轮后K400分类微调，16×224² | flat / clinical hierarchy |
| VideoMamba-Ti | ECCV2024状态空间视频模型 | 官方 Ti16 K400权重，16×224²，定制CUDA依赖 | flat / clinical hierarchy |

VideoMAE、MViT-V2与VideoMamba均为已发表方法；接入它们是对比完善，不是本论文的方法创新。VideoMAE V2的已发布大模型/蒸馏/预训练数据条件不同，不能把本实现写成v2，也不以不同数据规模的最高公开分数替代FSN上的实训。旧稿“MViT”若不是V2-S，不得把旧结果当本轮V2结果。

每个主干内部统一初始权重、数据、采样、优化器、有效batch、预算、选模指标。跨主干的预训练任务、空间分辨率、参数和计算成本不同，主表是可迁移性比较，不能声称纯架构因果效应或严格同FLOPs。必须列权重SHA、输入、可训练参数、实际显存、训练时间、推理延迟/吞吐；本地没有测得GPU成本时填“待测”。

**对比表不是把4个现代模型与历史不同划分的Original分数混排。** 旧Uni-AdaFocus的SSv2权重、8/12帧和动态crop另属协议；只有在相同数据角色、可核验历史和明确输入差异下才可另列背景/外部参照。旧稿直接32×224解码也尚非本轮缓存采样的严格复现。

## 2. 主对比与消融逐项回答什么

| ID / 比较 | 执行内容 | 能回答的问题 | 无法支持的推论 |
|---|---|---|---|
| B：视觉主干对比 | 4主干各flat与clinical hierarchy | 临床结构是否仅适合旧CNN，在新表征上能否成立 | 换成新模型本身就是新算法 |
| A1：容量对照 | R2 flat vs capacity_control vs aux_flat | 多加活跃参数/普通辅助监督，是否已解释收益 | 参数匹配等于函数类、梯度或FLOPs完全匹配 |
| A2：训练目标对照 | R2 flat / aux_flat / hierarchy | 层次标签改变表征的作用，与部署路由的作用 | 用aux_flat涨点证明硬路由更好 |
| A3：分组语义对照 | clinical vs固定random17/29/43 | 临床组是否比同尺寸任意组更适合视觉识别 | 只报告最弱随机组或最有利seed |
| A4：视觉分组扩展 | train特征生成visual taxonomy后另训 | 临床组与数据驱动视觉组的差别 | 聚类是新算法；validation选出的编码器完全未参与开发 |
| D1：路由解码 | 同hierarchy checkpoint hard/soft | 早期硬决策造成多少改错/改对 | 各自选不同checkpoint后全归因路由 |
| D2：真实路由 | actual coarse / flat mapped / flat aggregated分开 | 粗分类器本身如何，误差怎样传递 | flat映射准确率等于真实router性能 |
| D3：公平oracle | 独立flat与hierarchy都给予相同真实组 | 组内分类与路由瓶颈的边界 | oracle是可部署性能 |
| S1：采样敏感性 | R2 flat/hierarchy，16 vs32缓存帧 | 收益是否依赖输入帧数 | 重复出32帧获得了32个独立时间观测 |
| S2：类别权重 | R2 flat/hierarchy，none vs train-only sqrt_inverse | 分类不均衡处理是否改变结论 | 看test挑权重再称独立评估 |
| S3：表征适配 | R2 flat/hierarchy，finetune vs frozen官方主干 | 任务表征适配是否必要，分组收益是否依赖它 | frozen官方特征等于旧稿FSN分阶段训练的精确复现 |
| D4：时序诊断 | 固定checkpoint，none/static/shuffle | 所选帧集合与顺序是否影响预测 | 已证明针/手运动定位或临床机制 |
| G：来源隔离 | 指定domain从train/val排除，仅评估原clean test相应domain | 已知domain外推是否成立 | domain隔离等于患者/操作者隔离 |

singleton固定为消毒/固定，三套random仅重新划分其余五类，组大小、分类头尺寸、损失全部相同。视觉组可能改变singleton，单列，不与随机对照混淆。容量对照与aux_flat增加相同数量的活跃参数；`hierarchy`自身参数与flat并不完全相同，不能隐藏这点。

## 3. 预算、阶段与停止规则

固定训练seeds `42/2026/2027`。四卡表示4个独立单卡任务，不是DDP；不同阶段含共享配置，**不要重复训练共享项后挑更好结果**。

| stage | 配置数 × seeds | 用途 |
|---|---:|---|
| smoke | 4×1 =4，R2各mode一轮 | 管线/真实权重/梯度核验 |
| pilot | 4×1 =4，R2各mode正常预算 | 开发可行性；不构成论文结论 |
| benchmark | 8×3 =24 | 四主干flat/hierarchy |
| ablation | 7×3 =21 | R2四mode与3random；包含2个benchmark共享配置 |
| formal | 13×3 =39 | benchmark与ablation去重后的主矩阵 |
| sensitivity | 6×3 =18 | 16/32与none/sqrt_inverse单因素；包含2个主配置 |
| representation | 4×3 =12 | finetune/frozen；包含2个主配置 |
| extended | 19×3 =57 | formal加4个额外敏感性和2个frozen配置，去重 |
| robustness | 已训练主配置各3次推理 | none/static/shuffle；不重训 |

`--taxonomy-file`加入预先冻结visual扩展会多1配置×3seed；来源隔离使用独立派生协议和全新训练，其预算另计，不能把不同协议混成同一seed复制。**主实验是39次，57次是显式附加全部敏感性的上限，不自动启动。** 不从单seed涨点决定删保留哪组；资源不足须事先声明完整主表暂缺哪项，不写已完成。

suite写出配置ID、目的、比较关系和预计训练/推理数量。计划不启动，只有显式`launch --execute`才执行；运行中不改代码、pull、恢复旧motion任务。中止/失败/缺失seed在报告中保留。formal重复配置不能用不同训练设置替代；跨seed结果用mean±SD，不挑最好seed。

## 4. 数据与每张表的指标

先核对8181/8195数据版本、标签、原始录制组、近重复和既往曝光。已有786/823是开发验证。没有未用测试集时只交开发结果，不能承诺完整独立结论。患者/术者ID缺失时只写recording-disjoint。来源隔离工具拒绝将旧train/val移为test，拒绝缺类或历史未核验的test；不自动改选一个“表现更好”的来源。

1. **数据表**：每类clip数、录制组数、来源分布、时长、0.1秒以下数量、选中帧像素重复率、身份/去重审查覆盖；补标签定义、边界规则、许可/可共享形式。
2. **主模型表**：七类macro P/R/F1、accuracy、weighted F1；五类主macro-F1；三seed；资源与预训练。五类F1从完整七类混淆选5类平均，保留singleton→五类FP与五类→singleton FN。
3. **消融表**：A1/A2/A3逐项，逐类F1变化；消毒/固定贡献与五类收益分别列。visual全部登记结果，不根据test构组。
4. **路由表**：actual/router与两个flat映射分开；粗分类macro-F1；跨组错误和正确路由后的组内错误及分母；两侧oracle；同checkpoint hard/soft改对改错。
5. **鲁棒性表**：来源隔离与来源描述切片分开；时长/重复率；完整集主表之外固定排除`<0.1s`的敏感性；static/shuffle属于诊断。来源很少或单类缺失时列support与限制。
6. **不确定性表**：同seed、同主干、同采样的模型对照按整个原始录制group配对bootstrap；诊断对照额外要求同checkpoint SHA，允许的唯一变化为probe。训练seed SD和数据bootstrap CI分别解释；不能用单seed CI代替多seed。

极短片段如何变成输入帧必须如实交代。36缓存再选16/32并未恢复原始视频的真实PTS/帧率，像素相同统计不是连续运动真值。如果旧文“0.06秒动作”是标注边界问题，应回看原视频/标注并记录更正版本，不自动删样本制造提升。

## 5. 论文结论预先收紧

若七类涨而五类退步，不写“增强细粒度动作识别”；若random/容量足以解释收益，不写“临床结构独有贡献”；若soft优于hard，报告路由风险；若现代flat更强，保留负结果并调整适用范围。Grad-CAM只作为示例，不写成普遍机制证据；没有质量评分/专家质量标注，不声称临床质量评估能力。

完善实验能够回应可比性、归因和复现问题，**不能保证新颖性或录用**。当前层次分解、soft routing、辅助监督都是已有方法。论文仍需要可复用的数据资源、可靠的新发现和明确的适用边界，不能靠模型数量替代研究贡献。
