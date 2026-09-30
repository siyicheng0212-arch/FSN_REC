# CVM 2027：浮针细粒度视频识别的研究问题与冻结实验设计

状态：**方案与实现，尚无本方案的正式训练结果。** 此文件不是录用承诺，也不是模型性能报告。

## 1. 论文到底研究什么

建议工作标题：**Fine-Grained FSN Video Recognition: A Source-Aware Benchmark and an Audit of Clinical Hierarchies**。

中文理解：建立可信的浮针动作识别评估，检验临床分组的作用与边界。保留旧稿的七类标签、视频特征主干与组内分类头。重点从“层次算法天然更好”转为“临床知识何时帮助识别、何时因路由错误或视觉异质性而损害识别”。

七类 ID 固定为：0 消毒、1 进针、2 运针、3 扫散、4 再灌注、5 拔针、6 固定。临床 taxonomy 固定为 `[[0],[6],[1,2,5],[3,4]]`，组大小 `1/1/3/2`。这来自现有任务定义；不假设这些类构成可用的跨 clip 时间序列，不从七类标签推导医者/患者角色、动作并发真值或质量评分。

**预测对象是一个标注片段中的主动作。** 在线视频可能经过剪辑；每个 clip 单独输入，没有跨 clip 状态、邻居、流程位置、来源 ID、时长或真实组标签参与部署时的模型推理。clip 内帧顺序只是缓存保存的顺序；它也可能包含镜头切换，不等于连续临床流程。

## 2. 可以争取哪些贡献，哪些不能写

| 潜在贡献 | 形成贡献需要的证据 | 不能提前宣称 |
|---|---|---|
| 浮针视觉任务与可复现基准 | 清楚的动作定义、边界规范、样本版本、来源隔离、短 clip 采样说明、可共享资源与复现途径 | 仅因“医疗场景首次使用某网络”就有算法创新 |
| 临床 taxonomy 的实证研究 | 临床 vs 随机/视觉分组、CNN 与较新模型、五个歧义动作、跨来源和重复帧分析、完整负结果 | 临床分组总是优于平坦分类，或总分增益就是细动作收益 |
| 路由与表征收益的分离 | 同 checkpoint 硬/软解码、真实路由器、双侧真实组提示、容量对照、辅助监督后仍用 flat 头 | soft routing、增加分类头、CE 分解是新算法 |

层次监督、conditional softmax、粗到细分支、概率融合、置信度回退均有相关工作，见同目录两份文献审查。**本轮不加入另一个未经证实的运动模块，也不把通用方法换名字当创新。** 如果最后只有评估修复，没有足够的新数据资源或可推广发现，方法创新不足仍可能成为拒稿理由；应据结果决定稿件是否值得提交。

## 3. 三个主研究问题与预先约定的反证

**RQ1：临床分组是否比同规模的任意分组更适合视觉识别？**

固定临床组，比较三种预先固定的随机组。主随机对照保留消毒/固定两个 singleton，只重新划分其余五类为三类组与两类组，排除换掉“容易类”造成的混淆。完整七类重分组作为可选敏感性分析；不能用两个随机组表现差就挑出第三个。另用训练集特征均值构建视觉 taxonomy：这是数据驱动对照，不是新聚类方法。只有分组统计与标签来自 train；若特征编码器是 validation 选择的 flat best，必须披露这层开发集使用，不能称编码器完全未见 validation 决策。禁止使用最终 test 来选择编码器。该对照可能改变 singleton，须明确标注。若随机或视觉分组相当/更好，撤销“临床知识独有收益”的主张。

**RQ2：层次模型的收益来自表征学习、细分类器，还是路由决策？**

比较 flat、额外容量、hierarchical auxiliary supervision + flat prediction、旧式层次训练。对层次 checkpoint 同时计算硬路由、最大联合叶概率、真实组提示。对独立训练的 flat checkpoint 也给予相同真实组提示，保证公平。若只有 oracle 层次结果高而真实路由低，应报告路由瓶颈；如果 aux-flat 优于硬路由，说明层次标签可能有监督价值，但硬决策未必合适。若额外容量就解释收益，不能归因临床结构。

**RQ3：观察到的收益能否跨来源成立，是否依赖有效时序证据？**

报告来源、时长、重复帧率、每类结果与录制组 bootstrap。固定 checkpoint 做中间帧重复、clip 内确定性打乱等诊断；它们只能支持“对所采帧/顺序敏感”，不能证明关注手/针或临床机制。短 clip 敏感性分析保留主评估的完整数据，并额外报告 `<0.1s` 排除和重复率切片。若收益只在高重复、固定来源或固定类上出现，限定结论，不改主评估规则去制造涨点。

## 4. 第一优先级：先冻结真正的数据协议

旧稿的 8,181 clip 与近期 8,195 clip 不应当作同一版本。先逐项解释增删原因，记录 clips/records/labels/sampling 版本与文件 SHA；不能把不同版本的分数放到同一主表比较。

现有内部 786 validation 已用于多个新模块的筛选。之前反复使用的 823 validation 也不能自动升级成独立 test。服务器关机或模型换名称不会消除这种使用历史；从已看过的 validation 重新切分也不能产生独立 test。

可执行次序：

1. 从原始录制/重复家族确定 group，合并重上传、同源裁剪或近重复关联；跨 split 不得出现同 group、同源视频或相交片段。仅靠文件 hash 不能排除所有近重复，需要记录 dedupe 审查范围和遗漏风险。
2. 如果患者/操作者/治疗 session 身份可核验，则以关联图连通分量划分。身份缺失不能填假 ID；仅录制组隔离时就只写 recording-disjoint，不能写 patient-/operator-independent。
3. 优先保留此前确实未用于模型选择的原始 test；核验旧稿评估历史。旧 checkpoint 是否见过新评估样本也要核对。不存在未用 test 时，开发实验先只用 train/val；独立评估需新来源/新录制，不得伪造独立性。
4. 新医院/新来源可作为外部测试，但只有在未用于调参且类覆盖/标注口径明确时才成立。未知临床身份、混合公开来源和单个来源规模不足都应如实报告。
5. `python -m cvm.protocol audit` 冻结本地 manifests 和 SHA；列明此前所有已用于选择的 eval manifests。JSON 审查不能自行证明使用历史完整，必须由实验负责人核验。缺失身份/历史的限制不会被代码“修复”。

测试集用于最终锁定方案的批量评估：先完成所有训练/参数/解码规则冻结，再评估预注册的方法列表。允许每个预注册方法报告 test，禁止看到 test 后再选随机 taxonomy、调损失权重或继续开发并仍称其独立。

## 5. 模型与必要对照

所有候选都是共享主干的 clip 分类，重新实现旧稿方法而非承诺逐位复现旧训练。旧稿 checkpoint 的直接审计和新协议重训练的结果分开存放。

| 模式 | 训练监督 | 主输出 | 回答的问题 |
|---|---|---|---|
| `flat` | 七类 CE | flat 七类概率 | 所有比较的基础 |
| `capacity_control` | flat CE + 等权辅助七类 CE | 主 flat 头 | generic nonlinear capacity/额外监督是否足够解释 aux-flat 收益 |
| `aux_flat` | flat CE + group CE + true-group conditional CE | flat 七类概率 | 结构监督能否改善表征，同时保持 flat 的决策能力 |
| `hierarchy` | group CE + true-group conditional CE | 主表预先固定 soft；旧 hard 同 checkpoint 附列 | 旧式层次学习与硬路由造成的损失 |

capacity control 的额外七类头与小型通用非线性适配器，活跃训练参数与 `aux_flat` 的额外层次头精确相等；它匹配容量与额外监督，不声称功能完全相同。纯 hierarchy 的 trainable head 数与 flat 本来不同，需单独报告；不能把 aux 的容量匹配结论套到全部模型。

各损失逐样本求值并按整个 batch 的同一 leaf 权重归一化，**不能对每个组单独均值后相加**，否则同样的监督会暗中改变稀有组权重。主方案 lambda=1，不为层次版做更大搜索。主 best checkpoint 仅从 finetune 阶段选择；warm-up 的验证分数只记历史，不能把 head-only best 当 full finetune 结果。singleton conditional 概率固定为 1，没有可训练的细分类器。

设 group 为 $g(y)$，则：

$$L_H=-\log p(g(y)|x)-\log p(y|g(y),x)=-\log p_H(y|x).$$

$$p_H(y|x)=p(g(y)|x)\,p(y|g(y),x).$$

soft 解码选最大 $p_H(y|x)$；hard 解码先选最大组再在该组内选叶类。两者同 checkpoint 对比，不能把同一训练重复算作两套独立实验。oracle 是非部署诊断，标签不得进入真实推理。

## 6. Backbone 和公平性

实现中的最小集合：R(2+1)D-18（CNN）、MViT-V2-S（Transformer）、VideoMamba-Ti（ECCV 2024，较新架构）。旧稿其他 backbone 结果只有在版本、划分与设置可核实时才能作为既有结果；不可拼接不同协议。

每个 backbone 内各模式必须使用相同 checkpoint、主干初值、输入帧/空间变换、优化器、有效 batch、训练/验证次数和选择指标。允许不同架构使用原生输入：三种架构均为 16 帧，共用同一来源 clip/cache 和相同时间索引；R2+1D 官方权重使用 112×112 crop，MViT/VideoMamba 为 224×224。记录分辨率、帧数、FLOPs/参数/显存/耗时。**同帧数不等于同算力**，因空间分辨率与结构不同，不强行宣称整体预算严格相同。若需要对应旧稿，可预注册 R2+1D 32 帧敏感性分析，不能把它混入本轮 16 帧主表。

优先 Kinetics-400 checkpoint，公开具体 variant、预训练与微调路线、SHA。训练器不把随机初始化冒作预训练，也不在缺少权重时静默回退。VideoMamba 依赖官方版本的定制 CUDA 扩展，必须先真实 GPU 前后向与 checkpoint 完整载入核验；CPU mock 通过不等于它可在服务器训练。

共享 36-frame cache 再均匀选取 16 帧，是本轮明确的缓存协议；它与旧稿直接采 32 帧不保证一致。bin 中心时间不是经核验的 PTS。像素重复率可从缓存核验，不能据此推出真实独立帧数量、光流速度或连续临床动作。

## 7. 冻结配置与运行阶段

建议初始设置：AdamW，backbone lr `1e-5`、head lr `1e-4`、weight decay `.05`；5 轮 head-only warm-up（head lr `1e-3`），然后最多 60 轮 finetune、cosine、patience 10；batch 2 / accumulation 16、有效 batch 32，AMP bf16（GPU 支持时），梯度范数裁剪 5；class weights 主方案 `none`，固定额外敏感性方案为 train-only sqrt-inverse；同一架构所有模式同时统一调整 OOM 设置并创建新协议，不能只改单组。

这些是预注册起点，**不是已验证最佳参数**。先做真实 CUDA one-batch 梯度/显存/载入核验，再测一轮耗时，估算 deadline 前是否能完成。所有超参数改变都属于开发，需新版本；没有“某版跌了就私自改一组 lr”。

阶段 A：只读数据/权重审查、单元测试、真实单 batch smoke；不读取 test 做模型选择。

阶段 B：seed 42 的 R2+1D 四个模式，一轮和完整 dev 运行。目的为实现与可行性核验，不能以一次涨点宣布方法。硬/软/oracle 来自同一 hierarchy checkpoint。

阶段 C：主表冻结后执行 seeds `42/2026/2027`。最小主实验包括三个 backbone 的 flat/hierarchy，R2+1D 的 capacity/aux，以及三套固定 singleton 的随机 hierarchy，共 11 配置 × 3 seeds = 33 次训练。训练视觉 taxonomy 为第 12 配置，资源不足可列预先声明的扩展而不是挑结果。**这不要求自动启动 33 个任务**；suite 默认为生成 dry-run 计划，正式执行需要指定冻结 phase。旧 motion 的取消 seed/wave 永远不恢复。

如果算力不够，先删掉明确标注的扩展，不通过删掉不利 seed/随机分组压缩实验。主要论文结论需要主要对照完成；截至 deadline 只做出 seed42 开发结果，应降低结论或延期，不伪造 multi-seed。

阶段 D：冻结方法列表、模型与 manifest SHA，统一独立 test，生成去身份聚合报告；报告所有正式 seed，不选最好的一个。服务器目前关闭，本提交不会在服务器启动任何任务。

## 8. 指标、归因与统计

主指标为七类 macro-F1。同时报告 accuracy、macro precision/recall、weighted-F1；单标签七类任务中 micro-F1 等于 accuracy。Undefined precision/F1=0，主指标始终含七类。所有表/图统一用 fractions，收益另标百分点，避免旧稿 Gain 错行。

五类歧义动作评价从完整七类预测计算，选择五个类别的 per-class F1 再平均：跨组误判仍保留 FN，误入五类的 singleton 样本仍保留 FP。另列仅筛五类 GT 的 sensitivity，但必须保留它们被预测为 0/6 的错误；禁止删掉其他 logits 再称它为部署五类表现。真正 renormalized 或真实组限制的结果只标 diagnostic/oracle。

真实组路由器报告准确率、macro-F1、四组混淆；flat 的 argmax-leaf 再映射到组，与 flat probabilities 向上求和的 argmax 分别报告，不命名 actual router。未训练的 group/flat 头禁止评估成有效模型。

硬路由的错误可精确分成：

$$P(\hat y\ne y)=P(\hat g\ne g(y))+P(\hat g=g(y),\hat y\ne y).$$

按所有 clip/五类子集分别列出两个计数、分母、conditional fine accuracy 给定路由正确的样本数，以及每个真实组结果。group oracle-flat 与 oracle-hierarchy 都提供真实组，不把 oracle 提升当真实部署收益。

统计分两层：三 seed 的 mean±SD 描述训练随机性；固定 seed 成对预测按原始录制/重复 family group 做 paired cluster bootstrap，描述测试样本不确定性。bootstrap 区间不等于跨医院可靠性，也不代替 seed 重复。所有 comparisons 预先指定，报告 class/source 支持数和缺失类；小来源/类切片不过度解读。不要以 clip IID bootstrap 获得虚假的窄区间。

## 9. 论文中必须有的表和图

1. 数据表：版本变化、独立录制组/来源数、各类与短 clip/重复率统计、身份可核验比例、每 split 交叉审查。
2. 主表：三个 backbone flat/hierarchy soft/hard，各 seed 与 mean±SD、七类与五类 F1、资源。
3. 控制表：clinical/random3/visual、capacity/aux；预注册全部列出。
4. 路由表：实际 coarse、flat mapped/aggregated、两侧 oracle、跨组/组内错误分解。
5. 稳健性表：来源、时长、重复率、static/shuffle 诊断；外部测试存在时单列。
6. 错误图：扫散↔再灌注、进针/运针/拔针等双向错分。图与表由同一聚合 JSON 生成，不手工填第二份数字。

Grad-CAM 可以辅助展示，必须选固定/随机规则的成功与失败样本并如实解释；没有 ROI 或机制干预，不能据少量热图称模型证明了手/针定位。没有额外质量标准或质量标签，本论文不评价操作质量，不推导临床疗效。

## 10. 与之前审稿意见逐项对应

| 旧问题 | 本轮回应 | 仍需真实完成 |
|---|---|---|
| 方法创新不清 | 对照已有 conditional softmax/层次监督，贡献定位 benchmark + taxonomy/routing 实证 | 资源新颖性与实验发现，不能靠文字保证 |
| R2+1D 主要增益来自固定 | singleton/五类分离、完整七类混淆、固定 singleton 随机对照 | 正式预测，不引用旧总分代替 |
| flat 映射组不能替代路由器 | actual router 与 mapped/aggregated 三项分开；双侧 oracle | 分母、训练头和正确组样本支持数 |
| split/复现不足 | group/同源/区间/身份审查、SHA、暴露历史 | 完整来源与近重复核验，必要新独立样本 |
| 基线过时 | CNN + Transformer + VideoMamba2024，checkpoint strict load | 真 CUDA smoke、正式训练与公平设置 |
| 0.06s 如何取32帧 | 缓存重复政策、像素重复统计与排除敏感性 | 实际 cache metadata 不能只写假设 |
| Grad-CAM/应用结论过大 | 诊断与可部署指标分开，删除质量评估推断 | 不用可视化替代证据 |

## 11. 不成立时怎么收束

若 clinical 在随机/flat 前无优势，论文可报告可信负结果与资源，但不能保留“临床 hierarchy 提升五类细粒度识别”这一主要结论。CVM 接收 datasets/benchmark 与视觉识别主题并不意味着本稿创新门槛降低。

若只在反复使用的 validation 或单 seed 改善，称 exploratory result。若所有主干/来源都无稳定发现且数据不能被外部复用，本轮修复可能仍不足以支撑 CVM；停止追加模块，不把最有利 slice 重命名主任务。

## 12. 提交时间与资源发布

官方 [CVM 2027 CFP](https://iccvm.org/2027/callPaper.htm)：abstract 2026-10-23、full 2026-10-26，均为 23:59 GMT，对应北京时间 10-24/10-27 07:59；上限 14 页含参考文献，full/poster 出版路径不同。以上于 2026-10-01 北京时间核验，提交前重新检查官方页面。

本仓库只发布实现、协议模板、文献笔记及去身份聚合。视频、原始标注、患者/操作者映射、逐 clip 预测、原始路径日志、帧缓存与权重均保留私有。benchmark 论文需说明外部读者能够获得什么、如何复现实验与限制；未有授权的视频不会因投稿而自动公开。CVM 匿名稿使用匿名代码入口，不能直接用本实名 GitHub 链接破坏匿名要求。

运行入口和服务器 Codex 指令见 [RUNBOOK.md](RUNBOOK.md) 与 [../../prompts/run_cvm_codex.md](../../prompts/run_cvm_codex.md)。
