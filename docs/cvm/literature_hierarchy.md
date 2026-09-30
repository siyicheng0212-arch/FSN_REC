# 层次判别、路由与归因：文献及实验边界

核查日期：2026-10-01。本文档服务于 FSN 单 clip 七分类的 CVM 实验设计；不宣称实验已完成，也不宣称以下组合天然具备算法创新。本轮冻结为 benchmark、临床 taxonomy 与路由的实证研究，不新增运动或组专属特征模块。下文特征机制、校准及 fallback 是文献边界与未来选项，不属于本轮新增算法或训练计划。

## 1. 原稿实际做了什么

已阅读当前原稿《Procedure-Aware Coarse-to-Fine Framework for Fu’s Subcutaneous Needling Action Recognition》的方法、损失和实验部分。原稿使用共享视频特征、四类 coarse 头、三个针操作的细分类头及扫散／再灌注二分类头，推理采用先选组再选组内类的硬路由。四组大小是 1、1、3、2。原稿 Eq. (8) 的 coarse CE 与真实组内 CE 等权相加。

设 `p_c(g|x)` 是组概率，`p_f(y|g,x)` 是组内概率；单元素组的条件概率为 1。联合叶类概率是：

\[
q(y|x)=p_c(g(y)|x)p_f(y|g(y),x),\qquad \sum_y q(y|x)=1.
\]

则原稿等权目标就是 `-log q(y|x)`。这说明将硬路由换成最大联合叶概率，可以是同一 checkpoint 的概率解码修复，不要求把训练目标包装成新的层次损失。若损失带类权重、组内平均或各项权重，其实际等价性必须依据代码再核对。

原稿使用 8,181 片段、32 帧输入及原有 8:1:1 协议；近期 AdaFocus 的 7,372/823 或内部 6,586/786 是另一套协议。不可合并排名，也不能默认旧 checkpoint 没见过新的内部验证样本。

## 2. 八篇原始文献及其约束

下表记录的是实际阅读范围，避免把读过摘要写成通读全文。所有方法描述来自论文或作者来源；“FSN 启示”是本项目的推论。

| 文献与原始来源 | 已阅读内容 | 与 FSN 方案的关系 |
|---|---|---|
| Yan et al., **HD-CNN: Hierarchical Deep Convolutional Neural Network for Large Scale Visual Recognition**, ICCV 2015。 [论文](https://arxiv.org/abs/1410.0736) / [PDF](https://arxiv.org/pdf/1410.0736) | 摘要、引言、架构 §3：共享底层、粗类和专属细类组件、概率融合、条件执行。 | “先分容易的粗组，再专门处理混淆类”及概率加权已有直接先例；临床组替代类别组本身不足以成为新架构。 |
| Bertinetto et al., **Making Better Mistakes: Leveraging Class Hierarchies with Deep Networks**, CVPR 2020。 [论文](https://arxiv.org/abs/1912.09393) / [PDF](https://arxiv.org/pdf/1912.09393) | §3 HXE／层次软标签，§4 top-1 与层次距离的取舍，以及随机层次实验。 | 层次损失和语义软标签已有先例；更合理的层次错误不等于更高七分类 F1。必须同时报告叶类与组层面结果。 |
| Valmadre, **Hierarchical Classification at Multiple Operating Points**, NeurIPS 2022。 [会议页](https://proceedings.neurips.cc/paper_files/paper/2022/hash/727855c31df8821fd18d41c23daebf10-Abstract.html) / [PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/727855c31df8821fd18d41c23daebf10-Paper-Conference.pdf) | §2–3 问题／指标、§6.1–6.3 flat 与 conditional softmax、粗类学习和未见类实验。 | flat 是强对照，top-down 条件 softmax 未必占优；粗组内部可能高度异质。该文是反驳“有层次就更好”的最近核心依据。 |
| Shao et al., **FineGym: A Hierarchical Video Dataset for Fine-Grained Action Understanding**, CVPR 2020。 [论文](https://arxiv.org/abs/2004.06704) / [PDF](https://arxiv.org/pdf/2004.06704) | 三层 event／set／element 标注，Gym99／Gym288 设置，§4 的帧数量、帧打乱与时序诊断。 | 层次细粒度视频任务已有成熟先例；可借鉴 clip 内时间扰动测试，但不能借用体操完整动作顺序为 FSN 跨 clip 顺序作假设。 |
| Guo et al., **On Calibration of Modern Neural Networks**, ICML 2017。 [会议页](https://proceedings.mlr.press/v70/guo17a.html) / [PDF](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) | 摘要、温度缩放方法与校准实验／实现部分。 | 温度缩放是已有基线，不是本项目创新。一个正标量温度不会改变单头的 argmax；它可能改变多头联合概率及门控，因此需要单独归因。 |
| Goren, Galil & El-Yaniv, **Hierarchical Selective Classification**, arXiv 2405.11533，v2 为 2025-01-06。 [论文](https://arxiv.org/abs/2405.11533) / [PDF](https://arxiv.org/pdf/2405.11533) | 摘要、§2 风险／覆盖定义、§3 inference rules：叶概率校准后求父节点概率，不确定时退到粗节点。 | “低置信度时退到粗类”已有非常接近的研究；该工作改变输出精细程度，不能用粗类正确直接提高 FSN 必须输出七类的主指标。此处按预印本版本引用，不臆测会议信息。 |
| Choi et al., **Why Can’t I Dance in the Mall? Learning to Mitigate Scene Bias in Action Recognition**, NeurIPS 2019。 [论文](https://arxiv.org/abs/1912.05534) / [PDF](https://arxiv.org/pdf/1912.05534) | §3 场景对抗损失、人遮挡熵损失与现成 detector 生成遮挡区域。 | 背景／来源捷径是动作识别的已知问题。FSN 身体上下文可能有用，不能照搬“所有背景都应去除”；需区分真实辅助证据与来源标记。 |
| Adebayo et al., **Sanity Checks for Saliency Maps**, NeurIPS 2018。 [会议页](https://papers.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html) / [PDF](https://proceedings.neurips.cc/paper_files/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf) | 模型参数／标签随机化检查和 GradCAM、Guided GradCAM 的区别。 | 少量好看的热图不能证明机制。论文中的 GradCAM 通过其检查，Guided GradCAM 未通过；不能笼统写成“所有 GradCAM 都不可信”。 |

补充检索到 Karthik et al. 的 ICLR 2021 **No Cost Likelihood Manipulation at Test Time for Making Better Mistakes in Deep Networks**；OpenReview 本轮访问被验证页阻挡。作者 [CRM 代码说明](https://github.com/sgk98/CRM-Better-Mistakes)确认其采用 Conditional Risk Minimization。应作为后续更广泛的层次风险对照候选，但本轮不声称已读其论文全文。

## 3. 可以检验的研究问题

### RQ1：临床分组提升了细动作识别，还是只让单元素组更容易？

主结果保留七类 Macro-F1、accuracy 与逐类指标。额外报告三个针操作和扫散／再灌注这五类的 F1。五类 F1 应先在**完整七类预测**上逐类计算，再对这五类取平均；不能删掉另外两类的预测与样本后重算，从而隐去跨组误判。

必须对照同一主干、同样优化预算的 flat、临床层次、同大小随机分组。主随机对照保留 ID 0 消毒和 ID 6 固定两个 singleton，只重新划分其余五类，保持组大小 1/1/3/2、组件容量和计算预算。七类全随机仅作明确预注册的敏感性扩展。随机种子或分组清单提前固定，不能根据验证集选“最差随机组”。训练特征均值可以作为视觉分组来源；若采用混淆矩阵建组，则须来自训练集内的 out-of-fold 预测。两者都禁止用最终评估集建组，且不可混称一种构建方法。

拒绝条件：只有消毒／固定上涨；临床组不优于随机分组；或收益可由同容量通用额外头解释。出现这些结果时不支持“临床知识增强细粒度识别”。

### RQ2：硬路由造成多大损失，软预测实际恢复了什么？

同一个层次 checkpoint 同时评估硬路由和联合叶概率 `argmax_y q(y|x)`，避免把重新训练收益归给解码。直接测 coarse 头准确率、coarse Macro-F1、route error、`P(final correct | route correct)`、正确组内仍误判的数量。

flat 的叶预测映射到组、flat 叶概率按组求和，是有意义的对照，但不是层次模型真实 coarse 头性能；三者应分别命名。

真实组提示仅是诊断：层次模型用真实组细分类；flat 模型也获得同样提示，在真实组内屏蔽其他类并重新归一化。比较双方对三类针操作及扫散／再灌注的结果，不能拿层次 oracle 与无提示 flat 作部署性能比较。

拒绝条件：软预测只能恢复跨组错误，却未改善组内错分；或同权重软预测几乎不改预测。这些结果只能支持解码修复，不能支持新的视觉表征机制。

### 未来可选问题：新增特征机制是否提供七分类主干缺少的信息？（本轮不执行）

只有 RQ1/RQ2 和实际错例提供了稳定瓶颈后，才评估组专属特征残差候选。它应只使用当前 clip，不新增框，不预设医者／患者角色，也不能依赖源视频前后 clip。

候选至少对照：基础 flat；仅增加同容量通用残差；仅分组头；组专属残差配临床 taxonomy；组专属残差配预固定随机 taxonomy。若保留 flat 分支与层次分支的混合，还需固定混合与自适应混合对照。以上组合仍接近已有专家／层次模型，只能称候选，不能预先声称结构首创。

证明要求：完整七类结果、五类歧义结果、实际模块梯度／残差、同 checkpoint 开关分支改对／改错，以及独立训练 baseline。模块开关是贡献审计，不可冒充独立 baseline。

拒绝条件：分支几乎不改预测；改错多于改对；同容量通用残差等效；或只在反复选模型的开发集与一个训练 seed 上上涨。

### RQ3：增益是否来自来源和采样捷径？

划分必须隔离患者／原始视频；网络转载、剪辑和近重复视频应按可识别的源关联一起分组。无法核验患者身份时，明确这是 video-source 隔离，不能改称 patient-independent。

保留来源、真实时长、可解码帧数、采样后唯一帧数／重复比例等切片。超短 clip 的 32 帧由复制或补帧获得，应记录，而非暗示含 32 个独立时刻。clip 内帧打乱／反转是敏感性诊断；七类中方向相关类别可能被反转改语义，所以不能把反转后的原标签当成新的准确率金标准。

不新增人工框时，固定网格遮挡可作压力测试，只能称区域扰动，不能称“背景去除／动作区遮挡”。若使用现成 detector 或自动操作区，其检测误差、保留的身体线索与额外算力必须披露。热图作为示例，机制主证据使用预固定扰动和分支审计。

## 4. 校准和 fallback 的公平界限

所有温度、阈值、融合系数和候选选择只用训练内 validation／calibration，最终评估集保持封存。若数据量有限，可在训练集内按 source 做交叉拟合，并明确校准与选 best 是否复用了同一批样本。

概率混合候选

\[
p(y|x)=(1-a(x))p_{flat}(y|x)+a(x)q(y|x)
\]

是成熟的分布混合形式，本身不能视作创新。需要基础 flat、层次联合概率、校准后的各自预测、固定混合和自适应混合对照。新增 flat 头也增加容量，需报告参数差与同容量控制。

七分类主任务必须始终输出叶类。允许拒识或输出粗类的附加实验另报风险—覆盖曲线、细类覆盖率、各类覆盖率及剩余样本构成；不能只展示保留样本的准确率，也不能将 coarse correct 计入七类 Macro-F1。树距离也不是经临床验证的风险／严重程度，若要作临床代价矩阵须另有临床依据。

## 5. 论文主张与停止规则

论文有三种可能的合法结果，先不把它们写死：

1. 临床分组与特征机制通过公平对照，且在未用于开发的来源上改善歧义动作：可以围绕证据支持的机制形成方法贡献；仍需检查与现有层次专家方法的具体区别。
2. 只有软解码改善，专属机制未胜过通用控制：可以报告路由修复和任务评估，不声称新视觉机制。
3. 层次及专属机制均无稳定收益：停止算法涨点主张，以任务／数据协议和分组适用边界为主要研究发现；能否达到目标会议要求取决于数据价值、评估完整性和审稿判断，无法保证。

近期局部匹配四组是单 seed、内部验证、部分人工中止。它支持停止继续扩大该轮训练，不证明所有局部运动方法普遍无效。不得写成已完成多种子独立测试，也不得将启动／CUDA smoke test 成功写成模型成功。

本轮只实施冻结设计中的基础 flat、活跃参数匹配的通用辅助七类容量对照、层次辅助监督后 flat 输出、原层次训练及其硬／软解码；不追加组专属特征候选、置信度门控或另一种运动模块。容量对照与 aux-flat 匹配额外训练参数，不把这种匹配推广为纯 hierarchy 与 flat 的完全同构。不使用类别名字推导医者／患者角色、证据可见性或动作并发真值。未来若另有独立证据支持特征改动，应单独预注册新研究；若上述因素成为核心研究问题，需要额外人工复核，不能以“无需人工框”掩盖新增标注工作。
