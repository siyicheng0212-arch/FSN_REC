# FSN时序模型：代码与去身份结果快照

这是用于研究复核和网页端分析的独立发布快照；不是私有训练工作树，也不是完整数据集或端到端复现包。结果快照时间：2026-10-12T00:59:25.249444+08:00（北京时间）。

## 已完成结果

同一冻结VideoMAE特征，seed42，train7306 / Menzhen开发val888，完整七类Macro-F1选型；Accuracy来自同一个best checkpoint。全部八个候选的best预测已回放验收，负结果也保留。

| 模型 | Macro-F1（%） | 同checkpoint Accuracy（%） | best轮 | 实际完整轮 | 完整轮优化步 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原TCN | 76.5791 | 75.2252 | 98 | 100 | 19200 |
| 短TCN | 76.5471 | 74.6622 | 9 | 100 | 19200 |
| BiGRU | 76.0016 | 74.0991 | 4 | 100 | 19200 |
| 原MS-TCN++ | 78.0268 | 75.7883 | 16 | 100 | 19200 |
| 原ASFormer | 79.6585 | 77.5901 | 9 | 53 | 10176 |
| 视觉R MS-TCN++ | 77.6447 | 75.7883 | 42 | 85 | 16320 |
| 直接区间偏置ASFormer | 77.7702 | 75.7883 | 15 | 25 | 4800 |
| 自参照有界区间ASFormer | 80.2992 | 77.2523 | 13 | 23 | 4416 |

当前选择：自参照有界区间ASFormer。相对原ASFormer，Macro-F1提升0.640694个百分点，Accuracy下降0.337838个百分点。正确片段数689→686，改对38、改错41、错误变成另一错误14；不是所有类别都提升。

## 从哪里开始看

- [逐类结果与分析](reports/RESULTS.md)：每类support、precision、recall、F1、混淆与已知限制。
- [完整精度结果JSON](reports/results.json)：八模型的七类指标和7×7混淆矩阵。
- [完整逐轮曲线](reports/epoch_curves.json)：epoch / 总训练目标 / val F1 / val Accuracy / **当轮**优化步。
- [上下文与分组诊断](reports/context_diagnostics.json)：原AS与自参照AS的孤立/有邻居、邻居挑战、时长及去身份录像聚合。
- [实际预算与实验状态](reports/experiment_registry.json)：真实轮数、早停方式、源代码commit和未训练候选。
- [下个已授权对照](docs/NEXT_EXPERIMENT.md)：类别关系感知校准，仅设计，尚未实现/开训。

核心代码在 `experiments/hospital_asformer_anchor/model.py`、`hospital_asformer_interval/model.py`、`hospital_temporal_compare/model.py` 和vendor目录。源代码摘要与发布变换见 [provenance](reports/source_provenance.json)。模型里有历史预启动状态helper；那不是实验实时状态，完成结果以reports为准。TCN文件含明确标记为未训练的旧研究分支，不要把代码存在当成结果。

## 当前最好模型做了什么

冻结A输出每片1536维视觉特征（mean768+max768）及7维raw logits，共1543维。ASFormer保留原encoder、三个各10层decoder及分类/损失。只在合法decoder注意力窗口加共享MLP偏置：5个成对描述符→16 ReLU→1，无输出bias；修正为 `tanh(s_ij - s_ii)`，固定界限1、self/padding修正0、末层零初始化。新增112参数，总1096844。

描述符是真实中点时间差、时长比、上一stage隐藏特征cosine、两个片段的预测熵。允许增强或抑制邻居，不保证增分。孤立片段保持native单片段路径，不造邻居，不自动退回A，不跨录像或排除缺口。详情及合成验收见模型和测试。

## 本地阅读与合成测试

Python3.10+；依赖在requirements.txt。本快照合成验收环境为Python3.12.7 / PyTorch2.2.2 / NumPy1.26.4 / pytest9.1.1（CPU），服务器训练验收不在该公开测试命令中重跑。仅CPU合成测试，不读取任何真实数据，也不会启动训练：

```bash
python -m pip install -r requirements.txt
python -m tools.check_public_release
python -m pytest -q
```

Uni-AdaFocus适配代码也提供，但原版外部源码和官方权重不在本快照内，不能直接端到端运行。Uni截至快照仍在准备其独立8+36帧新输入，**没有正式训练结果**。VideoMAE的16帧缓存不能当作Uni的44帧输入。私有缓存、划分/排除证明、worker/run/completion调度代码未公开；本快照测试不是全部私有管线验收的替代。

## 给网页端ChatGPT的分析入口

请提供这个**分支**的README链接（不要只给仓库首页），让它再读取reports/results.json、epoch_curves.json、context_diagnostics.json和核心model.py；不能访问时，可直接上传这些公开文本文件。建议提问：

> 依据七类support/precision/recall/F1、混淆及完整训练曲线，比较原ASFormer、自参照ASFormer和负结果。区分事实与机制假设，考虑孤立片段、真实时长和邻居噪声；不要把开发val选型当独立测试，也不要把推理扰动当重训因果消融。评估下一个类别关系校准对照及论文证据缺项。

## 解释边界和隐私范围

Menzhen是反复用于选择checkpoint/结构的**开发验证集，不是独立测试集**；未核验患者独立。只有seed42，容量、损失、预训练/输入及实际早停预算不同，不是等预算架构因果。新增模块的论文新颖性、稳定泛化和发表可能性尚未由这些结果证明。

本分支只含代码、合成测试与去身份聚合。没有视频、逐片标注/预测、清单/排除行、患者或录像标识、原日志、缓存数组、模型权重、服务器路径/账号/设备标识。仅新增非main分支，不合并main。MIT上游代码与许可见THIRD_PARTY_NOTICES.md。
