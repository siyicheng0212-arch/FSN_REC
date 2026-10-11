# 已授权下一对照：类别关系感知校准

状态：设计已授权，**未实现、未预检、未排队、未正式训练**。须等Uni视觉-only真实完成及GPU释放再接，不中断当前缓存/训练。

直接基线是已完成自参照ASFormer，不换Uni特征。共享MLP输入5→19：既有五描述符 + 上一stage中心7类softmax + 邻居7类softmax；19→16 ReLU→1，无输出bias，末层零初始化。仍用tanh(s_ij-s_ii)、bound1、self/padding0。理论新增224参数，预计总1097068，尚非实测。

只改变该概率输入：原encoder/三个decoder/窗口/value/损失/梯度detach约定/native孤立路径不变。不能喂真实类别、val混淆、人为类别阈值或录像ID；不硬删异类邻居、不加门控/额外A raw logits/温度搜索。

fresh seed42，A冻结；train7306/val888、16segments/batch、192updates/完整epoch、AdamWlr0.001 wd0.01 clipgrad5；四stageCE+0.15合法邻接平滑cap16；float32禁TF32。最多100轮，原horizon不变；从第1轮Menzhen七类Macro-F1连续10完整epoch无严格新高内置早停，持平未提升，新高重置。

假设：类别分布信息能帮助区分不可靠邻居与合法动作过渡，减少扫散/再灌注与运针相关误报。可能损害：增强错误的高置信类别、稀有类被多数类邻居覆盖。必须报告七类support/precision/recall/F1、完整混淆、改对改错、孤立/有邻居/时长组与资源差异，不只总F1。按完整精度val七类Macro-F1选择，持平优先旧候选。

一次必要CPU/真实三步GPU验收后正式从头训练一次；旧基线不重训。完成需完整history与首次plateau/真实预算、best-last身份、最佳逐片独立回放、所有真实退出0与GPU释放。科学负结果照样公开去身份汇总，不重复重跑赌分。

后续时序研究仍围绕逐类不对称混淆、邻居噪声、真实时长、孤立证据和冻结视觉边界，每候选先登记证据/可证伪假设/单主因素/固定预算，查primary prior art；不把开发val增分当发表保证。
