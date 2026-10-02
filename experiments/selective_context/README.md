# 旧 A＋TCN 的逐片段修正采纳实验

旧 A＋TCN 在已使用的 val823 开发集上取得 79.59% Macro-F1。用相同 checkpoint 切断网络候选连接后为 78.79%，切断临床候选连接后为 78.78%；两次干预救回旧 TCN 的改错分别为 0 和 1，却失去原有纠错 5 和 8。旧模型 15 次把 A 的正确答案改错，其中 5 次发生在孤立片段。因此本实验研究**何时接受旧 TCN 对单片段 A 预测的修正**，不主张错误主要由坏邻居造成。

## 模型

对原始候选链按**已经核实的历史布局**调用旧 TCN；没有新 R、没有额外切链。输入是冻结 A 的视觉特征与七类 logits，旧 TCN 的真实源码、配置和 checkpoint 均须由服务器上的已核实私有资产提供。新门 G 逐片输出一个 0 到 1 之间的权重 alpha：

    new_logits[i] = A_logits[i] + alpha[i] * (old_TCN_logits[i] - A_logits[i])

旧 TCN 及 A 均冻结，只训练 G。若一条旧链本来只有一个片段，依然先调用**旧模型的原有单片段处理**，再由 G 决定采纳程度。强制 alpha=1 直接返回旧 TCN logits，alpha=0 直接返回 A logits；这是无舍入误差的实现检查。训练和推理时 G 不读取标签、来源、边界状态或其它片段的原始输入。

四种门从相同冻结旧 checkpoint 出发，接受相同数据和训练预算：

| 变体 | 学到的采纳比例 | 回答的问题 |
| --- | --- | --- |
| scalar | 所有片段共享一个值 | 简单整体收缩是否足够？ |
| class_conditioned | 按 A 的预测类别选择值 | 类别偏向能否解释收益？ |
| logits_only | 用当前片段的 A 与旧 TCN 七类分数 | 预测冲突本身是否够用？ |
| visual_logits | 再加入当前片段冻结的视觉特征 | 视觉证据是否提供额外价值？ |

必须另报既有 A 和**完全未修改的历史 A＋TCN**，不将它们重新训练后当作原对照。若复杂门不优于标量或按类门，不能声称学到了有用的逐片视觉采纳策略。

## 协议和限制

沿用已封存的 train7372 / val823、原视频分组及冻结 A 证据，不读 test、不改来源协议。训练门使用 train7372 标签，val823 只用于开发期选择 best 和报告，**不是独立测试**。旧 A 与旧 TCN 曾在 train7372 上训练，因此门读取的训练集旧预测是样本内预测；这有堆叠模型的训练/部署分布偏移风险。第一次运行只能提供开发证据；论文最终结论需录制记录级 out-of-fold 门训练输入或未参与开发的独立测试记录。

训练前必须核实历史逐片段文件 823 片的 clip ID、七类真值、冻结 A 预测和**同 checkpoint 原始旧 TCN 预测**逐片一致。历史布局参数 full_chain 或 eligible_segments 取自已完成的真实旧版审计；若是后者，训练和评估都在旧有结构边界处分别调用旧模型，包括孤立片段。不能靠切链后的新输出冒充历史 79.59%。任何缺失旧模型源码、权重、完整审计或原预测的情况均在正式输出创建前停止。

每组汇总 Macro-F1、Accuracy、逐类 P/R/F1/support、相对 A 与原旧 TCN 的改对/改错/错改另一错、扫散↔再灌注双向错分、孤立片段结果、按来源分层、门系数分布、实际更新参数及耗时。五次孤立片段改错和九次再灌注→扫散改错是预先指定的诊断，不应据此用 val823 标签调整损失或阈值。

## 服务器执行

先在独立非 main 工作树核实旧源码、旧 checkpoint、历史逐片段预测和审核配置，安装 CUDA PyTorch 与 pytest。原有的 legacy_template.json **不可直接运行**；必须使用已核实的旧模型 factory 与私有审计文件。默认不加 --execute 只生成只读计划，不创建正式输出：

    python -m experiments.selective_context.revision_run_suite \
      --feature-index "$FSN_INDEX" \
      --source-protocol-dir "$FSN_SOURCE_PROTOCOL" \
      --legacy-config "$FSN_VERIFIED_LEGACY_CONFIG" \
      --legacy-checkpoint "$FSN_OLD_TCN_CHECKPOINT" \
      --historical-predictions "$FSN_OLD_VAL_PREDICTIONS" \
      --chain-layout "$FSN_VERIFIED_OLD_LAYOUT" \
      --config experiments/selective_context/revision_seed42.json \
      --output "$FSN_NEW_PRIVATE_OUTPUT" \
      --python "$FSN_CUDA_PYTHON" --gpus 0,1 \
      --expected-commit "$(git rev-parse HEAD)"

核对计划中的完整路径、SHA、模型严格加载、原预测逐片一致、运行卡映射和新输出后，对**同一命令增加 --execute**，实际依次进行相关单测、真实 CUDA 三步预检、四种门训练及结果核验。两张空闲卡分两波运行、四张空闲卡分一波运行，均是独立实验而非 DDP。每次只用一个全新私有输出目录；失败保留现场，不自动续训、覆盖或调参。

公开分支只保存代码、测试和无身份说明。视频、标注、特征、历史逐片预测、旧审计、权重和原始路径日志留在服务器。
