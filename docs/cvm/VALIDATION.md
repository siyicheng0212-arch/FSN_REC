# 本次实现验证与尚未完成的实证

验证日期：2026-10-01（北京时间）。本文件记录实现验证，**不报告正式医疗视频训练或独立测试结果**。

已在 Python3.12 / PyTorch2.2.2+CPU / torchvision0.17.2+CPU 执行：

```bash
python -m unittest discover -s tests -p 'test_cvm_*.py' -q
# Ran 99 tests ... OK
python -m compileall -q cvm
bash -n scripts/run_cvm_4gpu.sh
git diff --cached --check
```

99项覆盖：冻结 manifest 的组/来源/已核身份/曝光历史/文件与实现SHA；模型概率分解和未训练头保护；同参数容量控制；四种模式的加权/非加权/末尾不足梯度累积；真实格式 synthetic36f224 cache + native16帧预处理 + injected tiny backbone 的一轮训练/验证/报告；warm-up高分不能成为FullFT best；SIGINT保留last且退出130；选中帧打乱保留集合；完整七类FP/FN的五类F1；actual/mapped/aggregated router；不同taxonomy下双方同组oracle；录制组bootstrap；隐私与不覆盖输出；固定矩阵和dry-run不启动。

专项 VideoMamba 合同测试使用模拟小 encoder，核对固定源码、依赖来源、strict key/shape、非有限参数与输出。不代表实际官方VideoMamba/CUDA模型已可运行。Torchvision实际模型做过CPU形状前向，但没有在本任务下载并实载其正式预训练二进制。

服务器仍需完成：官方checkpoint实际载入与SHA、真实cache及data protocol、CUDA FP32/AMP前后向、各训练路径的非零梯度、显存与每轮耗时、完整训练、独立test资格核验与最终实证。本次没有重启服务器，没有启动任何训练，没有恢复旧seed/wave。

预注册矩阵：pilot = 4个R2+1D模式×seed42；formal = 11配置×seeds42/2026/2027。编写和测试该矩阵不等于已经完成33次实验，也不能保证方法有益或CVM接收。

文献书目分别为 [references.bib](references.bib) 和 [references_hierarchy.bib](references_hierarchy.bib)；实际阅读范围在两份literature笔记中说明。
