# 给服务器 Codex 的任务

请在 FSN_REC 的 `codex/fsn-tcn-context-utility` 分支固定提交上准备并执行**已获授权的一轮seed42上下文参考实验**。先核实实际分支/提交和README，不在训练过程中pull。保留现有未提交工作、历史训练、旧失败现场和全部数据。

## 先识别资产，默认只读计划

找到完整train7372/val823对应的封存source protocol和冻结A feature index。逐文件SHA和协议必须一致，val823是开发验证集，不读test、不重新划分、不重新导出A。

分别识别两个基座：

1. 新参考版：同一冻结证据、同一结构、`strategy=all` 的已完成全接TCN `best.pt`。使用 `experiments/context_utility/reference.json`，不把历史79.20%写成新实验结果。
2. 旧版：寻找实际 `tcn_baseline.py`、原训练配置、79.59%报告和checkpoint。核实后写真实factory及**私有**legacy审计，填完整 `legacy_template.json` 字段，特别是A预测输入形式、归一化、卷积路径/统计、单片段策略、原损失/优化器、严格checkpoint加载。不能猜三个卷积块、不能用reference替代旧版。旧资产缺失就报告缺失，新参考版可独立继续完成。若旧版用本实现未支持的loss/padding/布局等，报告阻断并保留原模型，不静默修改。

真实legacy factory/相关新增代码应先完成测试、提交到独立分支；audit、数据和权重不入git。固定新版提交后才正式启动，禁止边跑边改。

检查环境Python确实有torch CUDA、numpy、pytest等依赖，当前可用物理GPU是空闲RTX3090；不干扰已有进程。两张卡用两波，四张卡四组同波。每组内部cuda:0由单GPU UUID映射决定，不是DDP。

对每个可用基座分别先运行：

```bash
"$FSN_UTILITY_PYTHON" -m experiments.context_utility.run_suite \
  --feature-index "$FSN_UTILITY_INDEX" --source-protocol-dir "$FSN_UTILITY_PROTOCOL" \
  --config "$FSN_UTILITY_CONFIG" --base-checkpoint "$FSN_UTILITY_BASE" \
  --output "$FSN_UTILITY_OUTPUT" --gpus 0,1 \
  --python "$FSN_UTILITY_PYTHON" --expected-commit "$FSN_UTILITY_COMMIT"
```

四卡改 `--gpus 0,1,2,3`。把变量替换为真实绝对路径；reference/legacy分别独立配置、checkpoint、全新仓库外输出，不能混合比较。只读计划不探GPU、不建输出。

## 执行已授权的本轮

所有计划和资产核验通过后，同样命令加 `--execute`。启动器先单测，再真实冻结特征CUDA三步smoke，然后：

- GPU0 continue、GPU1 aug 完成后，才GPU0 scalar_aug、GPU1 dynamic_aug；或四GPU同时各一组。
- A冻结；所有组同一基座checkpoint和新优化器状态、相同正常/anchor视图数量。后三组替换计划完全一致。
- 四组均exit0且完整result/best/history核实后，统一评估。

若reference和legacy均具备，按基座顺序运行，第二套前重新核实卡空闲；一套失败停止其后续波次，不自动改协议、重启或覆盖。不得恢复旧extra_wave、追加种子、临时调整阈值或batch、读取test、杀旧任务。OOM/NaN/缺梯度等保存现场报告；已经在跑的同波任务自然结束。

## 交付真实结果

只读汇总每套四组的Macro-F1、Accuracy、逐类P/R/F1/support、最佳轮次、实际步数/耗时/资源、新增及训练参数、改对/改错、扫散↔再灌注双向错误、source/duration/录像聚合。重点比较dynamic_aug与同增强的aug及scalar_aug，而不只与A比。

检查动态系数分布、左右/跨度、同checkpoint强制g=1、分组随机置换实际变动数，以及旧TCN原有纠错保留率。固定合成挑战只称合成稳健性，val823只称开发结果，单seed不写论文定论。不把预检/启动成功写成训练完成。

记录提交、输入SHA、GPU映射、pids/退出码和输出目录。去身份聚合报告可提交独立非main分支；视频、标注、特征、私有计划/审计、逐clip预测、原路径日志和权重不上传，不合并main。本轮完成并交付后暂停，不自行追加实验。
