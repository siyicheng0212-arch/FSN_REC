# 给服务器 Codex：来源自动监督的 A–R–D

请实际预检并运行仓库 https://github.com/siyicheng0212-arch/FSN_REC 的分支 `codex/fsn-source-relation`。
用户已确认临床拍摄允许流程、网络视频不允许流程；利用已有来源字段自动生成监督，**不要求新增逐对人工标注**。标签是source_rule_v1，不能写成人工核验连续性真值。

这次保留完整train7372/val823，不恢复内部划分，不重新划分，不读取额外test，不重训A，不启动旧local/context四卡任务或多seed计划。先阅读 `experiments/relation/README.md` 和 `run_suite.py`。

1. 只读检查现有GPU/训练进程/磁盘和可用CUDA Python；已有任务不停止、不续训、不重复启动。检查CUDA_VISIBLE_DEVICES；非全卡恒等映射会被launcher拒绝，核实物理GPU编号后在本任务shell中清除该变量，不改其他任务环境。launcher会记录UUID并固定子进程的物理GPU与CUDA编号对应。新代码使用独立干净worktree，主checkout、未跟踪备份和processed_server不动。固定该分支的实际完整commit SHA；执行命令使用`--expected-commit`校验。
2. 确认本地路径，使用任务专用环境变量：
   - FSN_PYTHON：已有CUDA Python。
   - FSN_WORKTREE：新独立worktree。
   - FSN_MANIFEST_DIR：历史full7372/823原清单目录。
   - FSN_CACHE_DIR：现成36f224缓存。
   - FSN_ORIGINAL_BEST：**已完整训练完成Original的七类best.pt**，不是官方SSv2初始化，不是失败局部模型/内部划分checkpoint。检查其原run_config/result/split_audit和训练清单SHA，确认来自full7372/823。代码还会检查checkpoint中保存的split_audit和finetune轮次；缺失时寻找真实对应权重，不补造审计字段。
   - FSN_OUTPUT_DIR：全新服务器输出，位于worktree之外。
   - FSN_EXPECTED_COMMIT：固定完整SHA。
3. 核对train SHA `093d0adf08dc1d382c0d2e1863a2f4724cb24ebd5b3d0af070e0c76ca989f1d3`；val SHA `ad785ec18b14b63616582a143384fac649e69e27d142dfcf87c6852f9d3c6f02`。任何缺失/不符就报告，不重新生成清单或把旧内部分割拼成full。7372/823是clip数，不能当R边数。
4. 运行所有 `test_relation*.py`。先跑来源inventory，核对lishui/menzhen为临床，bilibili/dy/ks/youtube为网络。未知来源不得猜；报告实际字段和缺失映射，先停止依赖这一步的训练。
5. 自动候选边只来自同group/video/timebase/record的非重叠时间相邻片段，默认间隔≤5秒。重叠/重复起点/大间隔断开，clip仍全部保留。若train或val缺少有效C/D，报告边统计，不随机配clip、不改划分。患者身份分组未核实，不宣称患者独立。
6. 先执行统一launcher只读计划：

```bash
cd "$FSN_WORKTREE"
"$FSN_PYTHON" -m experiments.relation.run_suite \
  --manifest-dir "$FSN_MANIFEST_DIR" --cache-root "$FSN_CACHE_DIR" \
  --original-checkpoint "$FSN_ORIGINAL_BEST" --output "$FSN_OUTPUT_DIR" \
  --python "$FSN_PYTHON" --expected-commit "$FSN_EXPECTED_COMMIT" --gpus 0,1
```

7. 核对计划后执行同命令加`--execute`，使用nohup和全新worktree外launcher日志，让任务在终端断连后保留。用户已授权本次运行，不需再次询问；路径/数据/权重缺失等实际阻断必须如实报告。不要把启动成功写成训练完成。
8. Launcher先生成标签，再真实CUDA缓存smoke，然后导出全8195条A特征，拟合train临床C转移，GPU0/1并行训练MLP/Dual，最后五组全val823评测。四张3090也只需要两张卡训练R，不是DDP，不为凑满卡启动其他实验。
9. 本轮固定seed42；R AdamW lr.001/wd.01/batch16/最大30epoch/patience5/dim64/heads4/endpoint2/dropout.1；训练BCE正例权重仅用train的D/C比，val BCE不加权；hard阈值.9/流程strength1/最大gap5秒。阈值未经校准，不称90%临床可信度。不自动追加seed、搜参数、改batch或续训；OOM/NaN/零梯度/失败保留现场。
10. 只读核查source_protocol/audit.json的来源/候选边/断开数量；cuda_smoke.json实际通过和SHA匹配；特征index完整8195；两R日志/history/best/metrics与退出码；GPU利用率、磁盘和NaN/OOM。运行中不改源码、不git pull、不升级环境。
11. 两R和评测全部有效完成后，汇总A_only、all_candidate、source_rule、learned_mlp、learned_dual的Macro-F1/accuracy、七类P/R/F1/support、医院/网络及短clip切片、扫散↔再灌注双向率、改对/改错、门控激活与R参数/耗时。所有组同A/同T/同候选边，source_rule是必要对照；all_candidate不是旧完整v3复现。
12. 823用于开发选模型，不是独立test。R标签与来源关联，不能仅凭高连接准确率声称理解临床连续性；若只赢全连接没赢source_rule，如实说明。只有去身份 `evaluation/public_safe_summary.json` 可交付到GitHub非main分支；视频、标注、特征、权重、逐clip预测、私有日志/协议不可上传。不要自动合并main。完整报告交付后停止自动化。

服务器关闭导致文件丢失时，先列出缺项；不能用随机A、官方初始化、合成数据或其他划分替代正式运行。
