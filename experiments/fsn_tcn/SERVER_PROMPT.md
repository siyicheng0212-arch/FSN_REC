# 给服务器 Codex 的最小 FSN-TCN 操作任务

请运行 FSN_REC 的最小 FSN-TCN 实验，完整保留 train7372/val823，复用冻结 Original 的现有封存特征和候选链。不得启动旧 extra_wave、续训旧实验、覆盖旧输出、读取 test或上传私有数据。只执行本任务固定配置中的种子，不自动追加。

1. 找到并检出 GitHub 分支 `codex/fsn-tcn-minimal` 的最新提交，在服务器创建独立干净 worktree；记录完整SHA。运行期间不 git pull、不改代码。先阅读 `experiments/fsn_tcn/README.md`。
2. 特征和来源协议优先只读复用 `/root/autodl-tmp/formal_source_relation_suite_7372_823_seed42_v1/features/index.jsonl` 与 `source_protocol/`。从既有 export protocol 查明 full7372/823 manifest路径和已完成的Original best.pt，核实实际文件SHA；不要把SSv2初始化权重当作已训练A。缓存优先 `/root/autodl-tmp/full_cache_36f224_trainval`。
3. 用户提到的旧 `tcn_baseline.py`、`hard_dual.py`、`evaluate_robustness.py` 不在本次公开基代码里。若服务器有实际旧TCN，先只读核对输入表示、网络、singleton行为与训练设置，用externalfactory适配，并用同权重/同输入验证全接parity。不要声称默认新reference复现了旧模型，也不要未经核对静默替换旧实验。如果缺少旧文件，明确采用config中标记的新reference做四组同协议比较，保留该限制，不复用旧TCN分数。
4. 核实CUDA Python（优先 `/root/miniconda3/bin/python`）、torch/torchvision/numpy/pytest、GPU0–3型号与空闲状态、磁盘。禁止停止他人的任务。四组不是DDP。
5. 固定一份配置：默认先做 `minimal_reference.json` 的seed42；若本轮计划多seed，必须在看本轮结果之前一次性列出所有seed，不根据涨点追选。R门阈值默认0.9固定，不搜索。四组TCN同初始化种子、宽度、层数、CE、优化器、训练预算；R加入train-only同来源不同临床记录的hardneg，val合成挑战不得进入R训练/选checkpoint。
6. 用 `python -m experiments.fsn_tcn.run_suite` 加完整绝对路径参数先生成只读计划，确认 full7372/823、SHA、配置、负例与挑战可构造、GPU映射和全新输出。输出建议 `/root/autodl-tmp/formal_fsn_tcn_minimal_7372_823_seed42_v1`，不允许已存在。确认计划后，同命令加 `--execute --expected-commit <完整HEAD SHA>`，按任务授权运行，不再次请求常规确认。
7. 启动器必须先完成单测、真实原缓存A/R CUDA smoke、TCN CUDA前向/反向/参数更新和断链smoke，成功才会训练。失败保留报告并停止，禁止绕过smoke或改某一组。无可训练连接时报告原因，禁止临时降低阈值。
8. 每seed先训练R，冻结R后四组TCN独立训练：all GPU0、source_rule GPU1、random GPU2、learned GPU3。种子之间串行；不得重复启动。重连以launcher_logs/pids.tsv和实际进程为准，启动成功不等于训练完成。
9. 四组exit0且result可解析后，汇总原val823和两种固定合成挑战，完整报告Macro-F1/Acc、七类指标、两个方向的扫散/再灌注错分、改对/改错、来源/时长切片、门控开启率、随机门实际位置差异、可训练clip数及优化步数、参数量/显存/耗时。区分四组独立训练与同checkpoint切换门控审计。
10. 所有评测只是开发验证及合成断点测试。不得称独立test、真实剪辑检测成功或论文结论。只交付安全聚合，不上传权重、特征、预测、视频、原标签或含路径日志，不自动合并main。完成固定计划后暂停。
