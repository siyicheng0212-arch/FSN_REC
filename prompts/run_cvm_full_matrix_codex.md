# 服务器Codex prompt：现代视觉模型、完整主对比与消融

请接手 `siyicheng0212-arch/FSN_REC` 的 `codex/cvm-fsn-evidence-v1` 分支。先完整读 `docs/cvm/EXPERIMENT_DESIGN.md`、`EXPERIMENT_MATRIX.md`、`RUNBOOK.md`、两份literature文档和`VALIDATION.md`，按冻结计划执行，不能只训练看起来会涨点的模型。

**本次授权为开发主矩阵：环境/私有数据审查、独立worktree、官方权重与依赖准备、四视觉模型真实CUDA核验、一次formal39训练、已训练checkpoint的val诊断和去身份聚合。默认不启动57次extended、不评测test、不运行来源派生test、不复活旧实验。** 完成后停止，不自动追加seed/调参波次。

## 保留现场与固定版本

- 服务器曾关闭，先确认实例、文件、GPU和CUDA Python是否仍可用；旧路径缺失就报告，不能假定数据已恢复。原始视频、标注和私有身份不上传。
- 不改主checkout的备份/processed_server、旧worktree/cache/best/last/log，不启动旧v3/motion/appearance/context/extra_wave/seed10/1217/1415。
- 新worktree建议`/root/autodl-tmp/FSN_REC_cvm_evidence_v2`。运行前从远程分支取得提交，记录完整SHA、clean状态、`cvm.__file__`、代码hash和环境。开始后不pull、不改源码；修代码需保留现场并在新版本输出重新冻结，不能继续混协议。
- 正式输出建议`/root/autodl-tmp/fsn_cvm_formal_v2`，新私有protocol/checkpoints.json放仓库外。任何已有输出先只读状态，禁止覆盖和重复launch。

## 1. 数据冻结

复用现存真实36×224 uint8缓存和内部train6586/val786清单（若仍存在），逐项核对digest、标签、split/group和文件。786/823已有模型选择历史，仅开发验证；旧8181和近期8195先解释版本差异，不混成绩。需要cache_split适配时只写全新私有manifest，不改原记录角色/标签/时间/组。找不到数据就报告缺失，不能生成synthetic医疗分数。

用`cvm.protocol audit`冻结train/val并提供已知prior-eval完整清单。group来源是原始录制/去重关系，不能拿clipID假冒group。患者/操作者身份未知时明确只录制组隔离，不造ID。禁止把旧val改test，禁止跨clip状态、source/duration/fileID/角色标签作为输入。

## 2. 四个视觉模型都核验

主模型为`r2plus1d_18`、`mvit_v2_s`、`videomae_base16`、`videomamba_tiny16`。四者都要flat/hierarchy真实训练。

- R2/MViT用torchvision官方K400完整权重，校验官方URL哈希前缀、严格400类加载后才删除头。
- VideoMAE用MCG-NJU pin`14ef8d856287c94ef1f985fe30f958eb4ec2c55d`，model_zoo官方ViT-B K400自监督800轮后400类fine-tuned权重，GoogleDrive官方入口id`18EEgdXY9347yK3Yb28O-GxFMbk41F6Ne`。不要换成v2、HF转换模型或纯encoder。记录来源/本地SHA/mean-pooling+fc_norm768；官方无完整publisher checksum不能假称已验证。使用安全torch加载，旧版本不支持Namespace白名单时不改成unrestricted pickle绕过。
- VideoMamba固定OpenGVLab pin`37355c26d0ae99ca2459f6d4044a5f509031a79f`，官方Ti16 K400与custom Mamba/causal-conv依赖；通用pip mamba-ssm不是替代品。

保护现有CUDA环境，必要依赖在隔离环境安装/编译。跑全部`unittest`，然后每个backbone用真实cache batch分别做FP32/bf16前后向，输出有限七类loss、各有效头梯度、主干梯度、显存和strict load_report。MAE与Mamba源码CPU形状测试不等于服务器真实官方权重/CUDA通过。任何一个缺依赖/权重时先报告，默认不偷偷跳过它并宣称完整formal。

原cache36，主输入均选16帧；R2空间112，另外3模型224，使用官方归一化/空间变换。预训练任务/分辨率/模型大小不同，跨主干不称纯架构因果比较或同FLOPs。

## 3. 39次主矩阵只执行一次

冻结`checkpoints.json`四项路径与external_repo，按RUNBOOK生成`--stage formal`计划。固定seeds42/2026/2027：

- 四主干flat/hierarchy，共8配置×3。
- R2额外capacity_control与aux_flat，共2配置×3。
- R2固定singleton random17/29/43 hierarchy，共3配置×3。
- 合计13配置×3=39。hierarchy主选模输出固定soft，同checkpoint另审计hard。所有损失权重1，none类权重，5warmup/60epoch/patience10，AdamW主干1e-5/head1e-4/warmup1e-3，wd.05，batch2/accum16有效32，clip5，bf16，全clip一致flip。只在warmup后以val七类macro-F1选best。

先显示dry-run：数量/配置ID/比较关系/权重SHA/协议SHA/代码SHA/GPU映射。用实际核验的Python nohup执行`run_cvm_4gpu.sh PLAN --execute`一次。四卡是4个独立任务，非DDP；不要再启动benchmark/ablation共享子矩阵。主结果以事前固定formal目录为准，旧pilot单列，不因重训更差改用pilot或另开seed42挑高分。

OOM/NaN/类别塌缩/无epoch时保留last/history/log并报告，不自动重启；必要batch调整须同backbone比较各模式统一、保持有效batch并全新协议/输出，不能单改一组混成绩。launcher单次锁、pids.tsv、exit_codes、result须一致；进程启动/best存在不等于完成。

## 4. 比较、消融和诊断报告

持续只读监测实际进程/GPU、disk、log尾、history、grad、best/result/退出码；每个run记录真实耗时、参数和峰值显存。缺失/中止任务保留状态，不填0、不择优补seed。

完成主矩阵后用`cvm.report --plans FORMAL/run_plan.json --output NEW_AGGREGATE.json --markdown-output NEW_TABLES.md`按配置汇总七类/五类mean±SD、逐类F1、缺失seed、比较和消融表。五类F1从完整7类混淆计算，保留singleton FP/FN；消毒/固定贡献分开。三随机组全部报告。

保留实际router、flat argmax映射组、flat概率求和组；跨组/组内错误分母；同checkpoint hard/soft改对改错。独立flat与hierarchy的oracle使用相同真实组，仅诊断，不是部署结果。同seed同主干模型比较使用`cvm.analysis compare`录制组bootstrap；与训练seed SD分别解释。

用已完成formal plan生成`--stage robustness --trained-plan ...`，固定clinical flat/hierarchy best做val none/static/shuffle，共24checkpoint×3=72次推理，0次重训；输出新目录。shuffle保留选中16帧多重集合。probe配对用`--same-checkpoint-probe`，SHA相同且仅probe变化，不能证明针/手定位或临床因果。

用`cvm.paired --plans FORMAL/run_plan.json ROBUSTNESS/run_plan.json --output NEW_PAIR_AGGREGATES --replicates 2000 --seed 42`从私有预测生成全部预声明配对bootstrap及公平oracle诊断；bootstrap seed不是新增训练seed。缺失pair保留，不做clip交集，不挑有利对照。聚合report只给跨seed描述，不能反推逐clip CI。

报告时长/来源/重复率切片，并额外固定排除`<0.1s`的敏感性，完整集仍为主指标。缓存bin中心不是已核验PTS，重复输入不是独立时间观测。来源切片不能叫来源留出泛化。

按`cvm.profile`对正式best运行真实CUDA合成shape部署profiling，预热20/测100、batch1/4，记录延迟/吞吐/显存与实际部署参数；没有跑的项写待测。不要猜GFLOPs。资源profile不是医疗精度实验。

## 5. 交付后停止

交付真实状态、完整负结果、每张表、39项完成/缺失清单、bootstrap及seedSD局限、尚缺独立test/标注可靠性/原始短片段PTS证据。若临床组不优于random/容量或五类退步，撤销相应主张，不隐藏。现代baseline是对比，不是创新；成熟层次方法也不能改名当新算法。

不自动执行extended57、视觉taxonomy、来源留出新训练或test。来源留出入口已写`cvm.source_holdout`，但须未曝光的原clean test且预声明来源；类数不足报告不足，不能把train/val提升test。原始32×224旧稿复现缺权重/variant/decoder/split单位时如实列缺项，本轮R2缓存32×112敏感性不能代替精确复现。

只将去身份聚合与无私有路径的代码/报告提交新的非main分支；视频、原标注、逐clip预测、身份、含路径原日志、cache、features和权重绝不上传，不自动merge main。停止自动化。
