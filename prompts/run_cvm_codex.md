# 服务器 Codex prompt：CVM 开发阶段，四卡核验与固定 pilot

你接手 FSN_REC 的 CVM 修订实验。请先读 `docs/cvm/EXPERIMENT_DESIGN.md`、`RUNBOOK.md` 和两份 literature 文档，再执行本任务。代码是 `siyicheng0212-arch/FSN_REC` 的 `codex/cvm-fsn-evidence-v1` 分支。目标是完成**旧稿临床层次结构的严格实验**，不是恢复之前失败的局部运动模块。

本次授权：只读盘点与备份检查，创建独立 worktree/私有协议和全新输出，取得明确官方预训练权重、必要依赖核验，真实 CUDA smoke，以及 smoke 通过后一次 seed42 四组 R2+1D pilot。缺失的关键信息应报告，已有可完成的开发工作继续做。不要重复询问已经授权的上述操作。

## 禁止事项与固定现场

- 不启动或续训旧 local-motion/local-appearance/context/v3，不恢复旧 seed10/1217/1415/extra_wave。服务器此前已关闭，不能假定原进程仍存在。
- 不改主 checkout 的未跟踪备份、processed_server、旧 worktree、best/last/log/cache；不删旧实验。
- 新 worktree 建议 `/root/autodl-tmp/FSN_REC_cvm_evidence`。从远程本分支创建，在运行前核验远程 SHA、工作树清洁、实际导入 `cvm.__file__`，记录 commit 与实现 hash。运行期间不改源码、不 git pull。
- 新协议建议 `/root/autodl-tmp/fsn_cvm_protocol_seed42_v1`；smoke/pilot 输出分别 `/root/autodl-tmp/fsn_cvm_smoke_seed42_v1` 和 `/root/autodl-tmp/fsn_cvm_pilot_seed42_v1`。若已存在，先只读查看状态，禁止覆盖或重复 launch；需要不同配置时用明确新版本目录。
- 不评测 test，不把 old validation 改成 test；不使用跨 clip 顺序、人工角色标签、source/duration/fileID 当模型输入。原稿8181和近期8195不是同版本，分数不可直接混排名。
- 不静默换随机初始化、删困难样本、按结果选随机组或种子，不只为某一组改超参数。

## 1. 只读盘点与协议适配

检查四张 GPU、CUDA Python（此前为 `/root/miniconda3/bin/python`，重启/更换服务器后必须重新核实）、磁盘、已有 manifest/cache/weights。若旧实例已释放、文件不可得，明确报告哪些缺失；不要生成假数据或拿 synthetic 测试替代正式结果。

优先核对内部开发清单 `/root/autodl-tmp/formal_clip_evidence_pilot/inner_manifests`：train6586/val786。历史 SHA：train `f0fc0ace561cd59c471c86d6ec674144de0d5ede6acf447a5a021787538ebf0a`，val `d6449196c3798a233a593db19da9045c92e02ba62bb6fa5a55d41bf118ee6580`。这786已用于旧模块比较，**只作为开发验证**。

缓存候选 `/root/autodl-tmp/full_cache_36f224_trainval`：原7372+823，逐记录核对 request_digest、uint8/36/224 shape 与对应文件。若内部val文件实际位于 cache/train，仅在**新私有适配 manifest**添加 `cache_split=train`；其 `split=val` 角色保留，clip/group/labels/video_path/start/end绝不改。原清单保持原样。禁止自动重建采样或复制视频。

用 `python -m cvm.protocol audit` 冻结 train/val。以 `--prior-eval-manifest` 提供已有用于选择的786及823等完整历史；找不到的历史明确标为不完整，不认证独立test。不能绕过组/源/身份重叠错误；缺失患者/术者ID时只称recording-disjoint。

## 2. 环境、官方权重与实载

在新 worktree 运行 `python -m unittest discover -s tests -p 'test_cvm_*.py' -v`；不要卸载现有可用 CUDA torch。记录依赖、CUDA与四张GPU。所有新增源代码在开始实验前固定；发现必须修复的代码错误时停止，保存现场，在新的提交/协议版本下重新审查，不能运行中修改。

本次四组仅 R(2+1)D-18。取得 torchvision 原始 K400 checkpoint，同一文件供四组用，strict loading + 官方 URL 哈希前缀验证完整400类模型，再重置任务头。可显式 `build_model(...allow_download=True)` 从官方入口预下载一次；正式 suite 只用其本地固定路径。生成仓库外 `checkpoints.json`。

先用真实 train/cache 示例完成 R2+1D 的 FP32 与 bf16 单 batch 前后向：原cache36，选16帧；native空间112；检查七类输出、有限loss、有效分类头梯度和finetune主干梯度、显存。未训练头无梯度是预期，不能当失败；singleton没有fine头。给出具体load_report和参数报告。

VideoMamba/MViT 此阶段不要求启动训练。可只读准备其正式权重；VideoMamba必须固定官方repo `37355c26d0ae99ca2459f6d4044a5f509031a79f` 与定制CUDA扩展，不用标准mamba-ssm替代。缺少其真实GPU前后向，就不能声称现代完整矩阵已准备好。

## 3. smoke → pilot，一次启动

按 RUNBOOK 生成 `--stage smoke` 的 plan，不带 `--execute` 先核对。默认4个独立R2+1D任务：flat、capacity_control、aux_flat、hierarchy；seed42；GPU0/1/2/3；warmup0、epoch1。确认计划SHA、cache、权重、输出和训练配置一致后执行一次。smoke是完整一轮真实管线核验，不是论文分数。

四组smoke均exit0、result.training_completed=true、报告可解析、分类头/主干梯度有限且活跃后，生成全新 `--stage pilot` plan并 nohup 启动一次。使用 `CVM_PYTHON` 指定核实的CUDA Python，`scripts/run_cvm_4gpu.sh PLAN --execute`；外层日志用全新文件且不得覆盖。以launcher的pids.tsv、exit_codes.tsv和实际进程为准，不硬编码PID。

pilot固定：seed42；5轮head-only warmup；最多60 finetune/patience10；batch2/accum16有效32；AdamW主干lr1e-5、headlr1e-4、warmupheadlr1e-3、weight_decay.05；不加类权重；bf16；clip_grad5；全clip一致horizontal flip；aux/group/conditional weights全部1。四组同一主干初值、顺序、采样与最大预算。主 best 仅从finetune验证七类macro-F1选择。

hierarchy使用group+true-group conditional CE，部署主输出固定soft联合叶概率；同checkpoint另审计旧hard路由和oracle。aux_flat仍输出flat七类；capacity_control匹配aux_flat额外活跃参数。不要把这些成熟方法称新算法。

OOM/NaN/长时间无epoch/单类塌缩时保存现场并报告；不盲目重启。只允许同一比较四组统一变更batch/accum保持有效32且使用全新协议输出；不静默修一组。人工中止记录interrupted，不能写训练完成。

## 4. 监测、汇总后停止

每次只读检查GPU/进程、日志尾、磁盘、最近history、best/last、result和退出码。核对四组进入finetune；history的首个优化窗口按组件梯度、实际有效batch、学习率和数据样本数一致。warmup分数不是FullFT best。

全部完成后汇总：

- 七类 macro precision/recall/F1、accuracy、weightedF1、逐类support/混淆。
- 五类主F1从完整七类混淆选择，保留消毒/固定误入五类的FP；单元素类贡献另列。
- 同hierarchy checkpoint hard/soft改对/改错；实际coarse router、flat叶argmax映射组、flat概率求和组分别报告。
- 公平的hierarchy oracle与独立flat oracle给予**同一taxonomy**真实组提示，标为诊断。
- 扫散↔再灌注和针操作双向错分；来源、时长、选中16帧像素重复率；短clip敏感性；资源与最佳finetune轮次。
- `cvm.analysis compare` 严格配对同clip/metadata并按原始录制group bootstrap；seed42 CI不是跨seed证据。`cvm.report --expected-seeds 42`列出真实状态，不写多seed。

可在完成训练后用固定checkpoint做val static/shuffle，各输出新目录；shuffle必须保留已选16帧集合。不要根据诊断结果调整本轮模型，重新开发必须新版本。

不要自动进入33次formal矩阵，不追加旧seed/wave，不做test。给出本轮真实结果、负结果和尚缺独立test/近期模型验证的清单后暂停。正式矩阵已在代码中预注册11配置×42/2026/2027，但不因某一单seed涨点宣称论文成立，也不删不利对照。

若发布，只导出去身份聚合到新非main分支；视频、原标注、逐clip预测、含路径原日志、身份表、cache、features、weights和私有配置不上传，不自动merge main。报告必须区分“代码检查”“启动”“训练完成”“单seed开发结果”和“独立测试证据”。
