# 给四张 RTX3090 服务器上的 Codex

请实际验证并运行下面的新模块四组对照。用户已决定跳过冻结特征
诊断，直接训练候选模块；不能继续启动探针或39次完整矩阵。

仓库：<https://github.com/siyicheng0212-arch/FSN_REC>
使用分支 **`codex/fsn-aligned-context`**，其中包含
`experiments/train_aligned_context.py` 和 `scripts/run_aligned_context_4gpu.sh`。
**先获取该分支真实 remote HEAD、核对本次交付提交 SHA**，再创建独立 worktree
`$FSN_WORKTREE`。不要直接使用旧
`codex/fsn-local-motion`，不要在主 checkout 上覆盖现有修改。
本 prompt 不编造提交 SHA；记录实际 `git rev-parse HEAD` 和远端对应 SHA。

以下路径仅在服务器本地确认并导出，不写入公开仓库：
`FSN_PYTHON`（CUDA Python）、`FSN_WORKTREE`（独立worktree）、
`FSN_MANIFEST_DIR`（现成内部清单）、`FSN_CACHE_DIR`（现成缓存）、
`FSN_CHECKPOINT`（官方SSv2权重）、`FSN_OUTPUT_DIR`（全新正式输出）、
`FSN_SMOKE_DIR`（全新私有烟雾检查和launcher日志位置）。

这次是 Original Uni-AdaFocus-TSM 的中间层模块训练：

- GPU0：`original`。
- GPU1：`context_plain`，普通全局／局部交互。
- GPU2：`context_aligned`，加入取帧与裁剪几何对应约束。
- GPU3：`local_capacity`，仅使用局部特征的等参数量对照。

四张卡、四个独立并行任务，不是 DDP。容量对照不等待Original结束。
四组从同一官方 SSv2 预训练初始化，
不能只给新组加载旧 Original 的七分类 best。先阅读
`experiments/ALIGNED_CONTEXT.md`、launcher 和 trainer。

1. 只读检查 `nvidia-smi`、现有训练进程、磁盘、真实 CUDA Python。
   将实际 Python 路径导出为 `FSN_PYTHON`，后续单测和启动使用同一环境。
   四张 RTX3090 必须真实可用、支持 bf16。保留旧输出和进程；如发现旧三卡
   或其他任务已经启动，先只读核对 PID／协议／结果，保留现场和状态，
   不能重复启动、自动停掉旧任务、覆盖或直接转换正在运行的旧计划。
   绝不启动旧 v3、local_motion、appearance、context、extra_wave 或
   seed10/1217/1415，不重建缓存、不自动重训丢失的旧 Original。
2. 输入优先复用：
   - manifest：`$FSN_MANIFEST_DIR`
   - cache：`$FSN_CACHE_DIR`
   - 官方预训练：`$FSN_CHECKPOINT`
   路径必须真实确认。launcher 核对历史 train6586/val786 和权重 SHA。
   七类齐全，train/val clip/group 无交叉，所有缓存36f224有效。
   val 存于原train缓存时使用只读 storage mapping；不改原清单的 split 或 SHA。
   内部清单有 `inner_split_role` 时按它识别train/val；原`split`保留缓存来源。
   `split=train, inner_split_role=val` 是验证样本，读取其现成train缓存。
   使用本次修复提交中的统一解析器，不改写清单、不绕过group/标签/cache检查。
   旧smoke失败现场保留，更新固定代码后在全新smoke位置重新检查。
   必要文件丢失则明确报告缺项；不要使用随机初始化或合成样本替代。
3. 用确认后的 CUDA Python 运行 `OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$FSN_PYTHON" -m unittest discover -s tests -p 'test_aligned*.py' -v`
   （47项新模块、wrapper、trainer、数据解析、launcher单测）。核对实际源码
   导入来自该 worktree，所有共享非分类头预训练张量完整加载；七类头和
   新模块才允许新初始化。确认四组同seed共享权重和七类头一致。
   保留 Original 原来的 detach／辅助损失策略，不能只检查最终 logits 梯度。
4. 创建全新私有 smoke 位置，在真实缓存batch4上执行专用 CUDA bf16烟雾：

   ```bash
   cd "$FSN_WORKTREE"
   CUDA_VISIBLE_DEVICES=0,1,2,3 "$FSN_PYTHON" -u -m experiments.train_aligned_context \
     --smoke-only \
     --manifest-dir "$FSN_MANIFEST_DIR" \
     --cache-dir "$FSN_CACHE_DIR" \
     --checkpoint "$FSN_CHECKPOINT" \
     --smoke-report "$FSN_SMOKE_DIR/smoke.json"
   ```

   四组每组实际三步，不是一次forward；初始eval与Original一致、loss有限、
   up投影首步梯度非零、下游投影/attention按零初始化在后续步骤获得梯度。
   时间/裁剪对应关系来自实际采样与crop状态，不得用帧序号硬凑。
   CUDA smoke 和任何失败现场必须真实记录；不能写“已通过”代替执行。
   smoke后若改动运行源码或输入，旧report失效，先提交再重新smoke。
5. 固定提交，保持隔离worktree clean，先只读生成正式计划：

   ```bash
   bash scripts/run_aligned_context_4gpu.sh \
     --manifest-dir "$FSN_MANIFEST_DIR" \
     --cache-dir "$FSN_CACHE_DIR" \
     --checkpoint "$FSN_CHECKPOINT" \
     --output-dir "$FSN_OUTPUT_DIR"
   ```

   正式目录必须完全不存在；不能预先mkdir这个目录。只有全数smoke通过且
   四张卡空闲后，才在上述命令添加
   `--execute --smoke-report "$FSN_SMOKE_DIR/smoke.json"`
   并用nohup启动一次。把launcher总日志放smoke目录或另外的新日志位置，
   避免shell重定向提前创建正式输出目录。记录launcher PID和实际任务PID。
6. 四组固定：seed42，5轮head warmup lr.001，最多100finetune/patience10，
   batch4/accum16有效batch64，workers4，SGD momentum.9 lr.002 wd.0005，
   global/stn/temporal LR ratios .5/.2/.2，sqrt_inverse七类权重，clip_grad20，
   bf16，context dim64/grid3/time_scale.25/spatial_scale1/lr_ratio1。
   不修改取帧、裁剪、划分或缓存；不使用跨clip顺序、人工框/角色，不评测test。
   OOM或NaN先保留现场报告；不偷偷改一组batch或重启。运行中不git pull、
   不改代码、不升级环境。某任务失败保留现场，其他正常独立任务可以结束。
7. 启动后只读确认GPU0/1/2/3四个实际映射、四个任务PID及日志；
   首次完整epoch核对history/best。
   确认进入finetune，新模块残差和梯度有记录。监控NaN/OOM、单类塌缩、
   进程退出、磁盘与长时间无epoch。四组各自核对进程退出码和结果文件，
   不需要等Original结束才启动容量对照。不能把启动成功称作训练完成。
8. 四组全部有效完成后，交付内部验证七类Macro-F1/accuracy、逐类P/R/F1/support、
   五类细动作指标、最佳轮次、训练耗时、参数和资源、扫散↔再灌注双向错误、
   来源/时长切片、其他退步类、模块梯度／残差及同checkpoint on/off改对改错。
   分清独立训练Original和同checkpoint关闭模块。选帧间隔/重复帧/剪辑风险
   要如实解释：缓存bin中心时间不是已核验PTS，几何attention不证明临床机制。
   val786同时用于选best，只有开发诊断意义；单seed涨点不是论文结论。
   先报告aligned是否优于Original、普通交互和容量对照，然后停止自动化。

只允许后续分享去身份聚合报告；不要上传视频、原标注、私有路径日志、
缓存、权重、特征或逐clip预测，不自动合并main，不自动扩展多seed。
