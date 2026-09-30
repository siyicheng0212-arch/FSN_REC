# 复制以下内容给有四张 GPU 的 Codex

请实际执行训练，不要只给计划。仓库是
https://github.com/siyicheng0212-arch/FSN_REC ，使用分支
`codex/fsn-local-motion`，先读 `experiments/LOCAL_MOTION.md` 和
`scripts/run_local_motion_4gpu.sh`。

目标：在 Original Uni-AdaFocus-TSM 上公平测试早层局部相关性匹配模块。
四张空闲 GPU 分别独立训练 `original`、`local_appearance`、`local_motion`、
`local_motion_context`，每个模型单卡；不是一个模型的四卡 DDP。
网络视频 clip 之间没有顺序，不用 v3、跨 clip 标签转移、人工角色/区域标注。
旧 layer2 三路差分模块已报告失败；本次不能宣称匹配版必然涨点。

1. 先检查本地 git 状态，保留已有修改；通过新 checkout/worktree 获取指定分支。
   找出服务器已有 CUDA Python、四张空闲且支持 bf16 的 GPU、官方 SSv2
   AdaFocus 预训练权重、现有 manifest 和 uniform 36f224 cache。
   路径从服务器实际文件确认，不能编造；不要随意升级工作的 CUDA 环境。
   优先复用已有按 group 分离的内部 train/val；没有内部划分就复现现有
   train/val 并明确它是开发验证。不要私自改划分、缓存、裁剪或采样。
   缺必要输入时报告具体缺项；不能改成随机初始化继续正式训练。
2. 记录提交SHA、实际导入 `ADAFOCUS_ROOT`、GPU/软件版本、manifest/权重摘要
   和类分布。检查 train/val 的 clip_id、group_id 无重叠，训练集七类齐全，
   缓存有效。检查预训练共享非分类头权重完整加载，新key只属于local_motion。
3. 跑模块/wrapper/trainer相关单测。然后四个variant各做真实缓存一个batch的
   CUDA bf16前向/反向/更新，正式形状36输入帧/8全局帧/12局部帧/128patch。
   检查有限loss、optimizer参数覆盖、module up.weight非零梯度；更新后检查
   down/evidence梯度，上下文gate和projection允许按零初始化逐步启动。
   检查同seed共享权重、初始eval输出与Original一致。不要把早期内部零梯度
   直接判为永久失活。若OOM，四个模型统一降低batch、提高累积，有效batch64。
4. 用launcher的FSN_*环境变量传入实际路径和四个GPU编号，先dry-run，再用
   nohup启动。统一seed42、100个finetune epoch上限、patience10、batch4、
   累积16、head warmup5、module warmup0、SGD lr0.002、momentum0.9、
   weight_decay0.0005、global LR ratio0.5、spatial/temporal ratio0.2、
   sqrt_inverse类权重，默认模块64维/window3/temperature0.07/context_grid2。
   不用 --test-after-training；输出新目录，禁止覆盖旧best.pt或结果。
   启动后确认四进程实际占各自GPU、跑过warmup进入finetune并保存日志/PID。
   持续完成监控和结果汇总；不要把“进程已启动”写成“训练成功”。
5. 完成后读取best.pt对应的result/history/val_predictions和自动on/off审计。
   报四模型macro-F1、accuracy、逐类F1、最佳轮次、耗时、实际训练参数量，
   并给source/duration切片、扫散↔再灌注双向错误及其他退步类别。
   检查残差/输入比例、梯度与同checkpoint开关的logit差、改对/改错例数。
   上下文版另外查context-only开关。说明选中帧间隔/重复帧/镜头切换风险；
   缓存bin中心时间不是物理速度或已核验PTS。对照外观版与运动版参数相同，
   计算量不同。必须区分独立训练Original比较和同checkpoint关闭模块比较。
6. 先交付seed42四组结果和失败诊断，不自行启动大范围调参。若有可信优势，
   提出同协议123/2026配对重复；单seed不等于论文证据。测试集若曾用于调参
   不能称独立测试。只分享汇总指标，原视频/身份/私有路径/模型权重不上GitHub。

补充：这版实际finetune可训练参数（partial BN后）Original27,208,867，
外观/运动27,277,283，上下文27,383,971；新增68,416/175,104。
前5轮head warmup只训练55,587参数。已有CPU验证，尚未做真实GPU训练。
全量unittest发现旧test_dedupe_videos导入scripts.dedupe_videos失败，属于
main已有问题，相关新模块测试必须通过，不要掩盖或混作新模块失败。
