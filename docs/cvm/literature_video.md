# FSN 单片段细动作识别：视频与医疗文献、基线和可复核协议

查阅日期：2026-09-30。研究对象是独立七分类片段；互联网来源片段之间**没有已知顺序**。本轮主线是原稿的共享视觉骨干和临床层次分组，不把 AdaFocus、跨片段流程建模或人工框引入主实验。层次方法的先行工作和创新边界见 [literature_hierarchy.md](literature_hierarchy.md)。

这里的“已检查”区分全文方法、官方实现与仅摘要。十一项视频／医疗来源的已核书目信息见 [references.bib](references.bib)。下述文献提供实验设计依据，不构成 FSN 上一定涨分、具有新颖性或获得 CVM 接收的证据。

## 原始来源及对 FSN 的实际含义

| 原始研究与出处 | 实际检查范围／贡献 | 可用于 FSN 的部分 | 不能直接推断的部分 |
| --- | --- | --- | --- |
| **Temporal Segment Networks: Towards Good Practices for Deep Action Recognition**，Wang 等，ECCV 2016。[原论文](https://arxiv.org/pdf/1608.00859)；[作者代码](https://github.com/yjxiong/temporal-segment-networks) | 全文 §3.1–3.3：把同一视频分成等时长段，各段采样片段，共享网络后聚合预测；另考察预训练、正则与测试视角。 | 对**一个已标注 clip 内**的分段采样提供清楚起点；保存取帧索引，区分输入帧数与测试视图数。 | TSN 的长程覆盖是同一视频内部的覆盖，不能给互不相邻的互联网 clip 排流程；其采样在快速细动作上未必最合适。 |
| **A Closer Look at Spatiotemporal Convolutions for Action Recognition**，Tran 等，CVPR 2018。[原论文](https://arxiv.org/pdf/1711.11248)；[CVF 出处](https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_A_Closer_Look_CVPR_2018_paper.html) | 全文 §3、§4.3：把 3D 卷积分解为空间卷积、非线性、时间卷积，保持近似参数量；比较训练／测试输入长度与多 clip 评估。 | R(2+1)D-18 是易复现的卷积参照，适合先验证 shared-backbone 层次头是否有用。 | 不是现代最强方法。改变测试长度并不免费获得更好的时间建模；原文 34 层和多视角成绩不等于本实现 18 层单视图成绩。 |
| **MViTv2: Improved Multiscale Vision Transformers for Classification and Detection**，Li 等，CVPR 2022。[原论文](https://arxiv.org/pdf/2112.01526)；[CVF 出处](https://openaccess.thecvf.com/content/CVPR2022/html/Li_MViTv2_Improved_Multiscale_Vision_Transformers_for_Classification_and_Detection_CVPR_2022_paper.html) | 全文 §4.1、§4.3、Kinetics 预训练消融：分解相对位置编码、残差 pooling、时空多尺度特征；比较不同预训练。 | MViTv2-S 提供不同于卷积的第二种骨干，torchvision 有明确 K400 权重与空间预处理。 | 多尺度视觉特征不等于临床类别层次；2022 模型不能单独支撑“截至 2026 的近期基线”。 |
| **VideoMAE: Masked Autoencoders Are Data-Efficient Learners for Self-Supervised Video Pre-Training**，Tong 等，NeurIPS 2022。[正式原论文](https://proceedings.neurips.cc/paper_files/paper/2022/file/416f9cb3276121c42eebb86352a4354a-Paper-Conference.pdf)；[固定官方 model zoo](https://github.com/MCG-NJU/VideoMAE/blob/14ef8d856287c94ef1f985fe30f958eb4ec2c55d/MODEL_ZOO.md) | 全文方法、消融和 domain-shift 小节，另检查官方模型／数据／800e 微调配置：高比例 tube masking 与视频自监督重建；ViT-B 有 16 帧、224、K400 自监督后 K400 分类微调权重。 | 已接入 `videomae_base16`，提供注意力 Transformer＋视频自监督迁移参照；明确两阶段与无额外数据的官方声明。 | “数据高效”不证明 FSN 从头自监督必然成功；不能把 masked-pretraining encoder、K400-finetuned 和跨数据蒸馏 checkpoint 混称相同基线。本实现未在 FSN 上训练，也未读取发布权重二进制。 |
| **VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking**，Wang 等，CVPR 2023。[原论文](https://arxiv.org/pdf/2303.16727)；[正式出处](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_VideoMAE_V2_Scaling_Video_Masked_Autoencoders_With_Dual_Masking_CVPR_2023_paper.html)；[固定官方 zoo](https://github.com/OpenGVLab/VideoMAEv2/blob/29eab1e8a588d1b3ec0cdec7b03a86cca491b74b/docs/MODEL_ZOO.md) | 全文 §3.2–3.3 与表 3–4，核查官方 ViT 模型与 checkpoint 路线：decoder dual masking 降低预训练成本，扩大无标注数据并分阶段微调；推理端仍可为同类 ViT-B mean-pool encoder。 | 说明比较应同时区分 encoder 家族和权重训练历史；可作另立的预训练规模／蒸馏扩展实验。 | 固定 zoo 的强 B 链接是 K710 teacher-distilled checkpoint，不能称 K400-only finetuned。当前 HF `VideoMAEv2-Base` 的自监督 encoder 又是另一发布路线。未为 FSN 接入 v2，不能把 v2 的数据／蒸馏收益归因于新增层次头。 |
| **VideoMamba: State Space Model for Efficient Video Understanding**，Kunchang Li 等，ECCV 2024。[原论文](https://arxiv.org/html/2403.06977v2)；[ECCV 出处](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3773_ECCV_2024_paper.php)；[官方实现](https://github.com/OpenGVLab/VideoMamba) | 原论文 §3、短视频实验，官方单模态模型、model zoo、训练／数据代码：双向状态空间 block，时空 token 序列，CLS 特征；Ti 有 K400 16 帧权重。 | 作为可执行的第三种、2024 年架构家族参照；每个片段独立调用，没有跨 clip 隐状态。 | 不是对 2026 最新 SOTA 的全面覆盖。SSM 顺序指 clip 内 token 顺序；不产生 clip 间流程关系。论文中的长视频速度和多视图成绩不能移植到 FSN/3090。注意另一篇同名近似论文不是本适配对象。 |
| **FineGym: A Hierarchical Video Dataset for Fine-grained Action Understanding**，Shao 等，CVPR 2020。[原论文](https://arxiv.org/pdf/2004.06704)；[作者项目](https://sdolivia.github.io/FineGym/) | 全文类别／时间标注体系与 §4.4：专家规则定义细类；细粒度性能受取帧数量、时间扰动、预训练域影响。 | 支持专家写明易混细动作的可见判据；支持静态帧／打乱 clip 内帧／采样覆盖的诊断。 | 体育快速全身动作的结果不保证针刺细动作同样依赖光流。语义层次和时间分段是不同标注；不可用临床 taxonomy 代替 clip 间时间标注。 |
| **A Dataset and Benchmarks for Segmentation and Recognition of Gestures in Robotic Surgery**，Ahmidi 等，IEEE TBME 2017。[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5559351/) | 全文 §II.C 与统一基准：LOUO 留出全部某术者 trial；LOSO 留出各术者某次 trial；分别测未知术者和已知术者的新 trial。 | FSN 分组至少隔离原始视频／录制 session；已知术者时再做术者隔离，避免把视频隔离叫作术者泛化。 | JIGSAWS 有连续机器人操作和运动学。其语法/HMM 的顺序约束不适用于无序 FSN 网络片段；其标签和分割指标不能直接作为 FSN 七分类结果。 |
| **A vision transformer for decoding surgeon activity from surgical videos**（SAIS），Kiyasseh 等，Nature Biomedical Engineering 2023。[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10307635/)；[出版页面](https://www.nature.com/articles/s41551-023-01010-8) | Methods：冻结 DINO 图像 ViT 特征、RGB/RAFT flow、时间 Transformer、监督对比原型；Results 分别检验视频、术者、机构和术式；分类与技能任务分开。 | 医疗细动作论文需要明确泛化对象；对照模型使用同一数据，机构外部集可揭示域偏移。RGB-only 主实验与光流扩展应分别报告预算。 | 其连续视频推理采用多窗口／多模型／TTA／聚合，不能给 FSN clip 间平滑背书。七分类动作正确不等于操作质量或疗效；技能判断需要相应监督与验证。 |
| **Data Splits and Metrics for Method Benchmarking on Surgical Action Triplet Datasets**，Nwoye、Padoy，2022 提交、2023 修订的 arXiv 预印本。[全文](https://arxiv.org/pdf/2204.05235)；[官方 CholecT50](https://github.com/CAMMA-public/cholect50) | 全文 §2–3：统一视频划分、交叉验证与类别覆盖；解释 per-video AP 与跨视频聚合 AP 的不同。此处不虚构正式会议出处。 | 所有方法使用同一冻结分组划分；披露类别支持数和聚合方式；视频级分组对照优先于随机 clip 混分。 | CholecT50 是 instrument–verb–target 多标签 triplet；其 mAP 不应不加解释地成为 FSN 单标签主指标。网络转载近重复还需要额外关联。 |
| **Action Detail Matters: Refining Video Recognition with Local Action Queries**（FocusVideo），Wang 等，CVPR 2025。[正式页面](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Action_Detail_Matters_Refining_Video_Recognition_with_Local_Action_Queries_CVPR_2025_paper.html)；[作者 PDF](https://april.zju.edu.cn/core/papercite-data/pdf/wang2025adm.pdf) | 核对正式出处与作者 PDF 的索引方法摘录；直接 PDF 阅读受访问限制，**未声称完整方法复核**。其主题是可学习 local action queries 与全局上下文。 | 对将来的无框局部细节路线，说明“学 local queries 再融合 global”已经有近邻研究。 | 当前方案未复现该法；不能声称已确定其所有实现、FSN 效果或计算成本，也不能把宽泛局部注意力作为首创。 |

这十一项并非横跨所有 2025–2026 方法的穷尽综述；选择依据是本实验可执行性、细动作证据和医疗评估。代表性家族对照与“最新最强基线”是不同主张。

## 最小可执行骨干组

建议以 **R(2+1)D-18、MViTv2-S、VideoMAE-B16、VideoMamba-Ti16** 完成同一层次头研究：分别覆盖卷积、多尺度注意力、视频 masked-pretraining 注意力和 2024 状态空间架构。先完整跑一个骨干的机制对照，再在其余三种验证可迁移性；不因某骨干失败而只发表最优骨干。接入适配器表示可执行代码就绪，正式预训练加载、服务器预检和 FSN 训练仍须执行，不能写成已有四种模型的结果。

| 实验注册名 | 预训练来源 | 输入与归一化（作用于现有缓存） | 权重／实现要求 |
| --- | --- | --- | --- |
| `r2plus1d_18` | torchvision `KINETICS400_V1` | 主协议建议 16 帧；空间 resize `[128,171]` → center crop `112²`；mean `[.43216,.394666,.37645]`，std `[.22803,.22145,.216989]` | 原 400 类全模型严格加载后移除 `fc`；保存权重 SHA256。 |
| `mvit_v2_s` | torchvision `KINETICS400_V1` | 16 帧；短边 resize `256` → center crop `224²`；mean `[.45]*3`，std `[.225]*3` | 全模型严格加载后移除分类 linear；保留原生实现，记录 torchvision 版本。 |
| `videomae_base16` | 官方 zoo 声明 K400 自监督 800 epochs → supervised K400 finetuning，无 extra data | 16 帧；短边 resize `224` → center crop `224²`；mean `[.485,.456,.406]`，std `[.229,.224,.225]` | 本地官方外部 checkout 和 **800e classification-finetuned** checkpoint 必填；`cvm/videomae.py` 返回 mean-pool＋`fc_norm` `[B,768]`，不使用 CLS，不把 decoder 导入分类网络。 |
| `videomamba_tiny16` | 官方 model-zoo 声明 ImageNet-1K → supervised K400；**16 帧** checkpoint | 16 帧；短边 resize `256` → center crop `224²`；mean `[.485,.456,.406]`，std `[.229,.224,.225]` | 本地官方外部 checkout 和 checkpoint 必填；`cvm/videomamba.py` 读取归一化 CLS `[B,192]`，无人工框、无跨 clip cache。 |

torchvision 实际报告 R2Plus1D 权重的 K400 评估为 `clip_len=16, frame_rate=15, clips_per_video=5`，MViTv2 为 `clip_len=16, frame_rate=7.5, clips_per_video=5`。参见 [R2Plus1D 官方文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r2plus1d_18.html)、[MViTv2 官方文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.mvit_v2_s.html)。**从现有 36 帧缓存固定均匀选择 16 个位置是本实现的迁移协议**，不是复现上述 K400 时间采样或多视图成绩。16 帧也不能掩盖长 clip 的运动欠采样。

缓存已经以保持长宽比的放大＋中心裁剪得到 `224²`；本轮不重新解码，也不能恢复缓存外的原画面。上述骨干变换再次作用于这些方形缓存帧，因而“使用官方空间变换／归一化”不等于“完整复现官方原始视频输入流程”。训练与验证均使用同一组固定时间位置，训练仅支持全 clip 一致的随机水平翻转或关闭增强；未实现分段随机取帧、随机空间 crop、光流或密集窗口。

扩展计划中的 32 帧是仅在 R(2+1)D 上预先声明的单因素输入覆盖对照；它仍来自同一 36 帧缓存，不能恢复原视频中未缓存的细运动。MViTv2／VideoMAE／VideoMamba 保持各自固定 16 帧契约；适配器不为取得更多时间点而插值 pretrained position 或换 checkpoint。

跨骨干空间分辨率、预训练阶段和原生正则不同；这个表考察结论跨家族是否稳健，不作等算力架构因果比较。同一骨干内 flat／capacity control／auxiliary flat／hierarchy 使用相同的起始 backbone 权重、采样、空间变换、种子、最大训练预算和早停准则，才能讨论头部机制；实际提前停止的 epoch／更新数另行报告。参数、训练显存、吞吐和单视图延迟分别实测，不能用论文 FLOPs 代替本机端到端成本。

## VideoMAE 已接入部分与未验证部分

固定官方提交为 `14ef8d856287c94ef1f985fe30f958eb4ec2c55d`，模型 git blob 为 `66dbe940cb559b06ccc7a5b616e41080fa43b2b9`。检查对象是以下原始文件：

- [模型与 `forward_features`](https://github.com/MCG-NJU/VideoMAE/blob/14ef8d856287c94ef1f985fe30f958eb4ec2c55d/modeling_finetune.py)：ViT-B 为 768 维、12 blocks、12 heads；`all_frames=16`、Conv3d tubelet2／patch16 产生 1568 tokens。`forward_features` 在所有时空 tokens 上平均后 `fc_norm`，无 CLS token。先构造原 400 类头；直接 `num_classes=0` 会碰到官方构造器对 head weight 的初始化，不能作为加载前捷径。
- [官方 800e K400 finetuning 配置](https://github.com/MCG-NJU/VideoMAE/blob/14ef8d856287c94ef1f985fe30f958eb4ec2c55d/scripts/kinetics/videomae_vit_base_patch16_224_tubemasking_ratio_0.9_epoch_800/finetune.sh)、[K400 数据与 normalization](https://github.com/MCG-NJU/VideoMAE/blob/14ef8d856287c94ef1f985fe30f958eb4ec2c55d/kinetics.py)：适配器使用 mean-pooling、drop-path 0.1、`init_scale=0.001`；关闭 activation checkpointing。K400 dataset 的归一化是 ImageNet mean/std；模型 `_cfg()` 的通用 `.5` 元数据并非这份 K400 实验的数据变换。
- [安装说明](https://github.com/MCG-NJU/VideoMAE/blob/14ef8d856287c94ef1f985fe30f958eb4ec2c55d/INSTALL.md)：已知官方 timm 为 0.4.8／0.4.12；适配器只导入 encoder 源文件，需要 NumPy、PyTorch、兼容 timm，不依赖原训练 runner 的 DeepSpeed，也不需要 VideoMamba custom CUDA kernels。

固定 zoo 的 **800e Finetuned** [官方 Google Drive 权重入口](https://drive.google.com/file/d/18EEgdXY9347yK3Yb28O-GxFMbk41F6Ne/view) 已核对，当前不下载二进制。相邻的 “Pretrained” 链接是 masked-pretraining artifact，不可替代该分类微调文件。官方 zoo 报告 `16×5×3` 测试视图；现有 FSN 缓存的单视图固定均匀 16 帧不复现原文 dense sampling／多 clip、multi-crop 配方。

```python
from cvm.videomae import build_videomae_base16

encoder = build_videomae_base16(
    repo_path="/path/to/VideoMAE",
    checkpoint_path="/path/to/official_vitb_k400_800e_finetuned.pth",
    seed=42,
)
# encoder(normalized_video[B,3,16,224,224]) -> features[B,768]
# encoder.feature_dim == 768; encoder.load_report contains provenance.
```

加载前验证 HEAD、模型 blob 与该文件 clean 状态，兼容 timm 版本另行记录。以 `weights_only=True` 读取文件，严格检查全部 native keys／shapes、非有限参数和 400 类头后再删除 head；不 interpolate position、不借用 v2 K710 子集头、不省略 decoder 差异。含 `argparse.Namespace` metadata 的文件需要支持 `safe_globals` 的 PyTorch；旧版本仅接受 tensor-only artifact，失败时给出明确错误。报告本地 SHA256 与来源声明，发布者 checksum 未取得，**严格兼容不认证训练历史**。

与 VideoMamba 的可学习 temporal position 不同，VideoMAE 默认 sinusoidal `pos_embed` 是普通 tensor，**不在 state_dict**；因此相同参数形状不证明 checkpoint 用过 16 帧／224 输入。若文件含官方 `args`，逐项核对模型、frames、segments、tubelet、resolution、pooling 和 K400 标签；缺少这些 metadata 时在 `load_report` 中把输入配置认证记为 false，而不是声称已证实。须归档官方入口来源，并由真实预训练权重的服务器加载与 forward 预检补足证据。

在 PyTorch2.2.2+CPU／timm0.4.12 上，13 项小型 fixture 测试已验证 strict 全头加载、mean-pool/fc_norm 梯度、歧义容器／错误 head／不完整 encoder／异常 metadata 拒绝、source pin 和调用者 RNG 保持。另用**实际已核 blob 的官方源码、随机初始化和合成零输入**完成 `[1,3,16,224,224] → [1,768]` 的 native CPU 前向，输出有限；实际计数为含 400 类头 **86,534,800** 参数，去头 encoder **86,227,200**。这些是源码接口／参数量检查，不是官方发布权重验证、FSN 精度或 GPU 成本。正式 pretrained/CUDA 前向和训练未执行。ViT-B 的 1568-token attention 训练成本必须在 3090 上实测；以 batch1、AMP 和有效 batch 的梯度累积预检，再固定所有头变体预算，不推测显存或训练小时数。

VideoMAEv2 固定 zoo 的公开 B 强权重来自 K710 teacher distillation；[HF distillation artifact](https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_b_k710_dl_from_giant.pth) 与 [当前 HF VideoMAEv2-Base](https://huggingface.co/OpenGVLab/VideoMAEv2-Base)（自监督 encoder）不等于上述 K400-only classification-finetuned route。它们可作为另一个显式命名的迁移条件，比较时重新严格匹配 native labels／head、记录 teacher／pretraining 数据。不能仅把文件改名或挑出 400 logits 就宣称完成相同训练历史的 v2 基线。当前选择 v1-B 的理由是有明确可核的 16f224 K400-finetuned artifact，而不是宣称 v1 比 v2 更强。

## VideoMamba 已接入部分与未验证部分

已检查并固定的官方代码提交为 `37355c26d0ae99ca2459f6d4044a5f509031a79f`，模型文件 git blob 为 `2625505bd72cb40842d1b25abd7f08bccc0a9bfe`。关键原始文件：

- [模型与 `forward_features`](https://github.com/OpenGVLab/VideoMamba/blob/37355c26d0ae99ca2459f6d4044a5f509031a79f/videomamba/video_sm/models/videomamba.py)：`videomamba_tiny(pretrained=False, num_classes=400, num_frames=16, img_size=224, kernel_size=1)`；192 维、24 blocks、空间 patch16、tubelet1；`forward_features(x, inference_params=None)` 返回最终归一化 CLS。官方 `pretrained=True` 使用占位 `your_model_path` 的 ImageNet 文件，**不是自动下载 K400 视频权重**。
- [官方 model zoo](https://github.com/OpenGVLab/VideoMamba/blob/37355c26d0ae99ca2459f6d4044a5f509031a79f/videomamba/video_sm/MODEL_ZOO.md)：明确 `input frames × crops × clips`；Ti16 K400 的原表为 `16×3×4`，不能把它称 FSN 单视图分数。
- [官方 K40016 训练配置](https://github.com/OpenGVLab/VideoMamba/blob/37355c26d0ae99ca2459f6d4044a5f509031a79f/videomamba/video_sm/exp/k400/videomamba_tiny/run_f16x224.sh)、[数据空间预处理](https://github.com/OpenGVLab/VideoMamba/blob/37355c26d0ae99ca2459f6d4044a5f509031a79f/videomamba/video_sm/datasets/kinetics.py)、[官方安装说明](https://github.com/OpenGVLab/VideoMamba/blob/37355c26d0ae99ca2459f6d4044a5f509031a79f/videomamba/README.md)。

权重下载链接在官方 model zoo 中已核对。直接文件入口是 [Hugging Face resolve](https://huggingface.co/OpenGVLab/VideoMamba/resolve/main/videomamba_t16_k400_f16_res224.pth)，备用为 [官方 Aliyun 链接](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_t16_k400_f16_res224.pth)。适配器不替用户下载。当前未取得该二进制文件，没有可诚实填写的发布者 SHA256。

适配器契约：

```python
from cvm.videomamba import build_videomamba_tiny16

encoder = build_videomamba_tiny16(
    repo_path="/path/to/VideoMamba",
    checkpoint_path="/path/to/videomamba_t16_k400_f16_res224.pth",
    seed=42,
)
# encoder(normalized_video[B,3,16,224,224]) -> features[B,192]
# encoder.feature_dim == 192; encoder.load_report contains provenance.
```

加载前验证外部 git HEAD、模型 blob 和被使用的代码目录无改动；检查已导入的 custom Mamba/causal-conv Python 文件与固定仓库一致。权重保持 `weights_only=True`，支持 raw tensor state dict 与一个明确的 `model`／`module`／`state_dict` 容器；只剥统一前缀。完整 400 类模型全部 key／shape 严格匹配后才删原 head。不插值时间位置、不 inflate、不默默跳过参数；8 帧、174 类 SSv2 或不完整 checkpoint 会报错。

每次加载记录本地 SHA256、路径、官方代码版本、加载 tensor 数、缺失／多余 key 和 discarded head。**兼容的 shape 与本地 SHA256 不证明权重确实经过官方训练**：须从上述官方入口取得文件，归档下载来源；load report 明确 `publisher_checksum_verified=False`。同样，源码检查不能证明已编译 CUDA 二进制的构建来源；保存构建环境和日志。

依赖需要仓库附带的 `Mamba(..., bimamba=True)` 实现和 CUDA 扩展，不能随意以最新版标准 `pip install mamba-ssm` 替换。官方安装参考 Python3.10、PyTorch2.1.1+cu118、timm0.4.12 和本仓库 `causal-conv1d`／`mamba`。实际服务器应在自己的驱动／CUDA／PyTorch组合上预检。旧 PyTorch 能读取 tensor-only 文件；含 `argparse.Namespace` 的训练 checkpoint 要求支持 `safe_globals` 的 PyTorch 或事先在可信环境导出 tensor-only 模型。不会回退到不受限制的 pickle 加载。

已在 PyTorch2.2.2+CPU 对**模拟小 encoder**完成合同测试：严格 key／shape、错误帧数和分类头、非有限权重、缺文件、提交／source 不匹配、模块导入注册以及特征梯度。该测试不是 VideoMamba 正式 forward、不是官方预训练文件验证、不是 CUDA 性能测试。正式 FSN 实验仍需真实数据和服务器执行。

服务器至少完成一次 batch1、FP32 和选定 AMP precision 的真实官方 encoder 前向＋反向，检查 `[1,192]`、loss／gradient 有限，以及一小批实际 FSN 训练与验证。成功后才启动 paired head 变体；扩展编译、权重或前向失败必须修复或如实标为未执行，不能换随机权重生成对比结果。4×3090 是资源条件，尚不提供可信的全训练时间／显存数值；先测每卡 batch 和峰值显存，再固定梯度累积与有效 batch。

## 同一组实验的公平协议

1. **任务与数据单位。**一次 forward 只接收一个 clip 的图像序列。标签、医院、文件名、来源 ID、真实 coarse group 和其他 clip 的预测均不进入 forward；只用于采样审计、训练标签、划分或报告。临床顺序用于解释 taxonomy 的来源，不能当输入时序先验。
2. **分组隔离。**同一原始视频、临床 session／病例、已确认同源转载及近重复关联为不可跨 split 单位。分别披露 source-video 隔离、已知术者隔离、机构留出；无法获得术者身份时不声称术者泛化。关联标识及不确定样本处理规则在训练前冻结。
3. **模型选择与测试。**已经反复看过以调方案的旧 test 是开发证据；新结论需要另一个未参与选择的分组留出集，或正确嵌套的 group CV。学习率／损失权重／校准只在训练内部开发集选择；最终 test 统一评估一次，不在 test 最优 checkpoint／seed 上做选择。
4. **实际取帧契约。**主协议训练／验证都在已有 `36f224` 缓存上使用 `round(linspace(0,35,16))`，没有训练分段随机／验证中心两套时间策略。扩展 R(2+1)D 的 32 帧对照把该式帧数换成 32。同一 backbone 的所有头变体共享该位置序列；训练只按同一种子规则进行整个 clip 的水平翻转。输出保存缓存契约与选择规则，**没有核验原视频 PTS、原始 frame IDs 或源 fps**。验证／测试预测导出的 `repeated_frame_fraction` 按**实际选中 N 个 RGB 缓存帧的逐像素相等性**计算 `1-unique/N`（主协议 N=16，扩展 N=32）；它不是原视频帧 ID 重复、时间间隔或运动量标注。原有 36 帧 metadata 的重复比例另存为缓存审计信息，训练 split 未额外计算选中帧重复率。短 clip／重复情况应分层报告，不能因结果差而事后删除。原视频时间审计、随机取帧或原视频更密覆盖属于后续独立实验，当前未实现。
5. **细动作时间诊断。**由 FineGym 的证据提出可检验问题：FSN 是否真正需要 clip 内运动？对冻结 checkpoint 比较原帧／静态重复／打乱同 clip 帧，记录逐类变化；这只是敏感性诊断，不证明某个时间模块的独立因果贡献。若要比较 dense 窗口与全 clip 分段覆盖，须两种都训练且同预算，同一骨干所有相关方法执行，不能只给候选更密集的输入。
6. **同骨干头变体。**每个 seed 共享同一严格加载 backbone；最大训练预算、有效 batch、优化器、精度、scheduler、增强、早停准则与数据顺序规则一致。早停导致不同方法的实际 epoch／optimizer steps 可能不同，必须报告，不声称实际训练更新量严格相同。临床组、同大小预先固定随机组、训练集构造视觉组、参数匹配 generic capacity control 按预注册对照执行。硬／软路由既报同 checkpoint 解码差异，也区分独立训练方法的差异。
7. **报告。**七类 macro-F1 为预定主指标，同时报告 accuracy、各类 F1／support、完整混淆、coarse group 误路由及组内条件错误。所有真实七类样本保留在分母。不同骨干的同组三个 seed 报 mean±SD；paired 差值与置信区间按 source/group 重采样，不能假设同一原视频切出的所有 clips 都独立。单标签 micro-F1 等于 accuracy，不当作第二个独立成果。
8. **部署与解释。**主结果始终给出一个七类预测；粗类回退／拒识另报 coverage 与风险，不计成七类正确。可视化或无框网格遮挡只能作为定性解释／区域扰动证据，不声称准确定位手／针或解释了临床技能；无需为了主研究增加人工 ROI 标注。

可写作的结论应来自以上对照：临床 taxonomy 在什么样的细动作、来源变化与路由不确定性下帮助或伤害识别；改善是否超出随机分组、额外容量和解码修复。若这些对照解释了全部收益，应撤销专门机制主张，保留诚实的基准与负面发现。先把问题及可证伪实验做实，不能以“加了较新骨干”或“预期涨点”替代论文贡献。
