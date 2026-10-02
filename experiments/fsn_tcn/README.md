# 最小 FSN-TCN：视觉关系判断与 TCN 内部断链

本实现回答一个可证伪的问题：在同一冻结 A 的合法候选链内，按视觉关系切断连接，能否比普通 A+TCN 少改错，同时保留原始临床链的收益。只有一阶段 TCN、普通七类 CE；没有新增多阶段、联合训练或辅助损失。

## 已实现什么

1. 保留已有封存的 train7372 / val823、A 特征、合法候选链、来源映射；不重训 A、不重切数据、不读取 test。
2. R-Dual 先训练。原临床候选对是来源政策正例，原网络候选对是来源政策负例；另加入同一 `source_collection` 内、不同 recording/group/video 的临床训练难负例。难负例不按动作标签挑选，不进入正式 TCN 链。
3. R best 按原 val 候选对的未加权政策 BCE 选择。阈值写在实验配置中，默认0.9；不搜阈值、不拿合成验证挑战选 checkpoint。阈值不是校准后的连续概率。
4. R 固定以后，四组 TCN 从同一种子的相同初始权重分别训练：`all`、`source_rule`、`random`、`learned`。所有组共用同一配置。每组内部的所有切段共用一套 TCN 权重；四组之间是独立训练的模型。
5. 切段发生在 TCN forward 之前。每小段独立计算，空洞卷积没有跨断点的特征；单独一个 clip 严格返回冻结 A 的 logits。
6. 随机门每条链、每种来源匹配 learned 的开边数量，只随机改变位置；结构 `eligible=False` 永远关闭。
7. 原 val823 和固定合成挑战分别评测：同来源临床跨记录拼接、同一临床记录顺序打乱。所有片段内容、标签不变，每个 val clip 恰好出现一次。合成跨记录只在明确声明的挑战链中出现，不是原数据候选连接。
8. 每个训练 checkpoint 另做同权重切换门控审计。它与四组独立训练比较是两种不同证据。

原始链间隔仍固定为5秒、非重叠且不桥接原结构断点。片段的标注时间和缓存 token 位置不是已核验的原视频帧 PTS。

## 对旧 A+TCN 的兼容边界

用户引用的三个本地文件 `tcn_baseline.py`、`hard_dual.py`、`evaluate_robustness.py` 没有出现在本次可访问的 GitHub 源版本。因此本包没有覆盖它们，也没有声称重新实现了相同历史 TCN。

`SegmentedTCN` 是通用包装器，可以复用符合下面契约的旧模型：

```python
class CompatibleTCN(torch.nn.Module):
    def forward(self, features, a_logits):
        # features: [T, global_dim + local_dim]
        # a_logits: [T, 7]
        # return: [T, 7] logits
        ...
```

输入特征在本参考实现中是 A 的全局 token 均值与局部 token 均值拼接。若历史代码使用不同特征处理，适配器必须恢复该处理；不要把一个不同的输入表示称为旧 baseline 复现。

旧 TCN 必须是无跨调用隐藏状态的时间模型。全接的长度≥2段会直接调用原 forward；为满足本方案，旧模型对 singleton 也必须已返回 A。若旧 singleton 行为不同，CUDA 预检会阻断，并报告无法同时满足“历史完全复现”和“孤立片段回 A”，不会静默改动旧模型。

外部配置示意（`kwargs`、优化器设置及特征适配需依据实际历史代码核对）：

```json
"baseline": {
  "kind": "external",
  "factory": "experiments.your_verified_adapter:build_tcn",
  "kwargs": {"width": 64},
  "source_dependencies": ["/absolute/path/to/historical/tcn_baseline.py"]
}
```

factory 接收 `input_dim=...` 并返回上面的两输入 scorer。其源码、明确列出的依赖、TCN checkpoint、A 特征与配置均校验 SHA。factory 能运行不等于历史复现已验证：需要用实际旧模型、同一权重与同一输入做配对 parity，记录差值，并核对原训练超参数。

`minimal_reference.json` 明确选择**新的参考基线**，不是旧模型复现：宽度64、4层、膨胀1/2/4/8、kernel3、dropout0.1、无 BatchNorm；最后残差投影零初始化。训练为 AdamW/普通未加权 CE，最多30epoch/patience5、每步4条未填充链。四组均采用这套设置。若要沿用旧实验，先核对兼容性和训练配置；不能拿新 reference 分数冒充旧 A+TCN。

## 训练时的一个重要差别

单 clip 必须严格回到 A，所以它没有可训练 TCN 梯度。每个训练 batch 的 CE 分母包含所有 clip；全是孤立片段的 batch 跳过 backward/optimizer，并记录次数。各组报告可训练 clip visits 和实际优化步数。这种差异应在结果解释中列出。

若 learned 固定门完全关闭，训练会报告没有可训练连接并停止，禁止降低阈值或偷偷添加边。原 train/val、旧 best/log/结果不改写。

## 四卡运行

依赖：已有 Original CUDA 环境的 torch/torchvision、numpy，以及 pytest。没有 DDP。GPU0先做原缓存 A/R 检查和 TCN 检查，再训练该种子的 R；随后四个独立 TCN 分别占用 GPU0–3，全部完成后评测。

先在独立、干净、已提交的 worktree 检查并只生成计划：

```bash
/root/miniconda3/bin/python -m experiments.fsn_tcn.run_suite \
  --feature-index /root/autodl-tmp/formal_source_relation_suite_7372_823_seed42_v1/features/index.jsonl \
  --source-protocol-dir /root/autodl-tmp/formal_source_relation_suite_7372_823_seed42_v1/source_protocol \
  --manifest-dir /absolute/path/to/full_7372_823_manifests \
  --cache-root /root/autodl-tmp/full_cache_36f224_trainval \
  --original-checkpoint /absolute/path/to/the_same_completed_Original/best.pt \
  --config /absolute/path/to/FSN_REC/experiments/fsn_tcn/minimal_reference.json \
  --output /root/autodl-tmp/formal_fsn_tcn_minimal_7372_823_v1 \
  --gpus 0,1,2,3
```

实际 manifest/A 路径由已有 export protocol 核实，不猜测；SHA不符会阻断。上面命令默认只读、不写输出、不启动 GPU。核对配置后，同一命令加 `--execute --expected-commit <完整HEAD SHA>` 才启动；可以自行用 nohup 后台运行。

启动器检查四 GPU 空闲与物理 UUID 映射、干净代码、全新输出、原清单及 checkpoint。预检顺序：55+项 CPU检查（以实际版本为准）→真实原缓存 CUDA A/R smoke→真实冻结证据 TCN 前向/反向/三步参数更新与断链检查→R→四组 TCN→统一原始/扰动评测。

预检会检查非零权重下的断链隔离，避免零初始化的残差头掩盖泄漏。失败保留现场，不自动重试、不覆盖、不启动旧任务、不修改运行中的代码。

默认配置是单个 seed42 的最小首轮。若从一开始计划多种子，启动前在一份新的固定配置中列出全部种子，例如 `[42, 3407, 2026]`；启动器只执行这份显式列表，并全部交付。A 始终是同一个 checkpoint，所以这叫固定 A 下 R/TCN 的种子稳定性，不是整个视觉系统的多种子稳定性。

## 输出与解释

私有输出：`run_protocol.json`、`frozen_config.json`、`.launch_once`、`run_commit.txt`；`launcher_logs` 含日志、pids和退出码；每个 `seed_N/relation` 含训练专用难负例、R的best/last/history/scores/result；四组分别有TCN的best/last/history/result/逐clip预测；evaluation保存合成链、关系分数、配对预测及四组结果。

只有 `public_safe_all_seeds.json` 与每seed的 `evaluation/public_safe_summary.json` 是可供人工审查后发布的聚合。包括总体/来源/时长及七类P/R/F1/support、扫散↔再灌注双向错分、改对/改错、原始临床开边率、人为断点误开率、参数和训练时间。不得上传视频、标签清单、含身份路径日志、特征、权重或逐clip预测。代码不会自动上传实验结果。

val823持续用于开发和选择 checkpoint，所有结果标记开发验证。合成拼接只能证明对指定构造断点的稳健性，不代表真实剪辑检测性能。原临床正例仍是来源政策假设，网络负例不是人工核验的剪辑GT。若 R-TCN 只超过 A、不超过普通 TCN或匹配开边数的随机控制，不能说学到了有用的断链位置。

CPU测试可验证函数、梯度和安全协议；当前工作环境没有CUDA或服务器数据，不能把这里的测试写成服务器训练已启动或已完成。
