# 指标与诊断

训练栈拥有两层指标:

1. 标准的 LM 训练/评估指标,
2. 针对 MoE、mHC、MTP、CSA/HCA 与 loss 健康度的 DeepSeek 模块诊断指标。

## LM 指标

主要实现于:

```text
training/training_metrics.py
training/loss_metrics.py
```

核心指标:

- loss,
- perplexity,
- token accuracy,
- top-k accuracy,
- entropy,
- 有效 token 数。

作用:

- 跟踪常规语言建模进展,
- 适用于 train/eval 输出,
- 支持忽略 label 与 pad mask。

## 模块诊断

实现分布于:

```text
training/full_deepseek_metrics.py
training/moe_metrics.py
training/mhc_metrics.py
training/mtp_metrics.py
training/deepseek_modules_metrics_utils.py
```

作用:

- 暴露在标量 LM loss 中不可见的内部行为,
- 帮助诊断 routing 坍塌、不稳定的 mHC 矩阵、MTP 行为以及 auxiliary loss,
- 同时支持紧凑的关键 top 视图与更详细的诊断视图。

## MoE 诊断

典型信号:

- router entropy,
- expert 负载分布,
- 被选中的 expert 比例,
- 活跃 expert 数,
- balance loss,
- 序列层面的不均衡度。

为何重要:

- MoE 可能悄然坍塌为少数几个 expert。
- loss 可能继续改善,而 routing 质量已经变差。
- expert 负载平衡对稀疏容量至关重要。

## mHC 诊断

典型信号:

- A/B/C 矩阵统计量,
- Sinkhorn 行/列和,
- alpha gate 取值,
- stream 混合行为。

为何重要:

- mHC 数值敏感。
- B 应表现为双随机的 residual mixing 矩阵。
- alpha gate 在训练早期不应爆炸。

## MTP 诊断

典型信号:

- 原始 MTP loss,
- 加权 MTP loss,
- 各 depth 的 loss,
- depth 权重。

为何重要:

- MTP 可提升表示质量,但权重过大也可能主导训练。

## 日志控制

| 参数 | 含义 |
| :--- | :--- |
| `module_metrics_every` | 每 N 个 optimizer step 收集一次诊断。 |
| `print_module_diagnostics` | 将诊断打印到控制台。 |
| `verbose` | `1` 完整诊断,`0` 紧凑的关键诊断。 |
| `log_grad_norm` | 在 train 统计中包含 gradient norm。 |
| `log_mem` | 打印 CUDA 内存统计。 |
| `metrics_jsonl_name` | JSONL 指标文件名。 |

## 实用建议

- 为追求最快的 CPU 冒烟测试,关闭或稀疏化诊断。
- 测试 MoE/mHC/CSA 改动时打开诊断。
- 仅关注高风险信号时使用 `verbose=0`。
- 使用 JSONL 指标方便后续绘图分析。
