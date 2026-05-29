# Optimizer 与 Scheduler 配置参考

训练栈支持:

- `adamw`
- `muon_adamw`

## AdamW 参数

| Parameter | Description |
| :--- | :--- |
| `learning_rate` | AdamW 的基础 learning rate。 |
| `min_learning_rate` | cosine schedule 的最终/最小 LR。 |
| `weight_decay` | AdamW 解耦的 weight decay,作用于 decay 分组。 |
| `betas` | AdamW 的 beta 系数。 |
| `eps` | AdamW 的 epsilon。 |

参数分组:

- decay 组:类矩阵的可训练权重。
- no-decay 组:bias、norm、embedding、LM head、标量/向量参数、mHC 门控/静态参数以及小型控制参数。

## Muon + AdamW 参数

| Parameter | Description |
| :--- | :--- |
| `optimizer_type="muon_adamw"` | 启用混合 optimizer。 |
| `muon_lr` | Muon 参数的可选独立 LR。默认值为 `learning_rate`。 |
| `muon_momentum` | Muon 动量系数。 |
| `muon_nesterov` | 使用 Nesterov 风格的动量更新。 |
| `muon_ns_steps` | Newton-Schulz 正交化迭代次数。 |
| `muon_eps` | Muon 数值 epsilon。 |
| `muon_weight_decay` | Muon 分组的解耦 weight decay。 |
| `min_muon_lr` | Muon schedule 的可选独立最小 LR。 |

Muon 用于二维矩阵参数,不适用于 embedding、LM head、norm、bias、标量/向量参数或 mHC 的小参数。

## Warmup Cosine Scheduler

主要类:`WarmupCosineLR`。

| Parameter | Description |
| :--- | :--- |
| `total_steps` | schedule 的总 optimizer step 数量。 |
| `warmup_steps` | 线性 warmup 步数。 |
| `min_lr` / `min_learning_rate` | cosine 衰减的下界 LR。 |
| `min_muon_lr` | 可选的 Muon 专用 LR 下界。 |

行为:

```text
step <= warmup_steps:
    lr increases linearly to base LR

step > warmup_steps:
    lr follows cosine decay to min LR
```

推荐的极小起步配置:

```yaml
optimizer_type: adamw
learning_rate: 0.0003
min_learning_rate: 0.00003
weight_decay: 0.1
warmup_steps: 1
```
