# Scheduler

主类:`WarmupCosineLR`。

## 作用

Scheduler 在 optimizer-step 粒度上控制 learning rate。

它支持:

- 普通 PyTorch optimizer,
- 混合 `Muon + AdamW` optimizer,
- AdamW 与 Muon 各自独立的 base LR,
- 可选的独立 Muon 最小 LR,
- 通过 `state_dict` 进行 checkpoint/恢复。

## 调度形状

```text
warmup:
    从 0 线性递增到 base_lr

warmup 之后:
    从 base_lr 进行 cosine 衰减到 min_lr
```

warmup 之后的公式:

```text
progress = (step - warmup_steps) / (total_steps - warmup_steps)
lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
```

## 主要超参数

| 参数 | 含义 |
| :--- | :--- |
| `total_steps` | 本次运行预期的总 optimizer step 数。 |
| `warmup_steps` | 线性 warmup 的 step 数。 |
| `min_lr` | 普通 optimizer 或 AdamW 分支的最终/最小 LR。 |
| `min_muon_lr` | Muon 分支的可选最终/最小 LR。 |

## 运行时方法

| 方法 | 含义 |
| :--- | :--- |
| `step()` | 推进一个 scheduler step 并更新 optimizer LR。 |
| `set_step(step)` | 将 scheduler 设置到指定 step,在恢复时有用。 |
| `get_last_lr()` | 返回当前 LR 列表。 |
| `get_lr_dict()` | 便于日志的 LR 字典。 |
| `state_dict()` | 可序列化的状态。 |
| `load_state_dict(state)` | 恢复状态与 LR 值。 |

## 推荐设置

CPU 小规模冒烟测试:

```yaml
total_steps: 2
warmup_steps: 1
min_learning_rate: 0.00003
```

Mini 训练:

```yaml
warmup_steps: 500
learning_rate: 0.0003
min_learning_rate: 0.00003
```

## 注意事项

- 调用顺序应为先 `optimizer.step()`,再 `scheduler.step()`。
- `total_steps` 应按 optimizer step 计数,而非原始 batch 数。
- 启用 gradient accumulation 时,optimizer step 数会少于 dataloader batch 数。
