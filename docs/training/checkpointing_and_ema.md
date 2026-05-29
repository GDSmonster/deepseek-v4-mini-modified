# Checkpointing 与 EMA

Checkpointing 与 EMA 属于训练基础设施,而不是架构本身,但它们对实验的可复现性至关重要。

## Checkpointing

实现:

```text
training/chekpoints.py
```

是的,该文件目前命名为 `chekpoints.py`。

## Checkpoint 可存储的内容

- 模型 state,
- optimizer state,
- scheduler state,
- grad scaler state,
- EMA state,
- epoch,
- step,
- 最佳指标,
- 配置快照,
- 额外元数据,
- RNG state。

## 关键函数

| 函数 | 作用 |
| :--- | :--- |
| `save_checkpoint` | 写入一个 checkpoint 与附属元数据。 |
| `load_checkpoint` | 恢复模型/训练状态。 |
| `cleanup_old_checkpoints` | 仅保留最近 N 个 step checkpoint。 |

## 重要超参数

| 参数 | 含义 |
| :--- | :--- |
| `ckpt_dir` | Checkpoint 目录。 |
| `run_name` | 存入元数据的运行名。 |
| `save_every` | 每 N 个 epoch 保存一次。 |
| `save_last` | 维护最新的 checkpoint。 |
| `keep_last_n_checkpoints` | 删除较旧的 step checkpoint。 |
| `monitor_name` | 用于选择最佳 checkpoint 的指标名。 |
| `monitor_mode` | `min` 或 `max`。 |
| `resume_path` | 用于恢复的 checkpoint 路径。 |
| `strict_resume` | 严格加载模型 state。 |
| `restore_rng_state` | 恢复 RNG 状态以增强可复现性。 |

## 原子保存

Checkpoint 写入会先使用临时路径,然后再重命名到最终位置。这降低了进程在保存中途中断时损坏最终 checkpoint 的概率。

## EMA

实现:

```text
training/ema.py
```

EMA 指模型权重的 exponential moving average(指数滑动平均)。

作用:

- 维护平滑后的模型权重,
- 可改善评估稳定性,
- 可通过 `eval_use_ema` 单独评估。

## EMA 超参数

| 参数 | 含义 |
| :--- | :--- |
| `use_ema` | 启用 EMA 跟踪。 |
| `ema_decay` | 衰减系数。值越大更新越慢。 |
| `ema_device` | EMA 影子权重所在 device,通常为 `cpu`。 |
| `ema_update_after_step` | EMA 开始之前的延迟。 |
| `ema_update_every` | 以 optimizer step 为单位的更新间隔。 |
| `eval_use_ema` | 评估时使用 EMA 权重。 |

## 实用建议

- 小规模冒烟测试中保持 EMA 关闭。
- 较长的运行中,在训练循环稳定后再启用 EMA。
- GPU 显存紧张时可将 EMA 存放在 CPU 上。
