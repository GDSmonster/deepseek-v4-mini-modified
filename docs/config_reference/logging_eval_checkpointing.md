# 日志、评估与 Checkpointing

## 日志

| Parameter | Description |
| :--- | :--- |
| `log_every` | 每 N 个 optimizer step 打印一行训练记录。 |
| `log_grad_norm` | 在训练统计中包含梯度范数。 |
| `log_mem` | 打印 CUDA 显存统计。 |
| `module_metrics_every` | 每 N 步收集一次 DeepSeek 模块诊断。 |
| `print_module_diagnostics` | 将模块诊断打印到控制台。 |
| `verbose` | 控制诊断的详细程度。 |
| `metrics_jsonl_name` | checkpoint 目录内 JSONL 指标文件的文件名。 |

## 评估预览

| Parameter | Description |
| :--- | :--- |
| `eval_preview` | 启用定性预览。 |
| `eval_preview_batch_idx` | 用于预览的批次索引。 |
| `eval_preview_sample_idx` | 批次内的样本索引。 |
| `eval_preview_max_context_tokens` | 显示的输入/上下文 token 数量。 |
| `eval_preview_max_new_tokens` | 自回归生成的 token 数量。 |
| `eval_preview_temperature` | 0 表示贪心,>0 启用采样。 |
| `tokenizer` | 可选的具有 `.decode` 的 tokenizer。 |
| `id2tok_fn` | 可选的自定义 id 到文本函数。 |

## Checkpointing

| Parameter | Description |
| :--- | :--- |
| `ckpt_dir` | 写入 checkpoint 与指标的目录。 |
| `run_name` | 易读的运行名称。 |
| `save_every` | 每 N 个 epoch 保存一次。 |
| `save_last` | 保存/更新最新 checkpoint。 |
| `keep_last_n_checkpoints` | 删除超出 N 个的更早 step checkpoint。 |
| `monitor_name` | 用于选择最佳 checkpoint 的指标。 |
| `monitor_mode` | `min` 或 `max`。 |
| `best_metric` | 恢复训练时的初始最佳指标。 |
| `resume_path` | 用于恢复的 checkpoint 路径。 |
| `strict_resume` | 严格的模型 state 加载。 |
| `restore_rng_state` | 恢复 Python/NumPy/PyTorch 的 RNG 状态。 |

## Drive 镜像

这些参数适用于 notebook/Colab 工作流:

| Parameter | Description |
| :--- | :--- |
| `drive_ckpt_dir` | 可选的外部 checkpoint 镜像目录。 |
| `copy_fixed_to_drive` | 将最新 checkpoint 复制到固定文件名。 |
| `fixed_drive_name` | 固定的 checkpoint 文件名。 |
| `fixed_drive_metrics_name` | 固定的指标文件名。 |
