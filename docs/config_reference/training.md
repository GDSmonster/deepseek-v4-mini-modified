# 训练配置参考

主要入口:`train_deepseekv4`。

## 设备与精度

| Parameter | Description |
| :--- | :--- |
| `seed` | 全局随机种子。 |
| `deterministic` | 在可能的情况下启用 PyTorch 的确定性行为。 |
| `device` | `auto`、`cpu`、`cuda` 或其他 torch 设备字符串。 |
| `amp_enabled` | 在支持时启用自动混合精度。 |
| `amp_dtype` | 请求的 AMP dtype:`bf16`、`fp16` 或 `fp32`。 |
| `fallback_bf16_to_fp16` | 当 CUDA 不支持 bf16 时回退到 fp16。 |

## Epoch 与 Step 控制

| Parameter | Description |
| :--- | :--- |
| `epochs` | epoch 数量。 |
| `start_epoch` | 恢复训练时的起始 epoch。 |
| `global_step` | 起始 optimizer step。 |
| `grad_clip` | 最大梯度范数。设为 `None` 可禁用。 |
| `grad_accum_steps` | 每个 optimizer step 累积的微批次数。 |
| `max_batches_per_epoch` | 限制每个 epoch 的批次数量,用于冒烟测试。 |
| `log_every` | 以 optimizer step 为单位的打印间隔。 |
| `on_oom` | OOM 行为。`skip` 会跳过 CUDA OOM 批次。 |

## 模块诊断

| Parameter | Description |
| :--- | :--- |
| `module_metrics_every` | 每 N 个 optimizer step 计算一次模块诊断。`0` 表示训练 epoch 结束后计算一次。 |
| `print_module_diagnostics` | 打印诊断表。 |
| `verbose` | `1` 输出完整诊断,`0` 仅输出关键指标。 |
| `log_grad_norm` | 跟踪梯度范数。 |
| `log_mem` | 打印 CUDA 显存统计。 |

## 评估

| Parameter | Description |
| :--- | :--- |
| `eval_every` | 每 N 个 epoch 评估一次。 |
| `eval_max_batches` | 限制验证批次数量。 |
| `eval_use_ema` | 若可用,则评估 EMA 权重。 |
| `eval_log_every` | 可选的评估日志间隔。 |
| `eval_preview` | 打印定性预览。 |
| `eval_preview_batch_idx` | 用于预览的验证批次索引。 |
| `eval_preview_sample_idx` | 预览批次内的样本索引。 |
| `eval_preview_max_context_tokens` | 显示的上下文 token 数量。 |
| `eval_preview_max_new_tokens` | 生成的新 token 数量。 |
| `eval_preview_temperature` | 0 表示贪心生成,>0 表示采样。 |
| `tokenizer` / `id2tok_fn` | 用于解码预览。 |
