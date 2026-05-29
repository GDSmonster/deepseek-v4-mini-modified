# 训练流水线

训练栈以 `train_deepseekv4` 为核心。

## 整体流程

```text
设置随机种子
解析 device 与精度
构建 optimizer:AdamW 或 Muon+AdamW
构建 warmup cosine scheduler
可选地创建 EMA
可选地从 checkpoint 恢复

for epoch:
    train_one_epoch
    可选地计算模块诊断指标
    可选地 eval_one_epoch
    保存 checkpoints
    追加 metrics JSONL
```

## `train_one_epoch`

功能:

- 将 dataloader 的 batch 规范化为模型 kwargs。
- 将张量移动到目标 device。
- 启用时应用 AMP/autocast。
- 计算模型 loss。
- 处理 gradient accumulation。
- 对 gradient 进行裁剪。
- 推进 optimizer、scheduler 与 EMA。
- 记录 loss、grad norm、LR 以及可选的模块诊断指标。

重要控制项:

- `grad_accum_steps`:每次 optimizer step 的 microbatch 数量。
- `grad_clip`:最大 gradient norm。
- `max_batches_per_epoch`:为冒烟测试限制工作量。
- `module_metrics_every`:控制模块诊断指标的计算频率。
- `on_oom`:目前支持跳过 CUDA OOM 的 batch。

## `eval_one_epoch`

功能:

- 计算 LM 指标:loss、perplexity、accuracy、top-k accuracy、entropy。
- 支持 EMA 评估。
- 可打印定性的 teacher-forced 与自回归预览。

重要控制项:

- `eval_every`:评估的 epoch 间隔。
- `eval_max_batches`:限制验证工作量。
- `eval_use_ema`:在可用时评估 EMA 权重。
- `eval_preview`:启用定性预览。
- `eval_preview_max_context_tokens`:预览中显示的上下文长度。
- `eval_preview_max_new_tokens`:生成续写的长度。
- `eval_preview_temperature`:为 0 时贪心,大于 0 时采样。

## Checkpoints 与 Metrics

Checkpoint 可包含:

- 模型 state
- optimizer state
- scheduler state
- scaler state
- EMA state
- RNG state
- 配置快照
- 额外的运行元数据

Metrics 以 JSONL 形式追加,便于检查与绘图。
