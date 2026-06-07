# DeepSeek-V4 Mini 文档

本目录在配置实验所需的层面上描述了模型、训练栈、数据流水线及 CLI,使你无需先通读所有源代码即可上手。

文档力求实用:

- 对每个模块的简要说明。
- 它在整体架构中扮演的角色。
- 可配置的超参数有哪些。
- 每个超参数会改变什么。
- 关于安全的极简/CPU 设置与较大研究设置的备注。

## 架构

- [架构总览](architecture/overview.md)
- [Attention 模块:MHA、HCA、CSA](architecture/attention_modules.md)
- [Linear Attention 变体](architecture/linear_attention.md)
- [HCA:Heavily Compressed Attention](architecture/hca.md)
- [CSA:Compressed Sparse Attention](architecture/csa.md)
- [MoE 与 Dense FFN](architecture/moe_and_ffn.md)
- [mHC 残差流](architecture/mhc.md)
- [MTP 辅助预测](architecture/mtp.md)

## 训练系统

- [训练流水线](training/pipeline.md)
- [Autocast 与精度](training/autocast_and_precision.md)
- [Scheduler](training/scheduler.md)
- [Muon 优化器](training/muon.md)
- [指标与诊断](training/metrics.md)
- [Checkpoint 与 EMA](training/checkpointing_and_ema.md)

## Inference 系统

- [Inference 总览](inference/overview.md)
- [Inference Cache 模式](inference/cache_modes.md)
- [HCA 与 CSA KV cache 机制](inference/kv_cache.md)

## 配置参考

- [模型 Config 参考](config_reference/model.md)
- [Attention Config 参考](config_reference/attention.md)
- [MoE 与 FFN Config 参考](config_reference/moe.md)
- [mHC Config 参考](config_reference/mhc.md)
- [MTP Config 参考](config_reference/mtp.md)
- [训练 Config 参考](config_reference/training.md)
- [Optimizer 与 Scheduler Config 参考](config_reference/optimizer.md)
- [数据 Config 参考](config_reference/data.md)
- [日志、评估与 Checkpoint](config_reference/logging_eval_checkpointing.md)

## 运维文档

- [CLI 参考](cli/reference.md)
- [数据集指南](data/datasets.md)
- [并行化指南](parallel/overview.md)
