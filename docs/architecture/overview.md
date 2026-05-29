# 架构概览

DeepSeek-V4 Mini 是一个可配置的因果语言模型,既保留了 DeepSeek-V4 论文中重要的架构思想,又足够小巧,可在 CPU 上检视与测试。

模型围绕 `DeepSeekV4LM` 与 `DeepSeekV4Block` 构建。

## 高层流程

```text
input_ids
  -> TokenEmbedding
  -> DeepSeekV4Block x n_layers
       -> attention 分支:MHA / HCA / CSA / 混合调度
       -> 前馈分支:dense SwiGLU / DeepSeekMoE
       -> 可选的 mHC 残差流,环绕 attention 与 FFN
  -> 最终 RMSNorm
  -> LM head
  -> 可选的 MTP heads
```

## 主要模块

### `DeepSeekV4LM`

完整的语言模型封装。

作用:

- 拥有 embeddings、blocks、最终 norm、LM head 以及可选的 MTP head。
- 计算因果 LM 损失。
- 通过配置开关支持 dense、MoE、HCA、CSA、混合 attention、mHC 与 MTP。

### `DeepSeekV4Block`

重复堆叠的 Transformer block。

作用:

- 按层选择 attention 类型。
- 选择 dense FFN 或 MoE。
- 用经典残差或 mHC 残差流封装 attention 与 FFN。
- 在需要时收集来自 MoE、mHC、CSA、HCA 与 MTP 的辅助输出。

## 重要设计选择

- 本项目偏好可读的 PyTorch 代码,而非自定义 kernel。
- 长上下文机制以教学方式实现,并非生产级 kernel。
- CPU 测试使用极小配置,但同样的配置接口可扩展到更大的实验。
- 论文中的工业化系统(如融合 MoE kernel、FP4 QAT、专家并行、磁盘 KV cache)有意不在本 mini 仓库范围内。

## 核心超参数分组

- 模型规模:`vocab_size`、`d_model`、`n_layers`、`max_seq_len`。
- Attention:`attention_type`、`n_heads`、`head_dim`、`compression_factor`、`top_k_blocks`、`window_size`。
- FFN/MoE:`ffn_type`、`mlp_hidden_dim`、`num_experts`、`top_k_experts`、`router_type`。
- mHC:`use_mhc`、`n_hc`、`mhc_sinkhorn_iters`。
- MTP:`use_mtp`、`mtp_depth`、`mtp_loss_weight`。
- 训练:优化器、LR 调度、AMP、EMA、checkpointing、日志、诊断。
