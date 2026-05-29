# Attention 配置参考

本文件记录 MHA、HCA、CSA 以及 hybrid attention 使用的 attention 超参数。

## 共享 Attention 参数

| Parameter | Used by | Description |
| :--- | :--- | :--- |
| `d_model` | all | 进入 attention 的隐藏层大小。 |
| `n_heads` | all | query head 数量。head 越多可建模的 attention 模式越多。 |
| `head_dim` | all | 每个 attention head 的大小。当前 CSA/HCA 要求 `n_heads * head_dim == d_model`。 |
| `attention_dropout` | all | 作用于 attention 概率的 dropout。 |
| `residual_dropout` | all | attention 输出投影之后的 dropout。 |
| `use_bias` / `use_attention_bias` | all | 启用投影 bias。 |
| `use_rope` | all | 启用旋转位置编码 (RoPE)。 |
| `rope_theta` | all | RoPE 基础频率。 |
| `rotary_dim` | all | 由 RoPE 旋转的 head 维度数量。 |
| `max_seq_len` | all | 用于缓冲区/校验的最大序列长度。 |
| `init_std` | all | 权重初始化标准差。 |

## HCA 专用参数

| Parameter | Description |
| :--- | :--- |
| `compression_factor` / `hca_compression_factor` | 被压缩为单个 KV 条目的 token 数量。值越大计算越省,但细节越少。 |
| `window_size` | 在压缩全局记忆之外加入的精确局部上下文长度。 |
| `use_attention_sink` | 添加可学习的 sink 条目以吸收 attention 质量。 |
| `use_grouped_output_projection` | 在投影前将 head 输出分组。 |
| `output_projection_groups` | 分组数量;必须能整除 `n_heads`。 |

## CSA 专用参数

| Parameter | Description |
| :--- | :--- |
| `compression_factor` | 每个压缩块包含的 token 数量。 |
| `top_k` / `top_k_blocks` | 每个 query 选取的压缩块数量。 |
| `window_size` | 附加在所选稀疏全局块之后的精确局部窗口长度。 |
| `indexer_dim` | indexer key/query 向量的大小。 |
| `n_indexer_heads` | 稀疏选择器中的打分 head 数量。 |
| `query_compression_dim` | 在生成 indexer/core query 之前的低秩 query 瓶颈。 |
| `use_indexer_score_bias` | 让 indexer 分数影响核心 attention logits。 |
| `use_separate_local_kv` | 保持局部分支投影独立于压缩全局分支。 |
| `use_attention_sink` | 添加可学习的全局 sink 条目。 |
| `use_grouped_output_projection` | 在 attention 之后启用分组投影。 |
| `output_projection_groups` | 输出投影分组数量。 |

## Hybrid Attention 参数

| Parameter | Description |
| :--- | :--- |
| `attention_type="hybrid"` | 启用按层调度的 attention。 |
| `attention_pattern` | 在各层间循环使用的元组/列表,例如 `[csa, hca]`。 |

示例:

```yaml
attention_type: hybrid
attention_pattern: [csa, hca]
```
