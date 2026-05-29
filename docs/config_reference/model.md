# 模型配置参考

主要类:`DeepSeekV4LMConfig`。

该配置控制完整的模型:embedding、block 数量、attention 选择、FFN/MoE、mHC、MTP、损失行为以及初始化。

## 核心形状

| Parameter | Description |
| :--- | :--- |
| `vocab_size` | 分词器词表条目数量。必须 > 0。 |
| `d_model` | 模型的隐藏层大小。所有 block 输入/输出均使用该宽度。 |
| `n_layers` | `DeepSeekV4Block` 层数。 |
| `max_seq_len` | 模型可接受的最大序列长度。 |
| `pad_token_id` | 可选的分词器 pad id。用于 attention mask 与 loss mask。 |
| `ignore_index` | 交叉熵忽略的标签值。默认 `-100`。 |

## 损失语义

| Parameter | Description |
| :--- | :--- |
| `labels_are_shifted` | 若为 true,标签已表示下一 token 目标;若为 false,标签会在内部进行 shift。 |
| `ignore_pad_token_in_loss` | 当设置 `pad_token_id` 时,在 loss 中屏蔽 pad token。 |

## Embedding 与 Norm

| Parameter | Description |
| :--- | :--- |
| `embedding_dropout` | token embedding 之后的 dropout。 |
| `scale_embeddings` | 启用时按隐藏层大小约定对 embedding 进行缩放。 |
| `tie_word_embeddings` | 共享 LM head 权重与 token embedding 权重。 |
| `rms_norm_eps` | RMSNorm 层的 epsilon。 |
| `init_std` | 默认正态初始化标准差。 |

## Attention 选择

| Parameter | Description |
| :--- | :--- |
| `attention_type` | `mha`、`hca`、`csa`、`hybrid` 之一。 |
| `attention_pattern` | 当 `attention_type="hybrid"` 时使用的层循环模式。 |
| `n_heads` | attention query head 数量。 |
| `head_dim` | 每个 head 的维度。若未设置,则由 `d_model / n_heads` 推导。 |
| `attention_dropout` | 作用于 attention 权重的 dropout。 |
| `residual_dropout` | 在残差融合前作用于 attention/FFN 输出的 dropout。 |
| `use_attention_bias` | 启用 attention 投影的 bias。 |
| `use_rope` | 启用旋转位置编码 (RoPE)。 |
| `rope_theta` | RoPE 频率基数。 |
| `rotary_dim` | 使用 RoPE 的 head 维度数量。必须为偶数且 <= `head_dim`。 |

## HCA/CSA 共享控制项

| Parameter | Description |
| :--- | :--- |
| `compression_factor` | CSA 压缩块大小。 |
| `hca_compression_factor` | HCA 压缩块大小。 |
| `window_size` | 精确局部滑动窗口的上下文长度。 |

## CSA 控制项

| Parameter | Description |
| :--- | :--- |
| `top_k_blocks` | CSA indexer 选取的压缩块数量。 |
| `indexer_dim` | 压缩 indexer key/query 向量的维度。 |
| `n_indexer_heads` | CSA indexer 使用的 head 数量。 |
| `query_compression_dim` | 可选的低秩 query 瓶颈维度。 |
| `use_attention_sink` | 添加可学习的 sink KV 条目。 |
| `use_grouped_output_projection` | 启用分组的 attention 输出投影。 |
| `output_projection_groups` | 投影分组数。必须能整除 `n_heads`。 |
| `use_indexer_score_bias` | 将 indexer 分数信号加入到核心 attention logits 中。 |
| `use_separate_local_kv` | 使用独立的精确局部 KV 分支。 |

## FFN 选择

| Parameter | Description |
| :--- | :--- |
| `ffn_type` | `dense` 或 `moe`。 |
| `mlp_hidden_dim` | dense FFN 隐藏层大小。 |
| `mlp_expansion_factor` | 当未显式给出隐藏维度时使用的 dense FFN 隐藏倍率。 |
| `mlp_multiple_of` | 将 dense 隐藏维度向上取整到该倍数。 |
| `mlp_dropout` | dense FFN 的 dropout。 |
| `use_mlp_bias` | 启用 dense FFN 及相关投影的 bias。 |

## MoE、mHC 与 MTP

这些配置组单独记录:

- [MoE 配置](moe.md)
- [mHC 配置](mhc.md)
- [MTP 配置](mtp.md)
