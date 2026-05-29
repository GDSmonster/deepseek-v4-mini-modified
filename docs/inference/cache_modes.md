# Inference Cache 模式

`InferenceConfig.cache_mode` 用于选择 prompt 与生成 token 如何与 cache 状态交互。

## `audit`

`audit` 是保守的参考模式。

作用:
- 构建便于检查的 cache 摘要。
- 使用完整的模型 forward 以获得参考 logits。
- 适用于 MHA、HCA、CSA 以及混合 attention。

适用场景:
- 审计正确性。
- 将完整上下文 logits 与 cache 化 logits 进行对比。
- 在不要求活动 decode 行为的情况下检查生成的 cache 元数据。

## `mha_decode`

`mha_decode` 是基线 multi-head attention 的活动 KV cache 路径。

作用:
- 存储真实的逐层 MHA keys 与 values。
- 从 cache 化的 K/V 张量出发逐 token 解码。

约束:
- 仅当所有 attention 层都是基线 MHA 时有效。
- HCA/CSA 模型应改用 `deepseek_decode`。

## `deepseek_decode`

`deepseek_decode` 是面向 DeepSeek 风格 HCA、CSA 及 CSA/HCA 混合模型的活动 cache 路径。

作用:
- 存储逐层的 HCA 压缩/全局状态以及 local window。
- 存储逐层的 CSA compressed main/index 状态以及 local window。
- 在 decode 步骤中保留 MoE、mHC 与 MTP 行为。

推荐参数:

```python
InferenceConfig(
    cache_mode="deepseek_decode",
    deepseek_prefill_mode="parallel",
    cache_dtype="fp32",
    return_cache_stats=True,
)
```

## Prefill 模式

### `parallel`

`parallel` 是默认的 DeepSeek decode prefill。

它将 prompt 通过 `DeepSeekV4LM.forward(...)` 整体跑一次,在每个 attention 模块前捕获真实的归一化 attention 输入,通过每个 HCA/CSA attention 模块投影这些状态,并基于完整 prompt 构建 compressed/local/pending cache。

这是最接近实际 inference prefill 的模式,因为不需要逐 token 重放 prompt。

### `sequential_debug`

`sequential_debug` 对每个 prompt token 调用一次 `model.forward_decode(...)`。

适用场景:
- 调试逐 token 的 cache 变更过程。
- 将各层 cache 状态与活动 decode 路径对比。
- 确认后续生成的 token 不需要完整 forward 即可处理。

它有意更慢,不应作为常规生成的默认值。

## Cache 统计

设置 `return_cache_stats=True` 可返回如下字段:

- `cache_mode`
- `tokens_seen`
- `sequence_length`
- `layers_by_cache_type`
- `cache_population`
- `deepseek_active_decode`
- `num_hca_compressed_entries`
- `num_csa_compressed_main_entries`
- `num_hca_pending_tokens`
- `num_csa_pending_tokens`
- `local_window_size`
