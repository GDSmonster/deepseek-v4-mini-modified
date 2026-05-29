# HCA 与 CSA KV cache 机制

活动的 DeepSeek cache 并不是标准的 Transformer KV cache。HCA 与 CSA 会对若干 hidden state 块进行压缩,保留一段 local sliding window,并保存尚未填满压缩块的 pending token。

## 共同形态

在 prefill 时,每个 block 捕获:

```text
x_norm = block.norm1(hidden_states)
```

该张量与 attention 模块所使用的归一化 attention 输入相同。Cache 构建器随后请求 attention 模块从 `x_norm` 投影出 cache 状态。

## HCA Cache

HCA 存储:

- `compressed_kv`:每个已完成 block 一个压缩向量。
- `compressed_positions`:每个压缩 block 的位置 id,使用该 block 的最后一个 token。
- `compressed_valid_mask`:每个压缩 block 是否至少包含一个有效 token。
- `pending_c`、`pending_z`:尚未达到 `hca_compression_factor` 的尾部状态。
- `local_c`:用于 sliding local 分支的近期 token 状态。

最关键的配置项:

- `hca_compression_factor`:被汇总进单个全局压缩条目的 token 数量。
- `window_size`:local sliding-window 长度。
- `use_attention_sink`:增加一条可学习的 sink key/value 分支。
- `use_rope`:根据存储的位置对 query 与 cache keys 应用 RoPE。

在 decode 阶段,HCA 将新 token 投影为 `C` 与 `Z`,追加到 pending/local cache,在 pending token 足够多时落盘一个压缩 block,并在 sink、压缩全局记忆与 local 记忆上进行 attention。

## CSA Cache

CSA 存储:

- `compressed_main`:被选中的全局分支所使用的压缩 values。
- `compressed_index`:用于挑选 top-k 全局 block 的压缩 index keys。
- `compressed_positions`:每个压缩 block 的位置 id。
- `compressed_valid_mask`:用于稀疏选择的有效 block 掩码。
- `previous_b_*`:用于下一次 block 压缩的之前重叠 B 状态。
- `pending_a_*`、`pending_b_*`:等待填满一个完整 block 的当前尾部状态。
- `local_c`:近期的 local 分支状态。

最关键的配置项:

- `compression_factor`:CSA block 大小。
- `top_k_blocks`:索引器选出的压缩 block 数量。
- `indexer_dim`:稀疏 block 索引器的潜在维度。
- `n_indexer_heads`:index query 头数。
- `query_compression_dim`:进行 index 评分前的 query 潜在维度。
- `window_size`:local sliding-window 长度。
- `use_separate_local_kv`:使用专用的 local KV 投影。

在 decode 阶段,CSA 将新 token 投影为 A/B main 状态及 A/B index 状态。当一个 block 准备就绪时,它会将当前 A block 与之前的 B block 一起压缩,模拟论文风格模块所使用的重叠压缩思想。

## Parallel Prefill

`deepseek_prefill_mode="parallel"` 通过单次模型前向从完整 prompt 构建上述 cache:

```text
full prompt forward
  -> capture x_norm at each layer
  -> HCA project C/Z and compress complete blocks
  -> CSA project A/B main + A/B index states and compress complete blocks
  -> store pending tail and local window
```

该模式在 prefill 期间确实会调用一次模型常规 forward。Prefill 完成后,生成的 token 走 `forward_decode` 路径。

## Sequential Debug Prefill

`deepseek_prefill_mode="sequential_debug"` 通过将 prompt 逐 token 经由 `forward_decode` 重放来填充同样的 cache 对象。

该方式有用之处在于它走的是与生成相同的状态变更路径,但速度更慢,也不太能代表实际的 prefill。

## 当前范围

cache 用纯 PyTorch 实现,可在 CPU 上测试。它有意不包含自定义 CUDA kernel、融合 attention kernel、paged attention 分配器或生产级服务调度器。
