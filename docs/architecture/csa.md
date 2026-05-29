# CSA:Compressed Sparse Attention

CSA 是本仓库中最重要的长上下文 attention 模块。

## 是什么

Compressed Sparse Attention 结合了:

1. KV 压缩,
2. 轻量级 indexer 打分,
3. top-k 稀疏块选择,
4. 在所选压缩块上的 MQA 风格 attention,
5. 精确的局部滑动窗口 attention。

它的设计目标是在降低 attention 开销的同时,保留有针对性的长程检索。

## 在 DeepSeek-V4 Mini 中的角色

CSA 是选择性检索分支。

适用场景:

- 长程事实很重要,
- query 应该 attend 到少数几个相关的全局区域,
- dense attention 过于昂贵,
- 仅靠 HCA 压缩过于粗糙。

CSA 对合成检索实验尤其重要,因为模型必须从干扰上下文中恢复出特定的 key-value 事实。

## 前向流程

输入:

```text
x: [B, T, d_model]
```

高层流程:

```text
x
  -> q_down_proj
  -> q_up_proj                    -> 用于核心 attention 的 Q
  -> index_q_up_proj              -> indexer 的 query
  -> index_weight_proj            -> indexer head 权重

x
  -> a_kv_proj, b_kv_proj         -> token 级 KV value
  -> a_z_proj, b_z_proj           -> 压缩 logits
  -> CSAOverlappedCompressor      -> 压缩后的 KV 条目

x
  -> a_index_kv_proj, b_index_kv_proj
  -> a_index_z_proj, b_index_z_proj
  -> CSAOverlappedCompressor      -> 压缩后的 indexer key

indexer query + 压缩的 indexer key
  -> CSALightningIndexer
  -> top-k 因果块索引

Q 关注:
  -> 选中的压缩 KV 条目
  -> 精确的局部 KV token
  -> 可选的 sink KV

输出
  -> 分组或 dense 输出投影
  -> 残差 dropout
```

## 重叠的 A/B 压缩

CSA 使用两条 KV 分支:

- `a_*` 分支用于当前压缩块,
- `b_*` 分支用于一个重叠的前一块贡献。

这在保持 PyTorch 可读性的前提下,近似论文的重叠压缩思想。

作用:

- 避免压缩条目对块边界过于脆弱,
- 让每个压缩块拥有更丰富的局部邻域,
- 仍然将序列长度大致缩短为 `compression_factor` 分之一。

## Lightning Indexer

indexer 决定每个 query 应当看到哪些压缩块。

输入:

- 压缩后的 indexer key,
- 低秩 indexer query,
- 每个 indexer head 的权重。

输出:

- 每个 query 的 top-k 压缩块索引。

重要属性:

- 选择是因果的;未来块会被 mask 掉。

## 因果规则

CSA 必须避免选中包含未来信息的压缩块。

对于 query token `t`,某压缩块只有在严格位于 query 当前压缩块之前时才有效:

```text
block_idx < floor(t / compression_factor)
```

当前/近期 token 由精确的局部窗口分支处理。

## 局部窗口分支

局部分支提供对近期 token 的精确访问:

```text
allowed[t, s] = s <= t and t - s < window_size
```

这一点至关重要,因为压缩可能会丢失 token 级细节。

## 主要超参数

| 参数 | 作用 |
| :--- | :--- |
| `d_model` | 进入与离开 CSA 的隐藏宽度。 |
| `n_heads` | 核心 attention 的 query head 数量。 |
| `head_dim` | 每个 query head 与压缩 KV 条目的宽度。要求 `n_heads * head_dim == d_model`。 |
| `compression_factor` | 每个压缩块包含的 token 数。 |
| `top_k` / `top_k_blocks` | 每个 query 选中的压缩块数量。 |
| `window_size` | 精确的局部上下文长度。 |
| `indexer_dim` | 压缩 indexer key 与 query 的宽度。 |
| `n_indexer_heads` | indexer 打分 head 的数量。 |
| `query_compression_dim` | 由核心与 indexer 共享的低秩 query 瓶颈宽度。 |
| `attention_dropout` | attention 权重上的 dropout。 |
| `residual_dropout` | 输出投影后的 dropout。 |
| `use_bias` / `use_attention_bias` | 启用投影 bias。 |
| `use_rope` | 应用 rotary 位置 embedding。 |
| `rope_theta` | RoPE 频率基数。 |
| `rotary_dim` | 使用 RoPE 的每 head 维度数。 |
| `max_seq_len` | 校验/缓冲区接受的最大序列长度。 |
| `init_std` | 初始化尺度。 |
| `use_attention_sink` | 添加可学习的 sink KV 条目。 |
| `use_grouped_output_projection` | 启用分组输出投影。 |
| `output_projection_groups` | 投影分组数,必须能整除 `n_heads`。 |
| `use_indexer_score_bias` | 可选地将 indexer 分数注入核心 attention logits。 |
| `use_separate_local_kv` | 为精确局部 KV token 使用单独的投影。 |

## 超参数如何影响行为

### `compression_factor`

更大的值:

- 缩短压缩内存,
- 降低 attention/indexer 开销,
- 让每个压缩条目代表更多 token,
- 可能削弱精细检索能力。

小型调试:

```yaml
compression_factor: 4
```

### `top_k_blocks`

更大的值:

- 让每个 query 检视更多全局块,
- 在长上下文检索中提升召回,
- 增加 attention 开销。

小型调试:

```yaml
top_k_blocks: 2
```

Mini 研究:

```yaml
top_k_blocks: 8
```

### `indexer_dim`

更大的值:

- 让稀疏选择更具表达力,
- 增加 indexer 投影与打分开销。

### `n_indexer_heads`

更多的 indexer head:

- 提供多种打分视角,
- 可改善块选择,
- 给 indexer 路径增加开销。

### `query_compression_dim`

这是 query 生成的低秩瓶颈宽度。

更小的值:

- 更便宜,
- 表达力更弱。

更大的值:

- query 表征更丰富,
- 参数与计算更多。

### `window_size`

更大的值:

- 保留更多精确的近期上下文,
- 局部 attention 计算开销更高。

## 推荐配置

CPU 冒烟测试:

```yaml
attention_type: csa
d_model: 32
n_heads: 4
head_dim: 8
compression_factor: 4
top_k_blocks: 2
window_size: 4
indexer_dim: 8
n_indexer_heads: 2
query_compression_dim: 8
rotary_dim: 8
```

Mini 研究:

```yaml
attention_type: csa
d_model: 256
n_heads: 4
head_dim: 64
compression_factor: 4
top_k_blocks: 8
window_size: 32
indexer_dim: 64
n_indexer_heads: 4
query_compression_dim: 64
use_attention_sink: true
use_grouped_output_projection: true
use_separate_local_kv: true
```

## 保护 CSA 的测试

相关测试:

- `tests/test_csa.py`
- `tests/test_deepseek_model.py`

覆盖的行为:

- 输出形状,
- top-k 未来 mask,
- 无未来信息泄漏,
- 有限梯度,
- 局部窗口行为,
- 压缩 KV 长度的缩减,
- 模型级集成。
