# HCA:Heavily Compressed Attention

HCA 是 DeepSeek-V4 Mini 中两个核心长上下文 attention 模块之一。

## 是什么

Heavily Compressed Attention 将多组 token 级 KV 条目压缩成长度更短的全局 KV 内存。每个 query 都会 attend 到:

1. 已完整压缩的全局 KV 块,
2. 精确的局部滑动窗口 KV token,
3. 可选的 attention sink 条目。

目标是让全局上下文变得廉价,同时保留近期的局部细节。

## 在 DeepSeek-V4 Mini 中的角色

HCA 是激进压缩分支。

适用场景:

- 序列较长,
- 全局上下文可以粗粒度表示,
- 局部语法/细节由滑动窗口处理,
- 你希望获得比 dense MHA 更低的内存与 attention 开销。

在 `hybrid` 模式下,HCA 层与 CSA 层互补:HCA 提供宽泛的压缩上下文,CSA 提供选择性的稀疏检索。

## 前向流程

输入:

```text
x: [B, T, d_model]
```

内部步骤:

```text
x
  -> q_proj                 -> Q: [B, T, n_heads, head_dim]
  -> kv_proj                -> C: [B, T, head_dim]
  -> z_proj                 -> Z: [B, T, head_dim]

(C, Z)
  -> HCATokenCompressor     -> compressed_C: [B, S, head_dim]

Q 关注:
  -> 因果上已完整的 compressed_C 块
  -> window_size 内精确的局部 KV token
  -> 可选的 sink KV

attention 输出
  -> 分组或 dense 输出投影
  -> 残差 dropout
  -> [B, T, d_model]
```

其中:

```text
S = ceil(T / compression_factor)
```

## 因果规则

对于全局压缩 attention,query token `t` 只能 attend 到压缩块 `s`,当且仅当该块在 query 当前压缩块之前已完整完成:

```text
allowed[t, s] = s < floor(t / compression_factor)
```

当前块不会通过压缩内存访问,因为它可能包含未来 token。当前/近期信息来自局部滑动窗口分支。

## 局部窗口规则

局部分支使用精确的 token 级 KV 条目:

```text
allowed[t, s] = s <= t and t - s < window_size
```

正是这一机制保留了精确的短程细节。

## 主要超参数

| 参数 | 作用 |
| :--- | :--- |
| `d_model` | 进入与离开 HCA 的隐藏宽度。 |
| `n_heads` | query head 数量。head 越多,query 子空间越多。 |
| `head_dim` | 每个 query head 与共享 KV 向量的宽度。当前实现要求 `n_heads * head_dim == d_model`。 |
| `compression_factor` / `hca_compression_factor` | 压缩成单个全局 KV 条目的 token 数。值越大,全局内存越便宜,信息损失越多。 |
| `window_size` | 每个 query 可见的精确近期 token 数量。值越大,保留的局部细节越多。 |
| `attention_dropout` | attention 权重上的 dropout。 |
| `residual_dropout` | HCA 输出投影后的 dropout。 |
| `use_bias` / `use_attention_bias` | 启用投影 bias。 |
| `use_rope` | 对 query 应用 rotary 位置 embedding。 |
| `rope_theta` | RoPE 频率基数。 |
| `rotary_dim` | 使用 RoPE 的每 head 维度数。 |
| `max_seq_len` | 校验/缓冲区接受的最大序列长度。 |
| `init_std` | 初始化尺度。 |
| `use_attention_sink` | 添加可学习的全局 sink key/value 条目。 |
| `use_grouped_output_projection` | 对 attention 输出使用分组投影。 |
| `output_projection_groups` | 分组数,必须能整除 `n_heads`。 |

## 超参数如何影响行为

### `compression_factor`

更大的值:

- 减小压缩序列长度,
- 减少全局 attention 内存,
- 让每个压缩条目概括更多 token,
- 过大会损害精细检索。

小型调试:

```yaml
hca_compression_factor: 4
```

面向长上下文:

```yaml
hca_compression_factor: 16
```

### `window_size`

更大的值:

- 提升局部精度,
- 增加局部 attention 开销,
- 减少激进压缩造成的伪影。

小型调试:

```yaml
window_size: 4
```

Mini 研究:

```yaml
window_size: 32
```

### `n_heads` 与 `head_dim`

它们控制 attention 容量。

在本实现中:

```text
n_heads * head_dim == d_model
```

示例:

```yaml
d_model: 256
n_heads: 4
head_dim: 64
```

### `use_attention_sink`

attention sink 是可学习的 KV 条目。它们为 attention 质量提供安全的归宿,并能在可用真实 key 较弱或被遮蔽时稳定 attention 模式。

### `use_grouped_output_projection`

分组输出投影在不引入自定义 kernel 的前提下,保留了论文启发的分组投影思想。

## 推荐配置

CPU 冒烟测试:

```yaml
attention_type: hca
d_model: 32
n_heads: 4
head_dim: 8
hca_compression_factor: 4
window_size: 4
rotary_dim: 8
```

Mini 研究:

```yaml
attention_type: hca
d_model: 256
n_heads: 4
head_dim: 64
hca_compression_factor: 16
window_size: 32
rotary_dim: 64
use_attention_sink: true
use_grouped_output_projection: true
```

## 保护 HCA 的测试

相关测试:

- `tests/test_hca.py`
- `tests/test_deepseek_model.py`

覆盖的行为:

- 输出形状,
- 压缩长度,
- 因果 mask,
- 局部窗口 mask,
- 有限梯度,
- attention 权重,
- 分组输出投影,
- 在 block/model 内部的集成。
