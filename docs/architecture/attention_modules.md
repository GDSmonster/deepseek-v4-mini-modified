# Attention 模块

DeepSeek-V4 Mini 通过 `attention_type` 支持四种 attention 模式:

- `mha`:标准的因果多头 attention。
- `hca`:Heavily Compressed Attention(高度压缩 attention)。
- `csa`:Compressed Sparse Attention(压缩稀疏 attention)。
- `hybrid`:在 `attention_pattern` 上的逐层调度,通常为 `["csa", "hca"]`。

本文件是 attention 的高层索引。两个 DeepSeek-V4 启发的核心模块各有专门页面:

- [HCA: Heavily Compressed Attention](hca.md)
- [CSA: Compressed Sparse Attention](csa.md)

## 标准 MHA

是什么:

- 一个基线的因果多头 attention 模块,可选启用 RoPE。

作用:

- 提供 dense attention 参考路径。
- 适合用于健全性检查、小型 CPU 训练与消融实验。
- 为 CSA/HCA 提供已知良好的 baseline,用于校验形状、因果性、梯度与损失行为。

关键超参数:

- `n_heads`:query/key/value attention head 的数量。
- `head_dim`:每个 head 的维度。
- `use_rope`:启用 rotary 位置 embedding。
- `rotary_dim`:接收 RoPE 的 head 维度数量。
- `attention_dropout`:attention 权重上的 dropout。
- `residual_dropout`:输出投影后的 dropout。
- `max_seq_len`:位置缓冲区与校验所允许的最大序列长度。

适用场景:

- 初步冒烟测试。
- 小型 CPU 训练。
- 将 dense attention 与压缩/稀疏变体进行对比。

## HCA 摘要

HCA 对 token 级别的 KV 条目进行激进压缩,然后在压缩后的内存以及精确的局部窗口 token 上执行 dense MQA 风格的 attention。

适合使用 HCA 的情况:

- 你需要强压缩。
- 长上下文比精确的全局 token 级内存更重要。
- 你希望获得更便宜的全局 attention 分支。

详见 [HCA](hca.md) 的内部细节与超参数。

## CSA 摘要

CSA 压缩 KV 块,使用一个轻量级 indexer 对压缩块进行打分,为每个 query 选出 top-k 块,并把这些稀疏全局块与精确的局部窗口 token 相结合。

适合使用 CSA 的情况:

- 检索质量很重要。
- 模型需要选择性的长程访问。
- 你正在测试合成 key-value 检索或长上下文行为。

详见 [CSA](csa.md) 的内部细节与超参数。

## 混合 Attention(Hybrid)

是什么:

- 在多个 attention 模块之间循环调度。

作用:

- 模拟论文中 CSA 与 HCA 的交错排列。
- 让某些层专注于稀疏检索,另一些层专注于高度压缩的全局上下文。

关键超参数:

- `attention_type="hybrid"`。
- `attention_pattern`:tuple/list,例如 `("csa", "hca")`。

示例:

```yaml
attention_type: hybrid
attention_pattern: [csa, hca]
```
