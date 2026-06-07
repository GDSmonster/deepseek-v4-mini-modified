# Linear Attention 变体

## 项目目标

本项目在 mini DeepSeek-V4-style 语言模型中加入一个因果线性注意力变体，用来研究基于 kernel 分解的 attention 是否能降低长序列场景下的注意力计算成本，同时在小规模语言建模和合成检索任务上保留足够的自回归建模能力。

核心研究问题是：

> 在一个 CPU 可测试的教学版 DeepSeek-V4-style 实现中，是否可以用因果线性 kernel attention 替换二次复杂度的 softmax attention，把注意力瓶颈从显式 token-token 两两交互转化为前缀状态聚合，同时保持稳定训练和可用的长上下文行为？

从更抽象的角度看，本项目并不只是“把一个 attention 模块换成另一个 attention 模块”，而是把 Transformer 中最核心的 token 交互机制重新解释为 kernel 计算问题。标准 softmax attention 通过 `exp(q^T k)` 定义 query 与 key 之间的相似性，本质上是在序列内部构造一个稠密的 pairwise kernel matrix。它表达力强，但计算和存储都随序列长度呈二次增长。

线性注意力的目标是对这个 pairwise kernel 做结构化分解：将原本依赖 `q` 和 `k` 两两组合的 kernel `K(q, k)` 近似写成两个独立特征映射的内积 `phi(q)^T phi(k)`。这样 attention 不再需要先构造完整的 `QK^T` 矩阵，而可以利用结合律先聚合所有历史 key-value 信息，再由当前 query 读取这个聚合状态。也就是说，本项目希望验证的是：在 DeepSeek-V4-style 模型这种包含压缩注意力、稀疏检索、MoE 和多预测头的复杂架构中，能否把 attention 的核心瓶颈从“显式两两比较”升维为“可递推的 kernel 状态建模”。

因此，这个变体的研究意义在于：它把长上下文建模问题从工程层面的降显存、降耗时，进一步转化为表示学习层面的 kernel 设计问题。不同的 `phi(.)` 实际上对应不同的归纳偏置：softmax 更擅长尖锐选择和精确 token 匹配，而线性 kernel 更偏向连续聚合和流式状态更新。比较两者在同一 DeepSeek-V4-style 骨架下的表现，可以帮助分析长上下文模型到底依赖多少显式 token-token 检索，多少可以由低秩/可分解的状态表达替代。

## 动机

Softmax attention 可以写成一种 kernel attention：

```text
Attn(q_i, K, V) = sum_j exp(q_i k_j^T) v_j / sum_j exp(q_i k_j^T)
```

它的二次复杂度来自于需要显式构造所有 query-key token 对。线性注意力则用一个正值特征映射替换或近似原来的 pairwise kernel：

```text
K(q, k) ~= phi(q)^T phi(k)
```

对于因果语言模型，这种形式可以写成递推的前缀状态：

```text
S_t = S_{t-1} + phi(k_t) v_t^T
z_t = z_{t-1} + phi(k_t)
o_t = phi(q_t)^T S_t / phi(q_t)^T z_t
```

本仓库中的实现使用 `phi(x) = elu(x) + 1 + eps`，保留标准的 Q/K/V/O 投影结构，并在特征映射之前对 Q/K 应用 RoPE。

## 为什么选择 DeepSeek-V4-style 架构

选择 DeepSeek-V4-style 架构作为线性注意力替换实验平台，主要是因为它不是一个只包含标准 MHA + MLP 的简单 Transformer，而是一个更接近现代大模型设计问题的组合系统。这个仓库已经包含 MHA、HCA、CSA、MoE、mHC 和 MTP 等模块，适合研究 attention 变体在复杂模型栈中的真实影响，而不是只在孤立 attention 层上做形状验证。

第一，DeepSeek-V4-style 架构天然关注长上下文效率。仓库中的 HCA 和 CSA 本身就围绕压缩、稀疏选择和局部窗口展开，说明该架构的核心问题之一是如何避免全量二次 attention。在线性注意力视角下，HCA/CSA 可以看作从结构稀疏和压缩路由角度减少 token 交互，而 linear attention 则是从 kernel 可分解角度减少 token 交互。把它们放在同一框架中比较，可以形成一个清晰的研究主线：长上下文效率可以来自稀疏化、压缩化，也可以来自 kernel 分解。

第二，DeepSeek-V4-style 架构包含 MoE 和多种残差/辅助预测机制，能够观察 attention 替换对整个模型系统的连锁影响。线性注意力改变的不只是 attention 的复杂度，也可能改变 token 表示的分布、MoE router 的输入、MTP 辅助头的预测难度，以及深层残差流中的信息传播方式。因此，在这个架构中替换 attention，比在最小 Transformer 中替换 attention 更能体现真实模型设计中的权衡。

第三，这个仓库是教学版、CPU-safe、模块化的实现，适合做可控 ablation。`attention_type` 已经抽象为 `mha/hca/csa/hybrid` 的可切换接口，新增 `linear` 后可以在相同 embedding、FFN、MoE、MTP 和训练配置下进行对照实验。这样可以把变量尽量集中在 attention kernel 上，而不是混入训练数据、模型规模、优化器或工程 kernel 的差异。

因此，本项目选择 DeepSeek-V4-style 架构并不是为了声称线性注意力能直接替代真实 DeepSeek 系列模型中的所有注意力设计，而是为了构造一个足够复杂、但仍然可控的实验平台：在同一模型骨架中比较 softmax attention、压缩/稀疏 attention 和 kernel-factorized linear attention，分析不同长上下文机制的成本、稳定性和表达能力边界。

## 实现范围

已经实现：

- `CausalLinearAttention` 的 full-sequence 因果前向路径。
- `DeepSeekV4LMConfig` 中的 `attention_type="linear"`。
- 与现有 attention factory 和 block 接口的集成。
- `need_weights=True` 时返回调试用 attention weights。
- 覆盖 shape、因果性、padding mask 行为和模型集成的 CPU-safe 测试。

暂未实现：

- 专门用于线性注意力解码的 prefix state cache，即缓存 `S_t` 和 `z_t`。
- Flash/fused kernel。
- 与 softmax attention 数值等价的保证。

## 建议实验

1. 使用 `attention_type in {"mha", "linear", "csa"}` 训练可对比的 tiny 模型。
2. 随序列长度增加，比较显存/内存占用和 wall-clock time。
3. 在 WikiText-2 或 TinyStories 上评估 next-token loss。
4. 使用 synthetic retrieval dataset 评估长上下文 key-value 检索能力。
5. 分析 softmax attention 依赖尖锐 token selection 的场景中，linear attention 的失效模式。
