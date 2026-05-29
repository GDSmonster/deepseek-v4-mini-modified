# DeepSeek-V4 Mini 知识点完整教程(中文)

> 本仓库是 DeepSeek-V4 技术报告中所有"系统性创新"的纯 PyTorch、教学化、可 CPU 跑通的小型实现。本文按"知识点 → 数学公式 → 代码位置"的顺序,把仓库中涉及的所有概念串成一份自洽的学习路径。
>
> 阅读建议:第 1–3 章是必备基础,第 4–8 章是 DeepSeek-V4 的五大创新点(HCA / CSA / MoE / mHC / MTP),第 9 章把它们拼装成完整模型,第 10 章之后讨论训练栈、推理、并行、Ablation 与工程实践。

---

## 目录

- [1. 仓库整体定位与阅读地图](#1-仓库整体定位与阅读地图)
- [2. 语言模型与因果 Transformer 复习](#2-语言模型与因果-transformer-复习)
- [3. Transformer 基础组件(`src/transformer_modules/`)](#3-transformer-基础组件srctransformer_modules)
  - [3.1 Token Embedding 与 weight tying](#31-token-embedding-与-weight-tying)
  - [3.2 RMSNorm](#32-rmsnorm)
  - [3.3 RoPE 旋转位置编码(可部分应用)](#33-rope-旋转位置编码可部分应用)
  - [3.4 SwiGLU MLP](#34-swiglu-mlp)
  - [3.5 标准 MHA 与因果注意力](#35-标准-mha-与因果注意力)
- [4. HCA:Heavily Compressed Attention(强压缩注意力)](#4-hcaheavily-compressed-attention强压缩注意力)
- [5. CSA:Compressed Sparse Attention(压缩稀疏注意力)](#5-csacompressed-sparse-attention压缩稀疏注意力)
- [6. DeepSeekMoE:稀疏专家 FFN](#6-deepseekmoe稀疏专家-ffn)
- [7. mHC:流形约束的超连接残差](#7-mhc流形约束的超连接残差)
- [8. MTP:多 token 预测辅助头](#8-mtp多-token-预测辅助头)
- [9. 拼装:DeepSeekV4Block 与 DeepSeekV4LM](#9-拼装deepseekv4block-与-deepseekv4lm)
- [10. 训练栈](#10-训练栈)
- [11. 推理与 KV cache](#11-推理与-kv-cache)
- [12. 数据流水线](#12-数据流水线)
- [13. 并行计算](#13-并行计算)
- [14. Ablation 实验体系](#14-ablation-实验体系)
- [15. 工程实践与仓库约定](#15-工程实践与仓库约定)
- [附录 A:符号约定](#附录-a符号约定)

---

## 1. 仓库整体定位与阅读地图

DeepSeek-V4 的核心论点是:在标准 dense Transformer 上,要进一步突破必须同时解决三件事——

1. **上下文长度**:朴素 full attention 的复杂度是 $O(T^2)$,百万 token 不可行 → **HCA + CSA**。
2. **模型容量**:dense scaling 既贵又浪费 → **MoE 稀疏激活**。
3. **训练稳定性**:深层网络的残差信号容易塌缩 → **mHC**;同时 **MTP** 提供更密集的监督信号,**Muon** 提供更优的曲率适配。

本仓库按这个思路把每个创新拆成独立模块,并保证**每个模块都能在 CPU 上单独跑测试**,代码组织为:

| 目录 | 职责 |
| :--- | :--- |
| `src/transformer_modules/` | 经典基础件:RMSNorm、RoPE、SwiGLU、MHA、TokenEmbedding、TransformerBlock |
| `src/` | DeepSeek-V4 的五大创新:HCA、CSA、MoE、mHC、MTP,以及把它们组合的 `DeepSeekV4Block` 与 `DeepSeekV4LM` |
| `training/` | 训练编排:Muon、AdamW、cosine warmup、autocast、EMA、checkpoint、各种 metrics |
| `inference/` | 推理:prefill / decode、HCA/CSA/MHA 的 KV cache、采样、audit |
| `data/` | 合成长上下文检索数据集 + Hugging Face 文本数据集预设 |
| `parallel/` | DDP 数据并行 + 分块层级模型并行(教学级) |
| `ablations/` | A1–A6 六组高层消融实验 |
| `scripts/` | 5 个 CLI:`data` / `train` / `inspect` / `inference` / `parallel` / `ablate` |
| `config/` | YAML 配置文件(不自动加载,作为参考与 ablation 输入) |
| `tests/` | CPU-safe 测试,覆盖每个组件的形状、因果性、梯度 |

**学习推荐顺序**:从基础件 → HCA → CSA → MoE → mHC → MTP → DeepSeekV4LM → 训练 → 推理 → ablation。

---

## 2. 语言模型与因果 Transformer 复习

### 2.1 自回归语言模型

给定 token 序列 $x_{1:T}$,自回归 LM 把联合分布拆成

$$
p(x_{1:T}) = \prod_{t=1}^{T} p(x_t \mid x_{<t}; \theta).
$$

训练目标是负对数似然(交叉熵):

$$
\mathcal{L}_{\text{LM}}(\theta) = -\frac{1}{N}\sum_{t} \log p(x_t \mid x_{<t}; \theta).
$$

仓库中 `src/mini_deepseek_class.py::DeepSeekV4LM._compute_lm_loss` 同时支持两种 label 约定(`labels_are_shifted=True` 或 HuggingFace 风格 `labels=input_ids`,内部自动 shift)。

### 2.2 因果 mask

attention 中,query token $t$ 只能看到 key 位置 $s \le t$:

$$
\text{mask}_{t,s} = \begin{cases} 0 & s \le t \\ -\infty & s > t \end{cases}.
$$

经过 softmax 后,$s>t$ 的概率严格为 0,保证训练等价于"在每个位置预测下一个 token"。

仓库中你会反复看到的写法是 **safe masked softmax**(`HCAAttention._safe_concat_softmax`):先用 `-inf` 填充非法位置,做 fp32 softmax,再用 `mask` 把那些位置乘回 0,最后**重新归一化**——这能正确处理"某一行全部被 mask"的边界情况(整行变成精确 0 而不是 NaN)。

---

## 3. Transformer 基础组件(`src/transformer_modules/`)

### 3.1 Token Embedding 与 weight tying

- **Token Embedding**:把 vocab 中的整数 id 映射为 $D$ 维稠密向量,矩阵 $E \in \mathbb{R}^{V \times D}$。
- **scale**:可选乘 $\sqrt{D}$(`scale_embeddings`)。
- **weight tying**:`tie_word_embeddings=True` 时,LM head 的权重直接等于 embedding 权重(节省 $V \cdot D$ 个参数,提升小模型质量)。仓库实现:

```python
# src/mini_deepseek_class.py
def tie_lm_head_to_embeddings(self) -> None:
    self.lm_head.weight = _get_token_embedding_weight(self.embedding)
```

### 3.2 RMSNorm

LayerNorm 同时做"减均值"与"除方差",但实践发现"减均值"对 Transformer 不是必需的。**RMSNorm**(Zhang & Sennrich, 2019)只用均方根做归一化:

$$
\text{RMS}(x) = \sqrt{\frac{1}{D}\sum_{i=1}^D x_i^2 + \varepsilon}, \qquad
y = \frac{x}{\text{RMS}(x)} \odot \gamma.
$$

参数只有可学习的 scale $\gamma \in \mathbb{R}^D$。对应实现:

```python
# src/transformer_modules/RMSNorm.py
mean_square = x.float().pow(2).mean(dim=-1, keepdim=True)
inv_rms = torch.rsqrt(mean_square + self.eps)
y = (x.float() * inv_rms).to(original_dtype) * self.weight
```

注意它**始终在 fp32 内做归一化计算**再转回原 dtype,这是 bf16/fp16 训练的标准稳定性技巧。

### 3.3 RoPE 旋转位置编码(可部分应用)

RoPE(Su et al., 2021)把位置信息直接乘进 Q/K 向量,等价于把每两维 $(x_{2i}, x_{2i+1})$ 视为复数 $x_{2i} + i\,x_{2i+1}$,并在位置 $m$ 处旋转角度

$$
\theta_{m,i} = m \cdot \omega_i,\qquad \omega_i = \text{base}^{-2i/D_{\text{rot}}}.
$$

旋转矩阵作用为:

$$
\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = 
\begin{pmatrix} \cos\theta_{m,i} & -\sin\theta_{m,i} \\ \sin\theta_{m,i} & \cos\theta_{m,i} \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}.
$$

关键性质:Q 与 K 的内积 $\langle q_m, k_n \rangle$ 只依赖**相对位置** $m-n$,因此外推性优于学习式 absolute position embedding。

**partial RoPE**:仓库中 `rotary_dim` 可以小于 head_dim,前 `rotary_dim` 维做 RoPE,剩余维度保持原样。对长序列泛化更稳。

实现:`src/transformer_modules/rope.py::RotaryEmbedding`,期望输入 `[B, T, H, Dh]`,支持自动位置、`[T]`、`[B,T]` 三种 `position_ids` 形式。

### 3.4 SwiGLU MLP

经典 FFN 是 $\text{FFN}(x) = W_2\,\sigma(W_1 x)$。**SwiGLU**(Shazeer 2020)用门控变体:

$$
\text{SwiGLU}(x) = (W_3 x) \odot \text{SiLU}(W_1 x), \qquad
\text{FFN}(x) = W_2 \cdot \text{SwiGLU}(x),
$$

其中 $\text{SiLU}(z) = z \cdot \sigma(z)$。门控让模型有选择地"放大/抑制"通道。LLaMA / DeepSeek 系列均采用。`src/transformer_modules/SwiGLU.py`:

- `expansion_factor=4.0` 默认,`hidden_dim` 可显式覆盖;
- `multiple_of` 把 hidden_dim 圆整到 GPU 对齐倍数。

### 3.5 标准 MHA 与因果注意力

多头注意力把 $D$ 维分成 $H$ 个 head,每个 head 独立做缩放点积注意力:

$$
\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{D_h}} + M\right) V,
$$

其中 $M$ 为因果 mask。多个 head 拼回去再过一次输出投影 $W_O$。

`src/transformer_modules/mha_baseline.py` 是仓库的 dense baseline,支持 RoPE、attention dropout、residual dropout、padding mask。后面 HCA / CSA 是它的**结构性替代**,但在训练时可以混用(`attention_type="hybrid"` 用 `attention_pattern` 按层循环切换)。

---

## 4. HCA:Heavily Compressed Attention(强压缩注意力)

### 4.1 动机

朴素 attention 的代价 $O(T^2 D)$ 在 $T$ 增大后无法承受。HCA 的思路:**过去远处的 token 不需要逐个保留,把它们压缩成更少的 KV 条目即可**。同时为了保留近处细节,再加一个"局部精确滑窗"分支。

### 4.2 数学结构

把序列按压缩因子 $m'$ 分成块,每块包含 $m'$ 个 token。每个 query token $t$ 看到三类 key/value:

1. **(可选)attention sink**:1 个全局可学习 token,缓解 softmax 把质量挤到第一个 token 的现象;
2. **compressed global**:对 $\lfloor t/m'\rfloor$ 之前**已完成的块**,每块用一个压缩 KV 表示;
3. **local exact**:大小为 $W$ 的滑动窗口,精确保留邻近 $W$ 个 token 的 KV。

设第 $s$ 块原始 KV 为 $C_{s\cdot m':(s+1)\cdot m'}$,压缩为单条 $\bar C_s$。HCA 计算:

$$
\text{HCA}(t) = \text{softmax}\!\Big(\big[\,q_t k_{\text{sink}}^\top,\; q_t \bar K_{<\lfloor t/m'\rfloor}^\top,\; q_t K_{[t-W+1:t]}^\top\,\big] / \sqrt{D_h}\Big) \cdot \big[V_{\text{sink}}, \bar V_{<\lfloor t/m'\rfloor}, V_{[t-W+1:t]}\big].
$$

注意三段 score **拼起来一次 softmax**,这样 token 在"压缩长程"和"精确局部"之间会自动竞争。

### 4.3 因果性细节(关键)

- compressed global 分支只允许看**严格在当前块之前**已经完成的块: $\text{allowed}[t,s] = (s < \lfloor t/m'\rfloor)$。
- 当前块自身由 local 分支负责。

这是 `_build_global_allowed_mask` 与 `_build_local_allowed_mask` 的代码意义,确保并行训练时不出现未来泄漏。

### 4.4 MQA(Multi-Query)与共享 KV

HCA 走 MQA 路径:Q 有 $H$ 个 head,但 K/V 在所有 head 间共享(`kv_proj` 输出维度只是 `head_dim`)。这把 KV cache 显存降到 $1/H$。

### 4.5 Grouped output projection

不再用一个 $D \to D$ 的稠密输出投影,而是把 head 分组,每组独立投影后求和。对 MQA 更对齐,也减小参数量。

### 4.6 代码位置与训练 / 推理两条路径

- 训练 forward:`src/deepseek_hca_attention.py::HCAAttention.forward`;
- 推理 forward:同文件 `forward_decode`,对应 cache 走 `inference/hca_cache.py::HCALayerCache`,详见第 11 章。

---

## 5. CSA:Compressed Sparse Attention(压缩稀疏注意力)

### 5.1 动机

HCA 把过去**全部均匀压缩**,但其实大多数远处 token 与当前 query 无关——只需要从大量压缩块中挑出 **top-k 最相关** 的块做精确 attention 即可。这就是 CSA 的核心:**带索引器的稀疏 attention**。

### 5.2 整体结构

CSA 的一层做四件事:

1. **共享低秩 query 路径**:先用 `q_down_proj` 把 $D \to d_q$,再用 `q_up_proj` 升回 $H D_h$,既给主 attention 提供 Q,也给 indexer 提供低维 query。
2. **a/b 两路压缩 KV**:每隔 $m'$ 个 token,用两套不同的压缩头(a-branch 与 b-branch)各产生一条压缩 KV 表示;两路 overlap 增强稳健性。
3. **Lightning Indexer**:用低维 indexer query 与压缩块的 indexer 表示打分,得到 $[B,T,S]$ 分数矩阵;每个 query 取 top-$k$ 块作为稀疏全局 attention 的 key/value。
4. **Local 分支**:与 HCA 一致,大小 $W$ 的滑动窗口。

最终输出仍是 sink + global(top-k 稀疏块)+ local 三段 score 拼起来一次 softmax。

### 5.3 索引器与 top-k

设压缩块表示为 $I_{\text{comp}} \in \mathbb{R}^{B\times S\times I}$,query 索引向量为 $Q_{\text{idx}} \in \mathbb{R}^{B\times T\times H_i\times I}$,索引头权重 $w \in \mathbb{R}^{B\times T\times H_i}$,则索引分数为:

$$
\text{idx\_score}[b,t,s] = \sum_{h} w_{b,t,h} \cdot \langle Q_{\text{idx}}[b,t,h,:],\; I_{\text{comp}}[b,s,:]\rangle.
$$

施加因果 mask `s < floor(t/m')` 后取 top-$k$:

$$
\mathcal{S}(t) = \text{TopK}_k\!\big(\{\text{idx\_score}[b,t,s] : s \in \text{allowed}\}\big).
$$

只有 $\mathcal{S}(t)$ 中的块进入主 attention。这把全局分支的代价从 $O(TS)$ 降到 $O(Tk)$,$k \ll S$。

代码:`src/csa_light_indexer.py::CSALightningIndexer`,主 attention `src/deepseek_csa_attention.py::CSAAttention`。

### 5.4 与 HCA 的关系

| 维度 | HCA | CSA |
| :--- | :--- | :--- |
| 全局分支 | 看**所有**已完成压缩块 | 只看 indexer 选出的 top-k 块 |
| 压缩路径 | 单路 compressor | a/b 双路 + 低秩 query |
| 适合层 | 浅层 / 偶数层 | 深层 / 奇数层 |

仓库默认 `attention_pattern=("csa","hca")`,即各层交替使用,在 quality / cost trade-off 上更稳。

---

## 6. DeepSeekMoE:稀疏专家 FFN

### 6.1 MoE 思想

dense FFN 每个 token 都跑过所有参数 $W_1, W_2$,计算量 $\propto $ 参数量。**MoE** 的想法:有 $N$ 个独立 expert(每个就是一个小 FFN),每个 token 只激活其中 top-$k$ 个,总参数大但计算量近似不变。

### 6.2 路由器(router)

仓库支持两种 router(`router_type`):

**learned**:线性层 $W_r \in \mathbb{R}^{N\times D}$,得分

$$
g_i(x) = \phi(W_r x)_i,\quad i=1,\dots,N,
$$

其中 $\phi$ 是 `router_score_fn`(默认 `sqrt_softplus`,即 $\phi(z)=\sqrt{\text{softplus}(z)}$,输出非负)。然后取 top-$k$ 的索引 $\mathcal{T}(x)$,并把对应得分归一化(`normalize_topk_weights`)作为权重 $\alpha_i$:

$$
\text{MoE}(x) = \sum_{i \in \mathcal{T}(x)} \alpha_i \cdot \text{Expert}_i(x) + \sum_j \text{SharedExpert}_j(x).
$$

**hash**:用 `input_ids` 的哈希直接确定路由,完全不学习——用于做"路由是否真的有用"的 ablation 基线。

### 6.3 Shared experts

DeepSeekMoE 设计中,部分容量始终激活作为"通用 expert",所有 token 必经。本仓库用 `nn.ModuleList` 表达多个 shared expert,而不是只把它合进一个更宽的 FFN(便于消融与诊断)。

### 6.4 Balance loss(辅助损失)

如果不做约束,router 倾向把所有 token 路由给少数 expert。仓库实现两种轻量 balance objective:

- `balance_loss`:让 expert 选择的整批分布接近均匀;
- `sequence_balance_loss`:对每个 sequence 独立做均衡;

它们的权重(`balance_loss_weight`、`sequence_balance_loss_weight`)若都为 0,LM 总损失就不会加 MoE aux loss(见 `mini_deepseek_class.py` 中 `has_moe_aux_loss` 判定)。

> 注意:这**不是** DeepSeek 论文里"无辅助损失"完整路由系统的复刻,而是教学化的简化版本,优先透明可读。

### 6.5 关键 dataclass

`DeepSeekMoEConfig`(`src/deepseek_moe.py`)字段一览:`num_experts`、`top_k`、`shared_experts`、`router_type`、`router_score_fn`、`normalize_topk_weights`、`router_jitter_noise`、`routed_scale`、`shared_scale`、`balance_loss_weight`、`sequence_balance_loss_weight` 等。

---

## 7. mHC:流形约束的超连接残差

### 7.1 朴素残差的问题

标准 Transformer 每层做:

$$
x_{l+1} = x_l + F_l(\text{Norm}(x_l)).
$$

在很深的网络中,这相当于把所有 sublayer 的输出"全部累加"到一条 stream 上,容易造成信号叠加塌缩 / 表征混叠。

### 7.2 mHC 的多 stream 残差

mHC(Manifold-Constrained Hyper-Connection)把残差扩成 $n_{\text{hc}}$ 条**并行 stream**,记为 $X \in \mathbb{R}^{B\times T\times n_{\text{hc}}\times D}$。每层做的更新由三组矩阵控制:

$$
X_{l+1} = B_l\,X_l + C_l\,F_l(A_l\,X_l),
$$

其中:
- $A_l \in \mathbb{R}^{B\times T\times 1\times n_{\text{hc}}}$,值落在 $(0,1)$:**pre-mix**,告诉 sublayer 用哪几条 stream 的混合作为输入;
- $B_l \in \mathbb{R}^{B\times T\times n_{\text{hc}}\times n_{\text{hc}}}$,**doubly stochastic**(每行每列和为 1):stream 之间做"流形上的混合",用 Sinkhorn 迭代得到;
- $C_l \in \mathbb{R}^{B\times T\times n_{\text{hc}}\times 1}$,值在 $(0,2)$:把 sublayer 输出 $F_l(\cdot)$ 写回各 stream 的强度。

直觉:多条 stream 提供"不同语义子流形"上的并行信号通道,$B$ 在它们之间做约束混合(因为 doubly stochastic,信号既不会爆炸也不会塌缩),$A$/$C$ 控制读/写。

### 7.3 Sinkhorn 投影

要把任意非负矩阵投影到 doubly stochastic 矩阵,可以用 Sinkhorn 迭代:交替对行/列归一化。仓库 `mHC_residuals_utils.py` 提供两种实现(普通与 log 域),以及 `sinkhorn_iters`、`sinkhorn_fp32` 等控制项。

### 7.4 结构化 API

仓库的 mHC 模块同时支持两种调用:

```python
# Wrapper API
X_next = mhc(X, sublayer)

# Modular API(更常用,见 deepseek_block.py 的 _mhc_update)
A, B, C = mhc.compute_ABC(X)
x_sub = mhc.pre_mix(X, A=A)
y_sub = sublayer_fn(x_sub)
X_next = mhc.update(X, y_sub, B_mat=B, C=C)
```

入口 `expand_residual_stream`(扩成 $n_{\text{hc}}$ 条)与 `collapse_residual_stream` / `HyperConnectionReadout`(读出最终单条)。

### 7.5 静态/动态参数化

`dynamic=True` 时,$A,B,C$ 是 token 相关、由网络生成的;`dynamic=False` 退化为静态可学习参数。`bounded_alpha` / `init_alpha` / `alpha_max` 用来稳定训练初期。

---

## 8. MTP:多 token 预测辅助头

### 8.1 思想

主 LM head 只预测下一个 token $x_{t+1}$,信号稀疏。MTP(Multi-Token Prediction)在每个位置额外预测 $x_{t+2}, x_{t+3}, \dots, x_{t+1+D_{\text{mtp}}}$,提供**深度方向的密集监督**:

$$
\mathcal{L}_{\text{MTP}} = \sum_{d=1}^{D_{\text{mtp}}} w_d \cdot \text{CE}\big(\text{Head}_d(\text{Transform}_d(h_t)),\; x_{t+1+d}\big).
$$

权重 $w_d$ 默认相等,可通过 `mtp_depth_loss_weights` 自定义(自动归一化)。

### 8.2 实现要点(`src/deepseek_mtp.py`)

- `MTPTransform`:每个深度独立的小 MLP,把 $h_t$ 转换成"看 $d$ 步未来"的表示;若 `use_mtp_transform=False` 退化为 Identity。
- `Head_d`:线性层 $D \to V$;`tie_with_lm_head=True` 时所有 head 与 LM head 共享权重(强约束 + 节省参数)。
- 标签构造 `build_mtp_labels`:对原 input_ids 做不同步长的偏移,padding/超出位置填 `ignore_index`。

### 8.3 与推理的关系

MTP 头主要服务训练。推理时它可作为 **draft model**(`inference/mtp_decode.py`):一次 forward 生成多个候选 token 用于 speculative decoding-like diagnostics。

---

## 9. 拼装:DeepSeekV4Block 与 DeepSeekV4LM

### 9.1 单层 block:`src/deepseek_block.py::DeepSeekV4Block`

无 mHC 时,标准残差路径:

$$
x \leftarrow x + \text{Attn}(\text{RMSNorm}(x)),\qquad x \leftarrow x + \text{FFN}(\text{RMSNorm}(x)).
$$

启用 mHC 时(`use_mhc=True`),变成:

$$
X \leftarrow \text{mHC-update}\big(X,\; \text{Attn}\!\circ\!\text{Norm}\big),\qquad
X \leftarrow \text{mHC-update}\big(X,\; \text{FFN}\!\circ\!\text{Norm}\big).
$$

`attention` 由 `attention_type` 决定(MHA / HCA / CSA / hybrid 按层分配),`ffn` 由 `ffn_type` 决定(dense SwiGLU 或 DeepSeekMoE)。

block 用 `_supports_kwarg` 在调用时按需传 `position_ids`、`attention_mask`、`need_weights`、`start_pos` 等参数,这是因为不同 attention 模块接口略有差异(MHA 不一定支持 `need_weights`,HCA/CSA 支持)。**这种"宽容签名"的写法在阅读 block 代码时尤其需要注意**。

### 9.2 顶层模型:`src/mini_deepseek_class.py::DeepSeekV4LM`

forward 流程:

1. **embedding**:`TokenEmbedding` → $[B,T,D]$;若 `use_mhc`,先 `expand_residual_stream` 成 $[B,T,n_{\text{hc}},D]$。
2. **N 层 block**:逐层走 `DeepSeekV4Block`,可选收集 attention / MoE / mHC aux。
3. **mHC readout / collapse**:`mhc_collapse_mode in {"readout","mean","sum","first"}`,把 stream 收回 $[B,T,D]$。
4. **final RMSNorm + LM head** → logits $[B,T,V]$。
5. **MTP head**(可选):对 hidden states 跑多深度预测,产出 `mtp_loss`。
6. **损失合成**:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \mathcal{L}_{\text{MTP}} + \mathcal{L}_{\text{MoE-aux}},
$$

只有当对应 head 启用且权重 > 0 时才加入。

### 9.3 推理快路径:`prefill_decode_cache` 与 `forward_decode`

模型同时暴露两个推理专用方法:

- `prefill_decode_cache`:一次性吃完 prompt,构建 HCA/CSA 层级缓存;
- `forward_decode`:每次只吃 1 token,从缓存里增量 attend。

二者把 cache 的 append 与压缩块 flush 全部下沉到 attention 模块的 `forward_decode`(详见第 11 章)。

---

## 10. 训练栈

### 10.1 优化器 1:AdamW

经典 Adam + decoupled weight decay。仓库 `training/adam_optmizer.py` 提供 **参数分组**:

- `nn.Linear.weight`、embedding:加 weight decay;
- bias、LayerNorm/RMSNorm 权重、scalar:不加 weight decay。

这是 LLaMA / GPT-NeoX 以来标准做法。

### 10.2 优化器 2:Muon(`training/muon_optimizer.py`)

Muon 的核心是 **用 Newton–Schulz 迭代近似把更新方向"正交化"**:

设 momentum 后的更新矩阵 $G$,Muon 不直接用 $G$ 做更新,而是用它的"近似零次幂"$\text{ortho}(G)$,即奇异值都被推到 1 附近的版本,使得每个奇异方向被均匀更新,不被某些大奇异值主导。

实现要点:

1. **维度**:Muon 只对 2D 矩阵参数有意义(linear/embedding 权重);其他参数交给 AdamW。这就是 `muon_adamw` hybrid 模式。
2. **Newton–Schulz 5 次迭代**(`zeropower_via_newtonschulz5`):

$$
X \leftarrow a X + b (X X^\top) X + c (X X^\top)^2 X,\quad (a,b,c)=(3.4445,\,-4.7750,\,2.0315).
$$

这是经典三阶/五阶迭代的等距版,比直接 SVD 快得多,数值稳。

3. 对长方矩阵优先用更小的那一边做迭代(避免 $X X^\top$ 矩阵过大)。
4. 加 `momentum` 与可选 `nesterov`,再做 `weight_decay`。

DeepSeek-V4 在大模型上观察到 Muon 在某些层比 AdamW 收敛更快、最终 loss 略低,本仓库主要是把它**透明实现**,而不是承诺生产级速度。

### 10.3 学习率调度(`training/scheduler.py`)

Warmup + cosine:

$$
\eta_t = \begin{cases}
\eta_{\max} \cdot \dfrac{t}{T_{\text{warmup}}}, & t \le T_{\text{warmup}} \\[4pt]
\eta_{\min} + \dfrac{1}{2}(\eta_{\max} - \eta_{\min})\Big(1 + \cos\Big(\pi\dfrac{t - T_{\text{warmup}}}{T - T_{\text{warmup}}}\Big)\Big), & t > T_{\text{warmup}}
\end{cases}
$$

支持给 Muon 与 AdamW 用不同的 `min_lr`(`min_muon_lr`)。

### 10.4 混合精度(`training/autocast.py`)

- 默认 `bf16`,若 GPU 不支持自动 fallback 到 `fp16` + `GradScaler`(`fallback_bf16_to_fp16`);
- bf16 下不需要 GradScaler;
- RMSNorm、softmax 在内部强制 fp32 计算,稳定性优先。

### 10.5 梯度裁剪与累积

- `grad_clip=1.0`(默认):全局 L2 norm 裁剪;
- `grad_accum_steps`:在 optimizer step 前累积 N 个 micro-batch,等价大 batch。

### 10.6 EMA(`training/ema.py`)

权重指数平滑:

$$
\theta_{\text{ema}} \leftarrow \alpha\,\theta_{\text{ema}} + (1-\alpha)\,\theta,\quad \alpha = \text{ema\_decay} \approx 0.999.
$$

可放在 CPU 上(`ema_device="cpu"`)以省显存。eval 时可选用 EMA 权重。

### 10.7 Checkpoint(`training/chekpoints.py`,**故意拼错**)

- 保存 model state、optimizer state、scheduler state、scaler state、RNG state、metric 历史;
- `keep_last_n_checkpoints` 自动滚动删除;
- 监控指标 `monitor_name`(默认 `eval_loss`,模式 `min`)单独保存 best;
- 支持 `resume_path` 恢复,可选 `restore_rng_state`。

> 文件名 `chekpoints.py` 是**故意拼错**的,大量 import 依赖此名,见 `AGENTS.md`。

### 10.8 训练入口

- 高层 API:`training.train_deepseek.train_deepseekv4(...)`(纯 Python kwargs,无 Hydra);
- 单 epoch 训练循环:`training/train_one_epoch.py`;
- 评估:`training/eval_one_epoch.py`,带可选**定性预览**(从 val 集采样一个样本生成续写,直观看是否在学)。

### 10.9 训练期 metrics

- LM:`loss_metrics.py`(perplexity、token-acc、overlap);
- MoE:`moe_metrics.py`(router 熵、expert 利用率、balance loss);
- mHC:`mhc_metrics.py`($A,B,C$ 范数、stream 余弦、Sinkhorn 误差);
- MTP:`mtp_metrics.py`(每深度 loss 与 acc);
- 综合:`full_deepseek_metrics.py` 把以上聚合为一份诊断快照。

---

## 11. 推理与 KV cache

### 11.1 三种 cache mode

| `cache_mode` | 用途 | 是否有 KV cache | 等价性 |
| :--- | :--- | :--- | :--- |
| `audit` | 调试/正确性审计 | 无,每步重算整个 prefix | 与训练 forward bit-equal |
| `mha_decode` | dense MHA baseline 的 KV decode | 标准 K/V cache | 与 audit 等价(数值容差内) |
| `deepseek_decode` | HCA/CSA 真正的增量 cache | HCA/CSA 各自的层级 cache | 数值与 prefill 等价 |

### 11.2 prefill / decode

`inference_autoregresive`(`inference/generate.py`)的两阶段流程:

1. **prefill**:把整段 prompt 喂给模型一次,对 `deepseek_decode` 还需用 `DeepSeekActiveCacheBuilder` 在每个 block 捕获 `RMSNorm(x)`,然后由 `inference/deepseek_cache_builder.py` 把这些层输入翻译成 HCA/CSA cache。
2. **decode**:每次只 forward 1 个 token,通过模型的 `forward_decode`,attention 模块的 `forward_decode` 增量更新 cache 并算出 1 个 logits。

`deepseek_prefill_mode="parallel"` 表示 prompt 一次并行处理(对应当前默认实现)。

### 11.3 HCA cache(`inference/hca_cache.py::HCALayerCache`)

包含三块状态:

- **compressed_kv**:已经"完结"的压缩 KV 块 $\bar C_{0:S}$;
- **pending_c / pending_z**:正在累积、还没满 $m'$ 的 token 状态(累满后 `flush_ready_blocks` 调用 compressor 压成一条加入 compressed_kv);
- **local_c / local_positions / local_valid_mask**:大小 $W$ 的滑窗精确 KV。

decode 时新 token 同时进入 pending(增加压缩块)与 local(滑窗滑动)。

### 11.4 CSA cache(`inference/csa_cache.py`)

CSA 比 HCA 多一路 a/b 压缩 + indexer 表示:

- `main_compressed`:a/b 压缩的全局 KV;
- `index_compressed`:indexer 用的低维压缩;
- `local`:同 HCA 的滑窗。

decode 时仍要先在 indexer 上选 top-$k$ 块,再做 attention。

> 仓库实现明确不允许把 raw MHA cache 和 HCA/CSA cache 混用(它们结构完全不同)。

### 11.5 采样(`inference/sampling.py`)

支持:

- `do_sample=False`:贪心 `argmax`;
- 否则 temperature → top-k → top-p(nucleus)→ 多项式采样;
- 可选 `repetition_penalty`、`presence_penalty`。

### 11.6 audit(`inference/audit.py`)

把同一 prompt 在 audit / mha_decode / deepseek_decode 三种模式下都跑一遍,逐 token 比对 logits 的最大差。这是验证"cache 实现是否正确"的核心测试基础。

---

## 12. 数据流水线

### 12.1 合成长上下文检索(`data/syntethic_long_context_retrieval.py`)

为 HCA / CSA 量身设计:每条样本是

```
key key_a is value_x  key key_b is value_y  ...  question : what is key_a ? answer : value_x
```

- key/value/filler 三个独立词表;
- filler 长度可控(`min_filler_tokens`、`max_filler_tokens`),把"被询问 key"的 value 推到很远;
- 模型必须**记住远处的 key→value 绑定**才能答对,直接考验长程依赖能力。

### 12.2 HF 文本数据集(`data/text_datasets.py`)

通过统一 preset 表加载 6 类公开语料,再自动分词、打 block、构成 `{input_ids, labels}` 的 dataloader:

| preset | 数据集 | 特点 |
| :--- | :--- | :--- |
| `tinystories` | `roneneldan/TinyStories` | 小儿故事,LM 收敛极快,适合 demo |
| `wikitext2` | `Salesforce/wikitext` | 经典 LM 基准 |
| `ag_news` | `fancyzhx/ag_news` | 新闻短文本 |
| `imdb` | `stanfordnlp/imdb` | 影评,长一些、领域偏移 |
| `minipile` | `JeanKaddour/minipile` | 多领域小型 pretraining mix |
| `fineweb_edu_10bt_mincols` | `EliMC/fineweb-edu-10BT-mincols` | FineWeb-Edu 教育子集采样 |

> **FineWeb / FineWeb-Edu 是什么?**FineWeb 是 HuggingFaceFW 发布的开源 CommonCrawl 网页 pretraining 语料(~15T token);**FineWeb-Edu** 是用分类器从 FineWeb 中筛出"教育/知识性"网页的子集(~1.3T token),小模型上质量普遍优于原始 FineWeb。本仓库用其 10B token 采样的 mincols 版本以适配研究规模。

### 12.3 Tokenizer

- TinyStories 用专用 BPE;
- 通用文本走 `tokenizers` 库即时训练或加载;
- 合成检索集用 word-level integer tokenizer。

### 12.4 inspect

`scripts/data_cli.py` 提供 `presets` / `synthetic-inspect` / `download` / `inspect-loader` 等命令,直接命令行查看 batch 形状、padding 比例、实际样本。

---

## 13. 并行计算

### 13.1 DDP 数据并行(`parallel/data_parallel.py`)

封装 `torch.distributed`:

- 自动 init,支持 NCCL / Gloo;
- `DistributedSampler` 在每个 rank 切分数据;
- rank-aware `save_checkpoint`(只 rank0 写盘);
- scalar metric 通过 `all_reduce` 聚合;
- CPU 单进程 gloo 也可跑(便于 CI/测试)。

### 13.2 模型并行(`parallel/model_parallel.py`,**层级 / 块级**)

不是 tensor parallel,而是**整块 block 放到不同 device**,activation 在边界搬运:

```python
model = wrap_model_parallel(model, devices=["cuda:0","cuda:1"], balance=[3,3])
optimizer = build_optimizer(model, train_config)  # 必须先 wrap 再建 optimizer
```

**V1 关键约束**:`balance` 每一项必须 > 0,即只能传"参与运算的 device",所以 `len(devices) ≤ n_layers`。这一条是常见踩坑点(详见 `AGENTS.md`)。

### 13.3 不实现的内容

定制 CUDA kernel、FP4/FP8 训练、NCCL 拓扑调度、DualPipe、真正的 expert all-to-all 并行——这些超出"小项目教学"的范围,仓库明确声明不做。

---

## 14. Ablation 实验体系

仓库把"一份 paper-faithful 实现"升级成"实验平台"。Ablation 入口 `ablations/suites.py`,共 6 组:

| 套件 | 研究问题 | 主要变体 |
| :--- | :--- | :--- |
| **A1** Hybrid Attention | CSA/HCA 混合是否优于 MHA / HCA-only / CSA-only? | dense_mha_baseline, hca_only, csa_only, hybrid_csa_hca, hybrid_hca_csa |
| **A2** 压缩 / 窗口 trade-off | `compression_factor`, `window_size`, `top_k_blocks` 的 grid | HCA & CSA 各自网格 |
| **A3** mHC 是否有用 | shallow vs deeper、有/无 mHC | MHA 与 hybrid 各做对照 |
| **A4** MoE 路由 | routed/shared expert、balance loss、hash 路由 | dense, MoE no_shared, shared_experts, no_balance, hash_routing |
| **A5** MTP 辅助损失 | 帮助还是干扰 next-token? | MTP off, depth/weight 扫描, 加权 depth |
| **A6** System-level stack | 各组件叠加效果 | baseline, +CSA/HCA, +MoE, +mHC, +MTP, full-minus 各项 |

`ablations/run_ablation.py` 顺序跑变体,每个 variant 跑完先 save checkpoint,再清空 Python/Torch/CUDA cache,避免变体间互相污染。结果落到 `outputs/ablations/{id}/{variant}/seed{n}/`,每变体一份 `final_metrics.json`,套件级 `summary.csv` + `summary.md`。

---

## 15. 工程实践与仓库约定

### 15.1 包结构与 import

`pyproject.toml::tool.setuptools.packages` 是**显式列表**,新增子包必须手动登记。`src` 是真实的 Python 包,所以 import 写法是 `from src.mini_deepseek_class import ...`(不是 layout root)。

`pythonpath = [".", "src"]` 让 pytest 同时能解析 `from src.X` 和 `from X`(后者用于 transformer_modules)。

### 15.2 测试约定

- 全部默认 CPU-safe,模型用极小尺寸(`d_model=16/32`、`n_layers=1/2`);
- GPU 测试用 `pytest.mark.skipif(not torch.cuda.is_available(), ...)` 或 `@pytest.mark.cuda`;
- Markers `slow`、`cuda` 在 `pyproject.toml` 注册,`--strict-markers` 开启;
- 顶层 `tests/test_*.py` 是 component 测试(model-tests CI job 用 `pytest tests/*.py` 这条 glob 不会扫子目录);
- 子目录 `tests/{data,training,inference,parallel,experiments}` 各自对应 CI 子任务。

### 15.3 CLI

`pip install -e .` 后会注册 5 个入口:

- `deepseekv4-data`:数据集 inspect / 下载 / tokenize;
- `deepseekv4-train`:tiny smoke 训练;
- `deepseekv4-inspect`:模型参数摘要 / 模块测试聚合;
- `deepseekv4-infer`:从 checkpoint 加载并生成;
- `deepseekv4-parallel`:plan / model-parallel-smoke / ddp-smoke;
- `deepseekv4-ablate`:跑 A1–A6。

不装 CLI 也可 `python -m scripts.<name>_cli ...`。

### 15.4 CI

`.github/workflows/ci.yml` 用 `dorny/paths-filter` 做**路径过滤**:只动了 `training/` 就只跑 training-tests;改 `pyproject.toml`、`requirements.txt`、`.github/workflows/**` 会触发**所有**子任务。可以根据"我改了哪些路径"反推应该手动跑哪些 `pytest`。

### 15.5 不要做的事

- 不要"修正" `chekpoints.py` 的拼写——所有 import 依赖此名;
- 不要把 `from src.X` 改成 `from X`;
- 不要提交 checkpoint、数据集、notebook 输出;
- 不要在 ablation runner 之外引入跨变体的全局状态(runner 间会清缓存)。

---

## 附录 A:符号约定

| 符号 | 含义 |
| :--- | :--- |
| $B$ | batch size |
| $T$ | 序列长度 |
| $D$ | hidden size(`d_model`) |
| $H$ | attention head 数 |
| $D_h$ | 每个 head 的维度(`head_dim`) |
| $V$ | vocab size |
| $L$ | 层数(`n_layers`) |
| $W$ | 局部滑窗大小(`window_size`) |
| $m'$ | 压缩因子(`compression_factor`) |
| $S$ | 压缩块总数,$S=\lceil T/m'\rceil$ |
| $k$ | CSA 的 top-k(`top_k_blocks`) |
| $N$ | MoE 专家总数(`num_experts`) |
| $K$ | MoE 每 token 激活专家数(`top_k_experts`) |
| $n_{\text{hc}}$ | mHC stream 数 |
| $D_{\text{mtp}}$ | MTP 深度 |

---

## 推荐学习路径速查

1. **想最快跑通**:`pip install -e ".[dev,data]"` → `pytest tests/test_deepseek_model.py` → `deepseekv4-train smoke --attention hca --ffn dense --max-batches 2`。
2. **想理解 attention**:依次读 `mha_baseline.py` → `deepseek_hca_attention.py` → `deepseek_csa_attention.py`,配合 `tests/test_hca.py`、`tests/test_csa.py` 看形状/因果检查。
3. **想理解 MoE**:`deepseek_moe.py` + `training/moe_metrics.py` + ablation A4。
4. **想理解残差结构**:`mHC_residuals.py` + ablation A3。
5. **想看完整训练循环**:`training/train_deepseek.py` → `train_one_epoch.py` → `eval_one_epoch.py`。
6. **想搞推理 cache**:先读 `docs/inference/kv_cache.md`,再看 `inference/hca_cache.py`、`inference/csa_cache.py`、`inference/deepseek_cache_builder.py`。

读完这份文档,你应该能回答:为什么 HCA 用 MQA?为什么 CSA 要双路 a/b 压缩?为什么 mHC 用 doubly stochastic 矩阵?Muon 为什么只用在 2D 参数?MTP 怎么和 LM head tie?HCA cache 的 pending 块什么时候 flush?——所有这些答案都已在仓库的源码与本文中对齐。
