# mHC 残差流

mHC 表示 Manifold-Constrained Hyper-Connections(流形约束超连接)。

## 是什么

mHC 将残差流从:

```text
[B, T, D]
```

扩展为:

```text
[B, T, n_hc, D]
```

然后每次子层更新都由三个受约束的映射控制:

```text
X_next = B X + C F(A X)
```

其中:

- `A` 将扩展后的流混合为子层输入。
- `B` 混合各残差流,并被投影到一个双随机矩阵附近。
- `C` 将子层输出注入回扩展后的流。

## 在模型中的角色

- 用更丰富的残差路由机制取代简单的残差相加。
- 在不改变内部子层隐藏尺寸的前提下,为模型增加另一维度的容量。
- 提升项目对 DeepSeek-V4 论文的还原度。

## 主要超参数

- `use_mhc`:在 `DeepSeekV4Block` 内部启用 mHC。
- `n_hc`:残差流的数量。启用时至少为 2。
- `mhc_sinkhorn_iters`:对 `B` 进行 Sinkhorn 归一化的迭代次数。
- `mhc_eps`:归一化时使用的数值 epsilon。
- `mhc_dynamic`:启用基于输入的 A/B/C 动态生成。
- `mhc_expand_mode`:`[B,T,D]` 如何扩展成 `[B,T,n_hc,D]`。
- `mhc_collapse_mode`:扩展流如何被折叠回去。支持的取值包括 `mean`、`first`、`sum` 与 `readout`。
- `mhc_use_log_sinkhorn`:使用 log 空间的 Sinkhorn 变体。
- `mhc_sinkhorn_fp32`:为稳定性,强制以 fp32 进行 Sinkhorn 计算。
- `mhc_init_alpha`:动态 A/B/C 分量的初始强度。
- `mhc_alpha_max`:有界动态门控的上限。
- `mhc_bounded_alpha`:用 tanh 约束动态 alpha 门控。

更底层的初始化控制:

- `static_a_stream0`:A 中对 stream 0 的初始偏好。
- `static_a_other`:A 中对其他 stream 的初始打分。
- `static_b_diag`:B 的初始对角打分。
- `static_b_offdiag`:B 的初始非对角打分。
- `static_c_stream0`:对 stream 0 的初始注入。
- `static_c_other`:对其他 stream 的初始注入打分。
- `init_std`:动态生成器的初始化尺度。

实践提示:

- `n_hc=2` 适合 CPU 测试。
- `n_hc=4` 与 mini 论文启发的默认值一致。
- 更多 Sinkhorn 迭代会使 `B` 更接近双随机矩阵,但计算开销也更大。
- mHC 比经典残差更敏感;调试时请保持极小配置。
