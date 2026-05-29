# mHC 配置参考

主要配置:`ManifoldHyperConnectionConfig`。

模型层级字段在 `DeepSeekV4LMConfig` 中以 `mhc_` 为前缀。

| Model Field | mHC Field | Description |
| :--- | :--- | :--- |
| `use_mhc` | n/a | 在 attention 与 FFN 周围启用 mHC。 |
| `n_hc` | `n_hc` | 扩展残差流的数量。 |
| `mhc_sinkhorn_iters` | `sinkhorn_iters` | 用于约束残差矩阵 B 的 Sinkhorn 迭代次数。 |
| `mhc_eps` | `eps` | 数值 epsilon。 |
| `mhc_dynamic` | `dynamic` | 启用与输入相关的 A/B/C 生成。 |
| `mhc_expand_mode` | n/a | 从 `[B,T,D]` 扩展到 `[B,T,n_hc,D]` 的模式。 |
| `mhc_collapse_mode` | n/a | 折叠回 `[B,T,D]` 的模式:`mean`、`first`、`sum`、`readout`。 |
| `mhc_use_log_sinkhorn` | `use_log_sinkhorn` | 使用 log 空间的 Sinkhorn。 |
| `mhc_sinkhorn_fp32` | `sinkhorn_fp32` | 在 fp32 下计算 Sinkhorn。 |
| `mhc_init_alpha` | `init_alpha` | 初始动态贡献。 |
| `mhc_alpha_max` | `alpha_max` | 受限动态 alpha 的上界。 |
| `mhc_bounded_alpha` | `bounded_alpha` | 使用 tanh 限定 alpha。 |

更底层的静态初始化字段:

| Field | Description |
| :--- | :--- |
| `static_a_stream0` | 流 0 的 A 初始分数。 |
| `static_a_other` | 其他流的 A 初始分数。 |
| `static_b_diag` | B 对角线的初始分数。 |
| `static_b_offdiag` | B 非对角线的初始分数。 |
| `static_c_stream0` | 流 0 的 C 初始分数。 |
| `static_c_other` | 其他流的 C 初始分数。 |
| `init_std` | 动态生成器初始化尺度。 |

推荐的 CPU 极小起步:

```yaml
use_mhc: true
n_hc: 2
mhc_sinkhorn_iters: 5
mhc_collapse_mode: readout
```

参考论文风格的 mini 起步:

```yaml
use_mhc: true
n_hc: 4
mhc_sinkhorn_iters: 20
mhc_collapse_mode: readout
```
