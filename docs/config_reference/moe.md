# MoE 与 FFN 配置参考

## Dense FFN 参数

| Parameter | Description |
| :--- | :--- |
| `ffn_type="dense"` | 选择 dense SwiGLU FFN。 |
| `mlp_hidden_dim` | 显式的隐藏层宽度。 |
| `mlp_expansion_factor` | 当未设置 `mlp_hidden_dim` 时,从 `d_model` 推导隐藏宽度的倍率。 |
| `mlp_multiple_of` | 将隐藏宽度向上取整到该倍数。 |
| `mlp_dropout` | MLP 内部的 dropout。 |
| `use_mlp_bias` | 启用线性层的 bias。 |

## MoE 参数

| Parameter | Description |
| :--- | :--- |
| `ffn_type="moe"` | 选择 DeepSeekMoE FFN。 |
| `num_experts` | 路由 expert 数量。 |
| `top_k_experts` / `top_k` | 每个 token 激活的路由 expert 数量。 |
| `expert_hidden_dim` | 路由 expert 的显式隐藏层宽度。 |
| `expert_expansion_factor` | 路由 expert 的隐藏倍率(基于 `d_model`)。 |
| `expert_multiple_of` | 将 expert 隐藏宽度向上取整到该倍数。 |
| `shared_experts` | 始终启用的 shared expert 数量。 |
| `shared_hidden_dim` | shared expert 的显式隐藏层宽度。 |
| `shared_expansion_factor` | shared expert 的隐藏倍率(基于 `d_model`)。 |
| `router_type` | `learned` 或 `hash`。 |
| `router_score_fn` | `softmax`、`sigmoid` 或 `sqrt_softplus`。 |
| `normalize_topk_weights` | 对所选 expert 权重进行归一化。 |
| `topk_weight_scale` | 对所选 expert 权重做乘法缩放。 |
| `router_jitter_noise` | 在训练时向路由 logits 加入噪声。 |
| `hash_routing_stride` | 确定性 hash 路由所用的 stride。 |
| `routed_scale` | 路由 expert 的输出缩放。 |
| `shared_scale` | shared expert 的输出缩放。 |
| `balance_loss_weight` | 全局平衡辅助损失权重。 |
| `sequence_balance_loss_weight` | 序列粒度平衡辅助损失权重。 |
| `dropout` / `mlp_dropout` | expert 的 dropout。 |
| `use_bias` / `use_mlp_bias` | 启用投影 bias。 |
| `init_std` | 初始化尺度。 |
| `eps` | 路由归一化的数值 epsilon。 |

推荐的 CPU 极小起步:

```yaml
ffn_type: moe
num_experts: 4
top_k_experts: 2
expert_hidden_dim: 64
shared_experts: 1
shared_hidden_dim: 64
```
