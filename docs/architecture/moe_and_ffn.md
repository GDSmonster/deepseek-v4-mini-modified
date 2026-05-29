# MoE 与 Dense FFN

每个 `DeepSeekV4Block` 的前馈分支由 `ffn_type` 选择。

支持的模式:

- `dense`:标准的 SwiGLU MLP。
- `moe`:DeepSeek 风格的路由/共享专家。

## Dense SwiGLU FFN

是什么:

- 一个使用 SwiGLU 风格激活的标准门控 MLP。

作用:

- 用于小型实验与消融的基线 FFN。
- 当 MoE 路由不是关注重点时适用。

关键超参数:

- `mlp_hidden_dim`:显式的隐藏宽度。若省略,宽度由 `mlp_expansion_factor` 推导。
- `mlp_expansion_factor`:从 `d_model` 到隐藏维度的乘数。
- `mlp_multiple_of`:将隐藏维度对齐到某个倍数,以获得更整洁的形状。
- `mlp_dropout`:MLP 内部的 dropout。
- `use_mlp_bias`:启用线性层 bias。

## DeepSeekMoE

是什么:

- 一个迷你的 DeepSeek 风格 mixture-of-experts FFN,具有路由专家、共享专家、可学习或哈希路由,以及负载均衡诊断。

作用:

- 增加稀疏的条件容量。
- 每个 token 仅激活 `top_k_experts` 个路由专家。
- 共享专家提供始终在线的容量。

主要超参数:

- `num_experts`:路由专家的总数。
- `top_k_experts`:每个 token 选择的路由专家数。
- `expert_hidden_dim`:每个路由专家内部的隐藏宽度。
- `expert_expansion_factor`:在未设置 `expert_hidden_dim` 时,从 `d_model` 推导路由专家宽度。
- `expert_multiple_of`:对齐专家隐藏宽度。
- `shared_experts`:始终评估的共享专家数量。
- `shared_hidden_dim`:共享专家的隐藏宽度。
- `shared_expansion_factor`:在未显式给出宽度时推导共享专家宽度。
- `router_type`:`learned` 或 `hash`。
- `router_score_fn`:打分非线性函数;支持的取值为 `softmax`、`sigmoid`、`sqrt_softplus`。
- `normalize_topk_weights`:将选中的专家权重归一化使其和为 1。
- `topk_weight_scale`:应用于选中专家权重的乘数。
- `router_jitter_noise`:训练时加到 router logits 上的随机噪声。
- `hash_routing_stride`:确定性哈希路由使用的 stride。
- `routed_scale`:对路由专家输出的缩放。
- `shared_scale`:对共享专家输出的缩放。
- `balance_loss_weight`:全局专家负载均衡损失的权重。
- `sequence_balance_loss_weight`:序列级均衡损失的权重。
- `dropout`、`use_bias`、`init_std`、`eps`:正则化、初始化与数值控制。

实践提示:

- 常规实验使用 `router_type="learned"`。
- 早期层确定性路由实验使用 `router_type="hash"`。
- 在 CPU 上保持 `num_experts` 较小,例如 4 或 8。
- 对 mini 模型而言,`top_k_experts=2` 是合理的默认值。
- 均衡损失是诊断/训练辅助手段,而不是论文中完整的工业级 auxiliary-loss-free 路由系统。
