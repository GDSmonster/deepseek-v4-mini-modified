# Muon Optimizer

Muon 是本项目中最重要的训练组件之一。

实现文件:

```text
training/muon_optimizer.py
```

## 它是什么

Muon 是一种针对 2D 矩阵参数的 optimizer。它先施加 momentum,再通过 Newton-Schulz 迭代对更新方向进行正交化。

在本仓库中,Muon 通过一个混合 optimizer 使用:

```text
HybridMuonAdamW = Muon 分支 + AdamW 分支
```

## 在 DeepSeek-V4 Mini 中的角色

DeepSeek-V4 论文对大多数矩阵型模型参数使用 Muon,而对不适合 Muon 的参数继续使用 AdamW。

本 mini 仓库沿用了这一思路:

- Muon 处理合适的 2D 隐藏矩阵。
- AdamW 处理 embedding、LM head、norm、bias、标量/向量参数,以及敏感的小型控制参数。

## Newton-Schulz 更新

函数:

```python
zeropower_via_newtonschulz5(G, steps=5, eps=1e-7)
```

作用:

- 接收一个 2D 梯度/更新矩阵,
- 对其进行归一化,
- 应用一次五次的 Newton-Schulz 迭代,
- 返回一个近似正交化的更新。

关键超参:

- `steps`:Newton-Schulz 迭代次数。
- `eps`:数值稳定性 epsilon。

## Muon 超参数

| 参数 | 含义 |
| :--- | :--- |
| `lr` / `muon_lr` | Muon 的 learning rate。 |
| `momentum` / `muon_momentum` | Momentum 系数。 |
| `weight_decay` / `muon_weight_decay` | Muon 分支的 decoupled weight decay。 |
| `nesterov` / `muon_nesterov` | 是否使用 Nesterov 风格 momentum。 |
| `ns_steps` / `muon_ns_steps` | Newton-Schulz 迭代次数。 |
| `eps` / `muon_eps` | 数值 epsilon。 |

## Hybrid Optimizer 构建器

函数:

```python
build_muon_adamw_optimizer(...)
```

重要参数:

| 参数 | 含义 |
| :--- | :--- |
| `learning_rate` | AdamW 的 LR,以及默认的 Muon LR。 |
| `muon_lr` | 可选的独立 Muon LR。 |
| `weight_decay` | AdamW 的 weight decay。 |
| `muon_weight_decay` | Muon 的 weight decay。 |
| `betas` | AdamW 的 beta 值。 |
| `eps` | AdamW 的 epsilon。 |
| `muon_momentum` | Muon momentum。 |
| `muon_nesterov` | Muon Nesterov 标志。 |
| `muon_ns_steps` | Newton-Schulz 迭代次数。 |
| `muon_eps` | Muon epsilon。 |

## 参数分组

Muon 应接收:

- 隐藏 Linear 权重,
- attention 投影矩阵,
- FFN/MoE 投影矩阵,
- 其他合适的 2D 内部变换。

AdamW 应接收:

- token embedding,
- LM head,
- norm 权重,
- bias,
- 标量/向量参数,
- mHC 静态/gating 参数,
- 小型 routing/控制参数。

## 实用配置

小规模调试通常使用 AdamW:

```yaml
optimizer_type: adamw
learning_rate: 0.0003
```

Muon 研究运行:

```yaml
optimizer_type: muon_adamw
learning_rate: 0.0003
muon_lr: null
muon_momentum: 0.95
muon_nesterov: true
muon_ns_steps: 5
muon_eps: 0.0000001
muon_weight_decay: 0.0
```

## 失败模式

- Muon 仅支持 2D 张量。
- 非有限梯度会抛出错误。
- 如果没有任何参数被分配给 Muon,构建器会报错,因为这通常意味着分组逻辑或架构假设有误。

## 为什么重要

Muon 在这里不仅仅是又一个 optimizer 选项。它是论文忠实训练故事的一部分:对大型矩阵参数使用不同的更新几何,与 AdamW 配合处理那些不应接受 Muon 更新的参数。
