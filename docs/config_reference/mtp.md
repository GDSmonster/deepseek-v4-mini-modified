# MTP 配置参考

主要配置:`MTPConfig`。

模型层级字段在 `DeepSeekV4LMConfig` 中以 `mtp_` 为前缀。

| Model Field | MTP Field | Description |
| :--- | :--- | :--- |
| `use_mtp` | n/a | 启用辅助 MTP head 与对应损失。 |
| `mtp_depth` | `mtp_depth` | 预测的未来 token 深度数量。 |
| `mtp_hidden_dim` | `hidden_dim` | 可选 MTP transform 的隐藏宽度。 |
| `use_mtp_transform` | `use_mtp_transform` | 在 MTP head 之前添加 transform。 |
| `mtp_activation` | `activation` | `silu`、`gelu`、`relu` 或 `identity`。 |
| `mtp_dropout` | `dropout` | MTP transform 中的 dropout。 |
| `use_mlp_bias` | `use_bias` | transform 层中的 bias。 |
| `mtp_loss_weight` | `mtp_loss_weight` | 全局 MTP 辅助损失系数。 |
| `mtp_tie_with_lm_head` | `tie_with_lm_head` | 将 MTP head 权重与 LM head 共享。 |
| `ignore_index` | `ignore_index` | 交叉熵忽略的标签值。 |
| `pad_token_id` | `pad_token_id` | 分词器 pad id。 |
| `mtp_depth_loss_weights` | `depth_loss_weights` | 可选的按深度损失权重。 |
| `mtp_validate_label_range` | `validate_label_range` | 校验标签 id 范围。 |

推荐起步配置:

```yaml
use_mtp: true
mtp_depth: 1
mtp_loss_weight: 0.3
use_mtp_transform: true
mtp_activation: silu
```
