# 多 token 预测(Multi-Token Prediction)

MTP 表示 Multi-Token Prediction(多 token 预测)。

## 是什么

主 LM head 预测下一个 token。MTP 增加辅助 head,用于预测更远的未来 token:

```text
主 LM head: token t + 1
MTP head 0:  token t + 2
MTP head 1:  token t + 3
...
```

## 在模型中的角色

- 提供额外的自回归监督信号。
- 鼓励隐藏状态编码不仅对下一个 token 有用的信息。
- 在 mini 规模上模拟 DeepSeek-V3/V4 的训练策略。

## 主要超参数

- `use_mtp`:启用 MTP head 与损失。
- `mtp_depth`:辅助预测深度的数量。
- `mtp_hidden_dim`:可选 MTP 变换内部的隐藏尺寸。
- `use_mtp_transform`:在每个 MTP head 之前插入一个小的变换。
- `mtp_activation`:变换中的激活函数。支持的取值:`silu`、`gelu`、`relu`、`identity`。
- `mtp_dropout`:MTP 变换中的 dropout。
- `mtp_loss_weight`:MTP 辅助损失的全局乘数。
- `mtp_tie_with_lm_head`:让 MTP head 与主 LM head 共享权重。
- `mtp_depth_loss_weights`:可选的逐深度损失加权。
- `mtp_validate_label_range`:对 MTP 标签校验 vocab 范围与 ignore index。
- `ignore_index`:被交叉熵忽略的目标值。
- `pad_token_id`:tokenizer 的 pad id,在构造 label 时使用。

实践提示:

- 从 `mtp_depth=1` 或 `2` 开始。
- 对论文启发的 mini 实验保持 `mtp_loss_weight=0.3`。
- 如果训练不稳定,先降低 `mtp_loss_weight`,再考虑完全关闭 MTP。
