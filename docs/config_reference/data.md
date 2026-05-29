# 数据配置参考

本项目支持本地合成检索数据以及 Hugging Face 文本数据集。

## 合成检索配置

主要配置:`SyntheticRetrievalConfig`。

| Parameter | Description |
| :--- | :--- |
| `num_train_examples` | 生成的训练样本数量。 |
| `num_val_examples` | 生成的验证样本数量。 |
| `block_size` | 输入序列长度。标签向前 shift 一个 token。 |
| `min_filler_tokens` | 事实与问题之间的最少干扰/填充 token 数量。 |
| `max_filler_tokens` | 最多的干扰/填充 token 数量。 |
| `num_keys_per_example` | 每个样本中的 key-value 事实数量。 |
| `vocab_filler_size` | 填充 token 类型数量。 |
| `num_key_types` | 可能的 key token 数量。 |
| `num_value_types` | 可能的 value token 数量。 |
| `batch_size` | dataloader 批量大小。 |
| `num_workers` | dataloader worker 数量。 |
| `seed` | 生成器种子。 |

作用:

- 测试模型能否在干扰上下文中检索与某个 key 关联的 value。
- 适用于无需下载数据即可进行 CSA/HCA 长上下文冒烟测试。

## Hugging Face 文本数据集预设

在 `data/text_datasets.py` 中配置。

| Preset | Dataset | Description |
| :--- | :--- | :--- |
| `wikitext2` | `Salesforce/wikitext`, `wikitext-2-raw-v1` | 小型语言建模基准。 |
| `tinystories` | `roneneldan/TinyStories` | 易于生成的小型语料。 |
| `ag_news` | `fancyzhx/ag_news` | 新闻领域的紧凑文本。 |
| `imdb` | `stanfordnlp/imdb`, `plain_text` | 较长的影评文本。 |
| `minipile` | `JeanKaddour/minipile` | 小规模多样化的预训练混合语料。 |
| `fineweb_edu_10bt_mincols` | `EliMC/fineweb-edu-10BT-mincols` | 教育类网页样本;请使用文档数量限制。 |

## 通用 HF 加载器参数

函数:`create_hf_text_dataloaders`。

| Parameter | Description |
| :--- | :--- |
| `preset_name` | 数据集预设键。 |
| `block_size` | 因果 LM 的序列长度。默认采用预设推荐值。 |
| `batch_size` | dataloader 批量大小。 |
| `num_workers` | dataloader worker 数量。 |
| `tokenizer_path` | 字节级 BPE 分词器的保存/加载路径。 |
| `vocab_size` | 分词器词表大小。 |
| `min_frequency` | BPE token 的最小出现频率。 |
| `max_tokenizer_documents` | 限制用于训练分词器的文档数量。 |
| `max_train_documents` | 限制用于构建数据集的训练文档数量。 |
| `max_validation_documents` | 限制验证文档数量。 |

输出 batch 格式:

```python
{
    "input_ids": LongTensor[B, T],
    "labels": LongTensor[B, T],
}
```
