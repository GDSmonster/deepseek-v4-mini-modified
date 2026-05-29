# 数据集指南

## 本地合成检索

适用场景:

- 无需任何下载。
- 需要确定性的长上下文 key-value 检索。
- 想快速对 CSA/HCA 进行烟囱测试。

CLI:

```bash
python -m scripts.data_cli synthetic-inspect --block-size 64 --batch-size 2
```

Python:

```python
from data.syntethic_long_context_retrieval import (
    SyntheticRetrievalConfig,
    create_synthetic_retrieval_dataloaders,
)

cfg = SyntheticRetrievalConfig(block_size=128, batch_size=4)
train_loader, val_loader, tokenizer = create_synthetic_retrieval_dataloaders(cfg)
```

## Hugging Face 文本 Preset

适用场景:

- 需要真实文本。
- 需要因果 LM batch。
- 希望通过文档数量限制让本地实验保持可控。

CLI:

```bash
python -m scripts.data_cli presets
python -m scripts.data_cli hf-info wikitext2
python -m scripts.data_cli hf-prepare wikitext2 --max-train-documents 1000
```

Python:

```python
from data.text_datasets import create_hf_text_dataloaders

train_loader, val_loader, tokenizer = create_hf_text_dataloaders(
    "wikitext2",
    block_size=256,
    batch_size=8,
    max_train_documents=20000,
    max_validation_documents=2000,
)
```

## Batch 约定

项目中所有 dataloader 应当产出以下两种格式之一:

```python
(input_ids, labels)
```

或:

```python
{
    "input_ids": input_ids,
    "labels": labels,
}
```

训练流程通过 `normalize_lm_batch` 对这些格式进行归一化。
