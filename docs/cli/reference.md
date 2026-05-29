# CLI 参考

本项目暴露六组命令。

editable 安装后:

```bash
deepseekv4-data
deepseekv4-train
deepseekv4-inspect
deepseekv4-infer
deepseekv4-ablate
deepseekv4-parallel
```

未安装时:

```bash
python -m scripts.data_cli
python -m scripts.train_cli
python -m scripts.inspect_cli
python -m scripts.inference_cli
python -m scripts.ablation_cli
python -m scripts.parallel_cli
```

## Data CLI

列出 preset:

```bash
python -m scripts.data_cli presets
```

检查合成数据:

```bash
python -m scripts.data_cli synthetic-inspect \
  --block-size 64 \
  --batch-size 2 \
  --num-train-examples 8
```

查看某个 HF preset:

```bash
python -m scripts.data_cli hf-info wikitext2
```

准备并检查 HF 数据:

```bash
python -m scripts.data_cli hf-prepare wikitext2 \
  --block-size 256 \
  --batch-size 8 \
  --max-tokenizer-documents 10000 \
  --max-train-documents 2000
```

## Train CLI

运行极简 CPU 烟囱训练:

```bash
python -m scripts.train_cli smoke \
  --attention mha \
  --ffn dense \
  --max-batches 1 \
  --quiet
```

试用 HCA:

```bash
python -m scripts.train_cli smoke \
  --attention hca \
  --ffn dense \
  --block-size 64 \
  --max-batches 2
```

试用 MoE:

```bash
python -m scripts.train_cli smoke \
  --attention csa \
  --ffn moe \
  --num-experts 4 \
  --top-k-experts 2
```

## Inspect CLI

模型摘要:

```bash
python -m scripts.inspect_cli model-summary --attention csa --ffn moe
```

针对某一模块组运行测试:

```bash
python -m scripts.inspect_cli module-tests csa --quiet
python -m scripts.inspect_cli module-tests training --quiet
python -m scripts.inspect_cli module-tests data --quiet
python -m scripts.inspect_cli module-tests inference --quiet
python -m scripts.inspect_cli module-tests ablations --quiet
```

## Inference CLI

使用 DeepSeek cache 从内置的手工 checkpoint 生成:

```bash
python -m scripts.inference_cli generate \
  --checkpoint outputs/deepseekv4_mini_muon_last_manual.pt \
  --config-json outputs/deepseekv4_mini_muon_last_manual.json \
  --prompt "key key_1 is value_7 question : what is key_1 ? answer :" \
  --synthetic-tokenizer \
  --cache-mode deepseek_decode \
  --deepseek-prefill-mode parallel \
  --max-new-tokens 16 \
  --no-do-sample \
  --return-cache-stats
```

无可用 tokenizer 时使用原始 token id:

```bash
python -m scripts.inference_cli generate \
  --checkpoint outputs/deepseekv4_mini_muon_last_manual.pt \
  --config-json outputs/deepseekv4_mini_muon_last_manual.json \
  --prompt-ids 1,4,5,6 \
  --cache-mode deepseek_decode \
  --max-new-tokens 8 \
  --no-do-sample
```

## Ablation CLI

运行 CPU 安全的烟囱消融:

```bash
python -m scripts.ablation_cli \
  --ablation A1 \
  --quick \
  --limit-variants 1 \
  --device cpu
```

依次生成全部 quick suite 运行:

```bash
python -m scripts.ablation_cli \
  --ablation ALL \
  --quick \
  --seeds 1 \
  --limit-variants 1
```

使用显式 seed 运行更大规模的 suite:

```bash
python -m scripts.ablation_cli \
  --ablation A6 \
  --seeds 1 2 3 \
  --device cuda \
  --max-batches-per-epoch 500 \
  --eval-max-batches 100
```

## Parallel CLI

查看 layer/device 放置计划:

```bash
python -m scripts.parallel_cli plan \
  --n-layers 6 \
  --devices cpu,cpu \
  --balance 2,4
```

运行 CPU 安全的 model-parallel forward:

```bash
python -m scripts.parallel_cli model-parallel-smoke \
  --devices cpu \
  --n-layers 2
```

使用 `gloo` 运行单进程 DDP 烟囱检查:

```bash
python -m scripts.parallel_cli ddp-smoke \
  --backend gloo \
  --n-layers 1
```

仅运行并行化相关测试:

```bash
python -m scripts.parallel_cli tests --quiet
```
