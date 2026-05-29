# Inference 总览

Inference 栈围绕一个公开的封装函数设计:

```python
from inference import inference_autoregresive

out = inference_autoregresive(
    model=model,
    prompt="key key_1 is value_7 question : what is key_1 ? answer :",
    tokenizer=tokenizer,
    max_new_tokens=32,
    cache_mode="deepseek_decode",
    deepseek_prefill_mode="parallel",
    do_sample=False,
    return_cache_stats=True,
)
```

`prompt` 可以是文本、token id 或张量。文本 prompt 需要带有 `encode` 与 `decode` 方法的 tokenizer。token-id prompt 不需要 tokenizer。

## 主流水线

```text
inference_autoregresive(...)
    -> encode prompt
    -> generate(...)
        -> prefill(...)
        -> decode_step(...) repeated for new tokens
    -> optional decode back to text
```

该封装函数返回生成的 token id、可选的解码文本、cache 统计信息、计时指标以及可选的 MTP draft 诊断信息。

## 推荐模式

适用于 DeepSeek 风格的 HCA/CSA/混合模型:

```python
cache_mode="deepseek_decode"
deepseek_prefill_mode="parallel"
```

该模式将 prompt 整体送入模型一次,捕获每一层归一化后的 attention 输入,构建真实的 HCA/CSA 各层 cache,然后从这些 cache 出发逐 token 解码后续 token。

## Debug 模式

当需要逐 token 比较 cache 行为时使用:

```python
cache_mode="deepseek_decode"
deepseek_prefill_mode="sequential_debug"
```

该模式通过对每个 prompt token 调用一次 `forward_decode` 来填充 cache。速度较慢,但可用于验证 cache 的状态转换。

## CLI

同样的生成路径以 CLI 形式暴露:

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

安装后的控制台入口为:

```bash
deepseekv4-infer generate ...
```
