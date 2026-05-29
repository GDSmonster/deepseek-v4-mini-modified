# Autocast 与精度

精度相关的辅助代码位于 `training/autocast.py`。

## 作用

该模块集中管理 device 与 AMP 行为,使训练循环保持简洁。

它处理:

- 选择 `cpu`、`cuda` 或 `mps`,
- 解析所请求的 AMP dtype,
- 在必要时让 CUDA 上的 bf16 回退到 fp16,
- 决定是否需要 grad scaler,
- 将嵌套 batch 移动到 device,
- 暴露一个安全的 `autocast_ctx`。

## Device 解析

函数:`resolve_device`。

可接受的取值:

- `auto`:优先 CUDA,其次 MPS,最后 CPU。
- `cpu`
- `cuda`
- `mps`
- 显式的 `torch.device`。

如果请求 CUDA 或 MPS 但其不可用,会直接报错而非静默回退。

## AMP dtype

支持的字符串值:

```text
bf16, bfloat16
fp16, float16
fp32, float32
none
```

关键函数:

- `resolve_amp_dtype`:将字符串映射为 torch dtype。
- `get_effective_amp_dtype`:检查请求的 dtype 是否实际可用。
- `cuda_supports_bf16`:检测 CUDA 对 bf16 的支持。
- `should_use_grad_scaler`:仅 CUDA fp16 才使用 grad scaler。
- `make_grad_scaler`:在需要时创建 scaler。

## `setup_device_and_precision`

返回一个供 `train_one_epoch` 与 `eval_one_epoch` 使用的字典:

```python
{
    "device": resolved_device,
    "device_type": "cpu" | "cuda" | "mps",
    "amp_enabled": bool,
    "amp_dtype_requested": str,
    "amp_dtype_effective": torch.dtype | None,
    "use_grad_scaler": bool,
    "scaler": scaler_or_none,
    "cache_enabled": bool,
    "fallback_bf16_to_fp16": bool,
}
```

## 实用默认值

CPU 测试:

```python
setup_device_and_precision(device="cpu", amp_enabled=False)
```

CUDA bf16 训练:

```python
setup_device_and_precision(device="cuda", amp_enabled=True, amp_dtype="bf16")
```

CUDA fp16 训练:

```python
setup_device_and_precision(device="cuda", amp_enabled=True, amp_dtype="fp16")
```

## 为什么重要

CSA、HCA、mHC 与 MoE 的数值敏感路径比最小 Transformer 更多。集中管理精度行为可以更方便地:

- 在 CPU 冒烟测试中关闭 AMP,
- 在可用时使用 bf16,
- 避免不必要的 grad scaling,
- 保持 batch 移动行为一致,
- 防止模型 forward 抛错时出现 context manager 相关的 bug。
