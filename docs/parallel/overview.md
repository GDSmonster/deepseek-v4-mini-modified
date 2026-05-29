# 并行化指南

`parallel/` 包含本仓库的第一层 PyTorch 原生并行化实现。它有意保持显式且具教学性:这些代码展示了如何在不依赖自定义 CUDA kernel 或厂商专有运行时特性的前提下,将 data parallel 与粗粒度的模型放置叠加到 `DeepSeekV4LM` 之上。

## 已实现的内容

### Data Parallel

Data parallel 基于标准的 `torch.distributed` 与 `DistributedDataParallel` 实现。

可配置项:

| 参数 | 作用 |
| --- | --- |
| `mode` | 设为 `ddp` 以初始化分布式 data parallel 执行。 |
| `backend` | `gloo` 用于 CPU 安全测试与通用执行,`nccl` 用于 CUDA 多 GPU 运行。 |
| `init_method` | 进程组的 rendezvous 方法。在 `torchrun` 下使用 `env://`,在单进程 CPU 烟囱测试中可用 `file://...`。 |
| `find_unused_parameters` | 当存在可选分支、某些参数在每一步未必接收到梯度时使用。 |
| `gradient_as_bucket_view` | 在支持时让 DDP 梯度共享 bucket 内存。 |
| `broadcast_buffers` | 跨 rank 同步 buffer;在本模型中通常不需要。 |
| `static_graph` | 当 forward 图稳定时启用 DDP static graph 优化。 |
| `save_rank0_only` | 默认让 checkpoint 写入仅在 rank 0 上进行。 |

主要辅助函数:

- `setup_distributed(config)` 初始化进程组并返回本地设备。
- `wrap_ddp_model(model, config, device)` 将模型移动到设备并进行 wrap。
- `build_ddp_dataloader(...)` 创建带 `DistributedSampler` 的 dataloader。
- `ddp_train_one_epoch(...)` 与 `ddp_evaluate(...)` 复用现有训练/评估循环并对标量统计进行聚合。

## Model Parallel

Model parallel 以逐层/逐块的放置方式实现。整个 Transformer block 被分配到不同设备上,activation 在 block 边界之间移动。

这并非 tensor parallel、专家 all-to-all 路由、流水线调度或论文中生产级的并行运行时。它是一种保持模型结构完整的透明近似实现。

可配置项:

| 参数 | 作用 |
| --- | --- |
| `devices` | 有序的设备列表,例如 `cpu`、`cuda:0,cuda:1`,或任意合法 PyTorch 设备字符串。 |
| `balance` | 可选的逗号分隔逐设备层数,例如 `2,2,4`。 |
| `model_parallel_strategy` | 当前用于记录意图;实际实现的路径是逐层/逐块放置。 |

V1 有意不允许出现空的设备槽位。每个 `balance` 条目都必须大于零,因此设备数量必须小于或等于 `n_layers`。例如,`n_layers=2`、`devices=cpu,cpu,cpu,cpu`、`balance=1,1,0,0` 会被拒绝。请只传入活动设备,例如 `devices=cpu,cpu` 与 `balance=1,1`。

主要辅助函数:

- `infer_auto_balance(n_layers, n_devices)` 尽可能均匀地分配层。
- `build_block_device_map(n_layers, devices, balance)` 返回逐层放置计划。
- `ModelParallelDeepSeekV4LM(model, devices, balance)` 包裹现有的 `DeepSeekV4LM`。
- `wrap_model_parallel(...)` 是便捷构造函数。

### Optimizer 顺序

训练时,务必在 wrap 模型之后再构建 optimizer。Wrapper 会将子模块移动到目标设备,因此 optimizer 必须看到最终的参数对象与放置位置。

正确顺序:

```python
model = DeepSeekV4LM(config)
model = wrap_model_parallel(model, devices=["cuda:0", "cuda:1"], balance=[8, 8])
optimizer = build_optimizer(model, train_config)
```

避免如下顺序:

```python
model = DeepSeekV4LM(config)
optimizer = build_optimizer(model, train_config)
model = wrap_model_parallel(model, devices=["cuda:0", "cuda:1"], balance=[8, 8])
```

## CLI

查看放置计划:

```bash
python -m scripts.parallel_cli plan --n-layers 6 --devices cpu,cpu --balance 2,4
```

运行 CPU 安全的 model-parallel 烟囱测试:

```bash
python -m scripts.parallel_cli model-parallel-smoke --devices cpu --n-layers 2
```

运行单进程 CPU DDP 烟囱测试:

```bash
python -m scripts.parallel_cli ddp-smoke --backend gloo --n-layers 1
```

运行 CPU 安全的并行化测试:

```bash
python -m scripts.parallel_cli tests --quiet
```

editable 安装后,同样的命令以下列方式暴露:

```bash
deepseekv4-parallel plan --n-layers 6 --devices cpu,cpu
deepseekv4-parallel model-parallel-smoke --devices cpu
deepseekv4-parallel ddp-smoke --backend gloo
```

## 无需 CUDA 即可测试

仓库仅测试可在 CPU 上验证的部分:

- 配置校验。
- 单进程标量聚合。
- world size 1 下的分布式 sampler。
- 单进程 `gloo` DDP forward/backward。
- 当所有 block 都放置在 CPU 上时的 model-parallel 等价性。
- 通过 model-parallel wrapper 的 mHC 兼容性。

多 GPU NCCL 吞吐、跨设备 activation 传输开销以及真正的专家并行需要 CUDA 硬件,这些测试有意不予声明。
