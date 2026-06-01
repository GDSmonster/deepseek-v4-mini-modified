"""
RMSNorm (Root Mean Square Layer Normalization)
相比 LayerNorm，RMSNorm 去掉了均值中心化步骤，只做缩放归一化，计算更快。
公式: y = x / sqrt(mean(x^2) + eps) * weight
"""

import torch 
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    RMS 归一化层。
    - 输入: [..., D] 任意前缀维度，最后一维为特征维度
    - 输出: [..., D] 归一化后的张量，与输入同形状
    - weight: 可学习的逐元素缩放参数，初始为全1
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")

        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.dim = dim
        self.eps = eps
        # 可学习的缩放权重，初始化为全1（不改变原始分布）
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      if x.shape[-1] != self.dim:
          raise ValueError(
              f"Expected last dimension to be dim={self.dim}, "
              f"but got x.shape[-1]={x.shape[-1]}"
          )

      original_dtype = x.dtype

      # 转为 float32 计算 RMS，避免 fp16/bf16 下溢出
      x_float = x.float()

      # 计算均方值: mean(x^2)
      mean_square = x_float.pow(2).mean(dim=-1, keepdim=True)
      # rsqrt = 1/sqrt(mean_square + eps)
      inv_rms = torch.rsqrt(mean_square + self.eps)

      # 归一化: x * (1 / rms)
      y = x_float * inv_rms

      # 转回原始精度
      if y.dtype != original_dtype:
          y = y.to(original_dtype)

      # weight 也转为对应精度，避免输出被提升为 float32
      weight = self.weight.to(dtype=original_dtype)

      # 乘以可学习权重
      y = y * weight

      return y
