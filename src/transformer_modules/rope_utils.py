"""
RoPE 旋转位置编码的工具函数。
核心操作: rotate_half —— 将向量的后半部分取负后与前半部分交换拼接，
实现二维旋转矩阵的乘法效果。
"""

from typing import Optional

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    将最后一维分成两半并旋转：[-x2, x1]
    这是 RoPE 公式中 "旋转" 的核心操作。
    
    例如 x = [a, b, c, d]，则 rotate_half(x) = [-c, -d, a, b]
    
    输入: [..., rotary_dim]（rotary_dim 必须是偶数）
    输出: [..., rotary_dim]
    """

    rotary_dim = x.shape[-1]

    if rotary_dim % 2 != 0:
        raise ValueError(
            f"rotate_half requires an even last dimension, got {rotary_dim}"
        )

    half = rotary_dim // 2

    x1 = x[..., :half]   # 前半部分
    x2 = x[..., half:]   # 后半部分

    # 拼接: [-后半, 前半]，实现90度旋转
    return torch.cat((-x2, x1), dim=-1)
