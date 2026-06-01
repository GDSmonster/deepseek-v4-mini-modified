"""
RoPE (Rotary Positional Embedding) 旋转位置编码
核心思想: 将位置信息编码为旋转角度，让 attention score 自然依赖于相对位置。
公式: RoPE(x, pos) = x * cos(pos * theta) + rotate_half(x) * sin(pos * theta)
其中 theta_i = 1 / base^(2i / dim)，不同维度对应不同频率。
"""

import torch
import torch.nn as nn
from typing import Optional

from src.transformer_modules.rope_utils import *


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码模块。
    
    输入: x [B, T, H, D]  (batch, 序列长度, 头数, 头维度)
    输出: 同形状张量，已注入位置信息
    
    支持:
    - 全量 RoPE (rotary_dim == dim)
    - 部分 RoPE (rotary_dim < dim): 只对部分维度旋转，其余维度保留
    - 自动位置 (position_ids=None, 使用 start_pos 偏移)
    - 显式位置 (position_ids: [T] 或 [B, T])
    """

    def __init__(
        self,
        dim: int,
        rotary_dim: Optional[int] = None,
        base: float = 10000.0):

        super().__init__()

        if rotary_dim is None:
            rotary_dim = dim

        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")

        if rotary_dim <= 0:
            raise ValueError(f"rotary_dim must be > 0, got {rotary_dim}")

        if rotary_dim > dim:
            raise ValueError(
                f"rotary_dim must be <= dim, got rotary_dim={rotary_dim}, dim={dim}"
            )

        if rotary_dim % 2 != 0:
            raise ValueError(
                f"rotary_dim must be even, got rotary_dim={rotary_dim}"
            )

        if base <= 0:
            raise ValueError(f"base must be > 0, got {base}")

        self.dim = dim
        self.rotary_dim = rotary_dim
        self.base = base

        # 逆频率向量: inv_freq[i] = 1 / base^(2i / rotary_dim)
        # 低索引 -> 高频(捕捉局部位置), 高索引 -> 低频(捕捉远距离位置)
        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32)
                / rotary_dim
            ))

        # 注册为 buffer: 不参与梯度更新，但会随模型 .to(device) 移动
        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False)


    def _build_position_ids(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        position_ids: Optional[torch.Tensor],
        start_pos: int,) -> torch.Tensor:
        """构建或验证位置 id"""

        if position_ids is None:
            # 默认: 从 start_pos 开始的连续位置
            return torch.arange(
                start_pos,
                start_pos + seq_len,
                device=device,
                dtype=torch.float32)

        if position_ids.device != device:
            position_ids = position_ids.to(device)

        if position_ids.dim() == 1:
            if position_ids.shape[0] != seq_len:
                raise ValueError(
                    f"position_ids with shape [T] must have length T={seq_len}, "
                    f"got {position_ids.shape[0]}")

            return position_ids.float()

        if position_ids.dim() == 2:
            if position_ids.shape != (batch_size, seq_len):
                raise ValueError(
                    "position_ids with shape [B, T] must match input batch/length. "
                    f"Expected {(batch_size, seq_len)}, got {tuple(position_ids.shape)}")

            return position_ids.float()

        raise ValueError(
            "position_ids must be None, shape [T], or shape [B, T], "
            f"got shape {tuple(position_ids.shape)}")

    def _build_cos_sin(
        self,
        position_ids: torch.Tensor,
        target_dtype: torch.dtype,
        device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        根据位置计算 cos/sin 张量。
        position_ids [T] -> cos/sin [1, T, 1, rotary_dim]
        position_ids [B,T] -> cos/sin [B, T, 1, rotary_dim]
        """

        position_ids = position_ids.to(device=device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)

        # 外积: position * inv_freq -> 每个位置在每个频率上的角度
        # freqs: [T, rotary_dim//2] 或 [B, T, rotary_dim//2]
        freqs = position_ids[..., None] * inv_freq

        # 拼接成完整维度: [cos(f1), cos(f2), ..., cos(f1), cos(f2), ...]
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = torch.cos(emb)
        sin = torch.sin(emb)

        # 增加维度以便广播到 [B, T, H, rotary_dim]
        if position_ids.dim() == 1:
            cos = cos[None, :, None, :]  # [1, T, 1, R]
            sin = sin[None, :, None, :]
        else:
            cos = cos[:, :, None, :]     # [B, T, 1, R]
            sin = sin[:, :, None, :]

        cos = cos.to(dtype=target_dtype)
        sin = sin.to(dtype=target_dtype)

        return cos, sin

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        start_pos: int = 0) -> torch.Tensor:
        """
        对输入施加旋转位置编码。
        
        输入: x [B, T, H, D]
        输出: 同形状，已编码位置信息
        
        RoPE 公式: y = x * cos(theta) + rotate_half(x) * sin(theta)
        """

        if x.dim() != 4:
            raise ValueError(
                f"RotaryEmbedding expects x with shape [B, T, H, D], "
                f"got {tuple(x.shape)}")

        batch_size, seq_len, _, head_dim = x.shape

        if head_dim != self.dim:
            raise ValueError(
                f"Expected x.shape[-1] == dim={self.dim}, got {head_dim}")

        original_dtype = x.dtype
        device = x.device

        position_ids = self._build_position_ids(
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            position_ids=position_ids,
            start_pos=start_pos,)

        cos, sin = self._build_cos_sin(
            position_ids=position_ids,
            target_dtype=original_dtype,
            device=device)

        # 部分 RoPE: 只旋转后 rotary_dim 维，前面的维度直接保留
        pass_dim = self.dim - self.rotary_dim

        if pass_dim > 0:
            x_pass = x[..., :pass_dim]       # 不旋转的部分
            x_rot = x[..., pass_dim:]        # 要旋转的部分
        else:
            x_pass = None
            x_rot = x

        # 核心旋转: x * cos + rotate_half(x) * sin
        x_rotated = (x_rot * cos) + (rotate_half(x_rot) * sin)

        if x_pass is not None:
            y = torch.cat((x_pass, x_rotated), dim=-1)
        else:
            y = x_rotated

        return y
