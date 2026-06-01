"""
SwiGLU MLP (Gated Linear Unit with SiLU activation)
DeepSeek / LLaMA 系列模型使用的 FFN 结构。
公式: out = down_proj( silu(gate_proj(x)) * up_proj(x) )
相比标准 FFN 多了一个门控分支，表达能力更强。
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def round_up_to_multiple(value: int, multiple_of: int) -> int:
    """将 value 向上取整到 multiple_of 的倍数（常用于对齐到 tensor core 友好的维度）"""
    if multiple_of <= 0:
        raise ValueError(f"multiple_of must be > 0, got {multiple_of}")

    return ((value + multiple_of - 1) // multiple_of) * multiple_of


@dataclass
class SwiGLUMLPConfig:
    d_model: int                        # 输入/输出维度

    hidden_dim: Optional[int] = None    # 中间层维度（若为 None 则由 expansion_factor 决定）
    expansion_factor: float = 4.0       # 中间层扩展倍数（hidden_dim = d_model * expansion_factor）
    multiple_of: int = 1                # 中间层维度对齐到此数的倍数

    dropout: float = 0.0
    use_bias: bool = False              # 通常 LLM 中不使用 bias
    init_std: float = 0.02

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")

        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0 when provided, got {self.hidden_dim}"
            )

        if self.expansion_factor <= 0:
            raise ValueError(
                f"expansion_factor must be > 0, got {self.expansion_factor}"
            )

        if self.multiple_of <= 0:
            raise ValueError(f"multiple_of must be > 0, got {self.multiple_of}")

        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(
                f"dropout must satisfy 0 <= dropout < 1, got {self.dropout}"
            )

        if self.init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {self.init_std}")

    def resolved_hidden_dim(self) -> int:
        """计算最终的中间层维度"""
        self.validate()

        if self.hidden_dim is None:
            hidden_dim = int(self.expansion_factor * self.d_model)
        else:
            hidden_dim = self.hidden_dim

        hidden_dim = round_up_to_multiple(hidden_dim, self.multiple_of)

        return hidden_dim


class SwiGLUMLP(nn.Module):
    """
    SwiGLU 前馈网络。
    
    结构:
        gate = gate_proj(x)        # 门控分支
        up   = up_proj(x)          # 值分支
        hidden = silu(gate) * up   # 门控激活后与值相乘
        out = down_proj(hidden)    # 降维回 d_model
    
    输入: [B, T, d_model]
    输出: [B, T, d_model]
    
    注意: 本模块不包含残差连接和归一化，这些在 TransformerBlock 中处理。
    """

    def __init__(self, config: SwiGLUMLPConfig):
        super().__init__()

        config.validate()

        self.config = config
        self.d_model = config.d_model
        self.hidden_dim = config.resolved_hidden_dim()

        # 门控分支: d_model -> hidden_dim
        self.gate_proj = nn.Linear(
            self.d_model,
            self.hidden_dim,
            bias=config.use_bias)

        # 值分支: d_model -> hidden_dim
        self.up_proj = nn.Linear(
            self.d_model,
            self.hidden_dim,
            bias=config.use_bias)

        # 降维: hidden_dim -> d_model
        self.down_proj = nn.Linear(
            self.hidden_dim,
            self.d_model,
            bias=config.use_bias)

        self.dropout = nn.Dropout(config.dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """正态分布初始化，bias 初始化为零"""
        for module in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.init_std,)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"SwiGLUMLP expects x with shape [B, T, d_model], "
                f"got {tuple(x.shape)}")

        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected x.shape[-1] == d_model={self.d_model}, "
                f"got {x.shape[-1]}")

        gate = self.gate_proj(x)    # 门控信号
        up = self.up_proj(x)        # 值信号

        # SwiGLU 核心: silu(gate) * up
        # silu(x) = x * sigmoid(x)，是一种平滑的门控激活
        hidden = F.silu(gate) * up

        out = self.down_proj(hidden)
        out = self.dropout(out)

        return out
