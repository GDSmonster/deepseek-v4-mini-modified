"""
Transformer Block (Pre-Norm 结构)
标准的 Transformer 层，采用 Pre-Norm 残差连接:
    x = x + Attention(Norm(x))
    x = x + MLP(Norm(x))
这是 DeepSeek / LLaMA 等现代 LLM 普遍采用的结构。
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Union

import torch
import torch.nn as nn


from src.transformer_modules.SwiGLU import * 
from src.transformer_modules.RMSNorm import *
from src.transformer_modules.mha_baseline import *
from src.transformer_modules.embedding_module import *


@dataclass
class TransformerBlockConfig:
    d_model: int                            # 模型维度
    rms_norm_eps: float = 1e-6              # RMSNorm 的 epsilon

    # --- 注意力相关配置 ---
    n_heads: int = 4
    head_dim: Optional[int] = None
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    use_attention_bias: bool = False
    use_rope: bool = True
    rope_theta: float = 10000.0
    rotary_dim: Optional[int] = None
    max_seq_len: int = 1024

    # --- MLP 相关配置 ---
    mlp_hidden_dim: Optional[int] = None
    mlp_expansion_factor: float = 4.0
    mlp_multiple_of: int = 1
    mlp_dropout: float = 0.0
    use_mlp_bias: bool = False

    # --- 初始化 ---
    init_std: float = 0.02

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")

        if self.rms_norm_eps <= 0:
            raise ValueError(
                f"rms_norm_eps must be > 0, got {self.rms_norm_eps}"
            )

        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {self.max_seq_len}")

        if self.init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {self.init_std}")

        attention_config = self.to_attention_config()
        attention_config.validate()

        mlp_config = self.to_mlp_config()
        mlp_config.validate()

        if attention_config.d_model != self.d_model:
            raise ValueError(
                "attention_config.d_model must match block d_model. "
                f"Got {attention_config.d_model} vs {self.d_model}"
            )

        if mlp_config.d_model != self.d_model:
            raise ValueError(
                "mlp_config.d_model must match block d_model. "
                f"Got {mlp_config.d_model} vs {self.d_model}"
            )

    def to_attention_config(self) -> "CausalMHAConfig":
        """转换为注意力模块的配置"""
        return CausalMHAConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            attention_dropout=self.attention_dropout,
            residual_dropout=self.residual_dropout,
            use_bias=self.use_attention_bias,
            use_rope=self.use_rope,
            rope_theta=self.rope_theta,
            rotary_dim=self.rotary_dim,
            max_seq_len=self.max_seq_len,
            init_std=self.init_std)

    def to_mlp_config(self) -> "SwiGLUMLPConfig":
        """转换为 MLP 模块的配置"""
        return SwiGLUMLPConfig(
            d_model=self.d_model,
            hidden_dim=self.mlp_hidden_dim,
            expansion_factor=self.mlp_expansion_factor,
            multiple_of=self.mlp_multiple_of,
            dropout=self.mlp_dropout,
            use_bias=self.use_mlp_bias,
            init_std=self.init_std,)


class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer Block。
    
    结构 (Pre-Norm + 残差连接):
        x = x + Attention(RMSNorm(x))   # 注意力子层
        x = x + SwiGLU(RMSNorm(x))      # 前馈子层
    
    输入: x [B, T, d_model]
    输出: x [B, T, d_model]
    
    注意: 本模块是"干净"的基线实现，不包含 MoE/mHC/HCA/CSA 等 DeepSeek 创新组件。
    """

    def __init__(self, config: TransformerBlockConfig):
        super().__init__()

        config.validate()

        self.config = config
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len

        # 注意力子层的 pre-norm
        self.norm1 = RMSNorm(dim=config.d_model, eps=config.rms_norm_eps)
        # 因果多头注意力
        self.attention = CausalMultiHeadAttention(config.to_attention_config())

        # MLP 子层的 pre-norm
        self.norm2 = RMSNorm(dim=config.d_model, eps=config.rms_norm_eps)
        # SwiGLU 前馈网络
        self.mlp = SwiGLUMLP(config.to_mlp_config())

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        need_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        参数:
            x: 隐藏状态 [B, T, d_model]
            attention_mask: 可选 [B, T]，1=有效，0=padding
            position_ids: 可选 RoPE 位置
            start_pos: RoPE 偏移
            need_weights: 是否返回 attention 权重
        """

        if x.dim() != 3:
            raise ValueError(
                f"TransformerBlock expects x with shape [B, T, d_model], "
                f"got {tuple(x.shape)}"
            )

        B, T, C = x.shape

        if C != self.d_model:
            raise ValueError(
                f"Expected x.shape[-1] == d_model={self.d_model}, got {C}"
            )

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length T={T} exceeds max_seq_len={self.max_seq_len}"
            )

        # === 注意力子层: Pre-Norm + Attention + 残差 ===
        residual = x
        x_norm = self.norm1(x)

        attn_result = self.attention(
            x_norm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            start_pos=start_pos,
            need_weights=need_weights)

        if need_weights:
            attn_out, attn_weights = attn_result
        else:
            attn_out = attn_result
            attn_weights = None

        x = residual + attn_out  # 残差连接

        # === MLP 子层: Pre-Norm + SwiGLU + 残差 ===
        residual = x
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out   # 残差连接

        if need_weights:
            aux = {"attn_weights": attn_weights}
            return x, aux

        return x
