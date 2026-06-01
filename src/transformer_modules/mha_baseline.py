"""
标准因果多头注意力 (Causal Multi-Head Attention)
这是 Transformer 的核心组件，实现 self-attention + 因果 mask。
流程: x -> Q/K/V投影 -> RoPE -> scaled dot-product attention -> 输出投影
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transformer_modules.rope import *


@dataclass
class CausalMHAConfig:
    d_model: int                            # 模型维度
    n_heads: int                            # 注意力头数

    head_dim: Optional[int] = None          # 每个头的维度（默认 d_model // n_heads）

    attention_dropout: float = 0.0          # attention 权重上的 dropout
    residual_dropout: float = 0.0           # 输出投影后的 dropout

    use_bias: bool = False                  # Q/K/V/O 投影是否使用 bias

    use_rope: bool = True                   # 是否使用旋转位置编码
    rope_theta: float = 10000.0             # RoPE 的基频
    rotary_dim: Optional[int] = None        # RoPE 作用的维度数（默认全部）

    max_seq_len: int = 1024
    init_std: float = 0.02

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")

        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {self.n_heads}")

        if self.head_dim is None:
            if self.d_model % self.n_heads != 0:
                raise ValueError(
                    "If head_dim is None, d_model must be divisible by n_heads. "
                    f"Got d_model={self.d_model}, n_heads={self.n_heads}"
                )
            head_dim = self.d_model // self.n_heads
        else:
            head_dim = self.head_dim

        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")

        inner_dim = self.n_heads * head_dim

        if inner_dim != self.d_model:
            raise ValueError(
                "For baseline CausalMHA, n_heads * head_dim must equal d_model. "
                f"Got n_heads={self.n_heads}, head_dim={head_dim}, "
                f"inner_dim={inner_dim}, d_model={self.d_model}"
            )

        if not (0.0 <= self.attention_dropout < 1.0):
            raise ValueError(
                "attention_dropout must satisfy 0 <= attention_dropout < 1, "
                f"got {self.attention_dropout}"
            )

        if not (0.0 <= self.residual_dropout < 1.0):
            raise ValueError(
                "residual_dropout must satisfy 0 <= residual_dropout < 1, "
                f"got {self.residual_dropout}"
            )

        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {self.max_seq_len}")

        if self.init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {self.init_std}")

        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be > 0, got {self.rope_theta}")

        if self.rotary_dim is not None:
            if self.rotary_dim <= 0:
                raise ValueError(
                    f"rotary_dim must be > 0 when provided, got {self.rotary_dim}"
                )

            if self.rotary_dim > head_dim:
                raise ValueError(
                    f"rotary_dim must be <= head_dim. "
                    f"Got rotary_dim={self.rotary_dim}, head_dim={head_dim}"
                )

            if self.rotary_dim % 2 != 0:
                raise ValueError(
                    f"rotary_dim must be even, got {self.rotary_dim}"
                )


class CausalMultiHeadAttention(nn.Module):
    """
    因果多头自注意力。
    
    输入: x [B, T, d_model]
    输出: out [B, T, d_model]
    
    核心流程:
    1. 线性投影得到 Q, K, V  (每个 [B, T, n_heads, head_dim])
    2. 对 Q, K 施加 RoPE 位置编码
    3. 计算 attention score: Q @ K^T / sqrt(head_dim)
    4. 应用因果 mask (上三角为 -inf，确保只能看到之前的 token)
    5. softmax 归一化得到 attention 权重
    6. 加权求和: attention_weights @ V
    7. 合并多头后线性投影输出
    """

    def __init__(self, config: CausalMHAConfig):
        super().__init__()

        config.validate()

        self.config = config

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = (
            config.head_dim
            if config.head_dim is not None
            else config.d_model // config.n_heads)

        self.inner_dim = self.n_heads * self.head_dim
        self.max_seq_len = config.max_seq_len

        self.use_rope = config.use_rope

        # Q/K/V/O 四个线性投影
        self.q_proj = nn.Linear(self.d_model, self.inner_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(self.d_model, self.inner_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(self.d_model, self.inner_dim, bias=config.use_bias)
        self.out_proj = nn.Linear(self.inner_dim, self.d_model, bias=config.use_bias)

        # RoPE 模块（只作用于 Q 和 K）
        if self.use_rope:
            self.rope = RotaryEmbedding(
                dim=self.head_dim,
                rotary_dim=config.rotary_dim,
                base=config.rope_theta,)
        else:
            self.rope = None

        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """正态分布初始化所有投影层"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _shape_projection(self, x: torch.Tensor) -> torch.Tensor:
        """将投影结果 reshape: [B, T, inner_dim] -> [B, T, n_heads, head_dim]"""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        构建因果 mask（上三角矩阵）。
        True 表示该位置被遮蔽（不允许 attend）。
        即: token t 只能看到位置 <= t 的 key。
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,)

    def _validate_attention_mask(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        seq_len: int) -> torch.Tensor:
        if attention_mask.dim() != 2:
            raise ValueError(
                f"attention_mask must have shape [B, T], "
                f"got {tuple(attention_mask.shape)}")

        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"attention_mask must have shape {(batch_size, seq_len)}, "
                f"got {tuple(attention_mask.shape)}")

        return attention_mask

    def _safe_masked_softmax(
      self,
      scores: torch.Tensor,
      allowed_mask: torch.Tensor,
      dim: int = -1) -> torch.Tensor:
      """
      安全的带 mask 的 softmax。
      处理边界情况: 当某一行所有 key 都被 mask 时，输出精确的零向量（而非 NaN）。
      """

      if allowed_mask.dtype != torch.bool:
          allowed_mask = allowed_mask.bool()

      mask_value = torch.finfo(scores.dtype).min

      # 被 mask 的位置填充极小值
      masked_scores = scores.masked_fill(~allowed_mask, mask_value)

      # fp32 softmax 保证数值稳定
      weights = F.softmax(masked_scores.float(), dim=dim).to(dtype=scores.dtype)

      # 清除 mask 位置可能残留的概率
      weights = weights * allowed_mask.to(dtype=weights.dtype)

      # 重新归一化（处理全 mask 行）
      denom = weights.sum(dim=dim, keepdim=True)
      weights = torch.where(
          denom > 0,
          weights / denom.clamp_min(torch.finfo(weights.dtype).tiny),
          torch.zeros_like(weights))

      return weights


    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        need_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播。
        
        参数:
            x: 隐藏状态 [B, T, d_model]
            attention_mask: 可选 [B, T]，1=有效token，0=padding
            position_ids: 可选 RoPE 位置
            start_pos: RoPE 偏移量
            need_weights: 是否返回 attention 权重
        
        返回:
            out [B, T, d_model]，可选返回 attn_weights [B, H, T, T]
        """

        if x.dim() != 3:
            raise ValueError(
                f"CausalMultiHeadAttention expects x with shape [B, T, d_model], "
                f"got {tuple(x.shape)}")

        B, T, C = x.shape

        if C != self.d_model:
            raise ValueError(
                f"Expected x.shape[-1] == d_model={self.d_model}, got {C}")

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length T={T} exceeds max_seq_len={self.max_seq_len}")

        if attention_mask is not None:
            attention_mask = self._validate_attention_mask(
                attention_mask=attention_mask,
                batch_size=B,
                seq_len=T,)

        # === Q/K/V 投影 ===
        q = self._shape_projection(self.q_proj(x))  # [B, T, H, Dh]
        k = self._shape_projection(self.k_proj(x))  # [B, T, H, Dh]
        v = self._shape_projection(self.v_proj(x))  # [B, T, H, Dh]

        # === RoPE: 给 Q 和 K 注入位置信息 ===
        if self.rope is not None:
            q = self.rope(q, position_ids=position_ids, start_pos=start_pos)
            k = self.rope(k, position_ids=position_ids, start_pos=start_pos)

        # 转置为 [B, H, T, Dh] 以便计算 attention score
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # === Scaled Dot-Product Attention ===
        # score = Q @ K^T / sqrt(d_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        # === 因果 mask: 上三角填充 -inf ===
        causal_mask = self._build_causal_mask(seq_len=T, device=x.device)
        mask_value = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(
            causal_mask[None, None, :, :],  # 广播到 [B, H, T, T]
            mask_value)

        # === 可选的 padding mask ===
        if attention_mask is not None:
            key_padding_mask = attention_mask[:, None, None, :].to(
                device=x.device, dtype=torch.bool)  # [B, 1, 1, T]
            attn_scores = attn_scores.masked_fill(~key_padding_mask, mask_value)

        # === Softmax (fp32 保证稳定性) ===
        attn_weights = F.softmax(attn_scores.float(), dim=-1).to(dtype=attn_scores.dtype)
        attn_weights = self.attention_dropout(attn_weights)

        # === 加权求和: attention_weights @ V ===
        context = torch.matmul(attn_weights, v)  # [B, H, T, Dh]

        # === 合并多头: [B, H, T, Dh] -> [B, T, H*Dh] ===
        context = context.transpose(1, 2).contiguous()
        context = context.view(B, T, self.inner_dim)

        # === 输出投影 ===
        out = self.out_proj(context)
        out = self.residual_dropout(out)

        if need_weights:
            return out, attn_weights

        return out

    def forward_decode(
        self,
        x_t: torch.Tensor,
        cache,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        """
        解码阶段的单步推理（使用 KV cache）。
        输入: x_t [B, 1, d_model] 当前时间步的隐藏状态
        """
        if x_t.dim() != 3 or x_t.shape[1] != 1:
            raise ValueError(f"forward_decode expects x_t [B,1,D], got {tuple(x_t.shape)}")

        B, T, C = x_t.shape
        if C != self.d_model:
            raise ValueError(f"Expected hidden size {self.d_model}, got {C}")

        q = self._shape_projection(self.q_proj(x_t))
        k = self._shape_projection(self.k_proj(x_t))
        v = self._shape_projection(self.v_proj(x_t))

        if self.rope is not None:
            q = self.rope(q, position_ids=position_ids)
            k = self.rope(k, position_ids=position_ids)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 将当前步的 K, V 追加到 cache 中
        cache.append(k, v, position_ids)
        # 获取完整的 K, V 历史
        k_all, v_all = cache.get_kv()

        # 当前 query 对所有历史 key 做 attention
        attn_scores = torch.matmul(q, k_all.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.shape[0] != B:
                raise ValueError(
                    "decode attention_mask must have shape [B,T_cache], "
                    f"got {tuple(attention_mask.shape)}"
                )
            key_padding_mask = attention_mask[:, None, None, :].to(
                device=x_t.device, dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(
                ~key_padding_mask,
                torch.finfo(attn_scores.dtype).min)

        attn_weights = F.softmax(attn_scores.float(), dim=-1).to(dtype=attn_scores.dtype)
        attn_weights = self.attention_dropout(attn_weights)
        context = torch.matmul(attn_weights, v_all)
        context = context.transpose(1, 2).contiguous().view(B, T, self.inner_dim)
        out = self.out_proj(context)
        out = self.residual_dropout(out)

        aux = {"attn_weights": attn_weights} if need_weights else {}
        return out, cache, aux
