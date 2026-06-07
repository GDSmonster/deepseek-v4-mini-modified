"""
Causal linear attention for the mini DeepSeek-V4 stack.

This module keeps the same projection/RoPE/output interface as the baseline
MHA module, but replaces softmax(QK^T)V with a positive feature map:

    K(q, k) ~= phi(q)^T phi(k)

The causal full-sequence path uses prefix sums, so it does not materialize the
[T, T] attention matrix unless debug weights are explicitly requested.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transformer_modules.rope import RotaryEmbedding


@dataclass
class CausalLinearAttentionConfig:
    d_model: int
    n_heads: int
    head_dim: Optional[int] = None

    attention_dropout: float = 0.0
    residual_dropout: float = 0.0

    use_bias: bool = False

    use_rope: bool = True
    rope_theta: float = 10000.0
    rotary_dim: Optional[int] = None

    max_seq_len: int = 1024
    init_std: float = 0.02
    feature_map: str = "elu"
    eps: float = 1e-6

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
                "For CausalLinearAttention, n_heads * head_dim must equal d_model. "
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

        if self.feature_map != "elu":
            raise ValueError(f"feature_map must be 'elu', got {self.feature_map}")

        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")

        if self.rotary_dim is not None:
            if self.rotary_dim <= 0:
                raise ValueError(f"rotary_dim must be > 0 when provided, got {self.rotary_dim}")
            if self.rotary_dim > head_dim:
                raise ValueError(
                    f"rotary_dim must be <= head_dim. "
                    f"Got rotary_dim={self.rotary_dim}, head_dim={head_dim}"
                )
            if self.rotary_dim % 2 != 0:
                raise ValueError(f"rotary_dim must be even, got {self.rotary_dim}")


class CausalLinearAttention(nn.Module):
    """
    Causal multi-head linear attention.

    The implementation is intended for research ablations. It is not a drop-in
    numerical equivalent of softmax attention; it changes the attention kernel.
    """

    def __init__(self, config: CausalLinearAttentionConfig):
        super().__init__()
        config.validate()

        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = (
            config.head_dim if config.head_dim is not None else config.d_model // config.n_heads
        )
        self.inner_dim = self.n_heads * self.head_dim
        self.max_seq_len = config.max_seq_len
        self.use_rope = config.use_rope
        self.eps = config.eps

        self.q_proj = nn.Linear(self.d_model, self.inner_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(self.d_model, self.inner_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(self.d_model, self.inner_dim, bias=config.use_bias)
        self.out_proj = nn.Linear(self.inner_dim, self.d_model, bias=config.use_bias)

        if self.use_rope:
            self.rope = RotaryEmbedding(
                dim=self.head_dim,
                rotary_dim=config.rotary_dim,
                base=config.rope_theta,
            )
        else:
            self.rope = None

        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _shape_projection(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim)

    def _validate_attention_mask(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        if attention_mask.dim() != 2:
            raise ValueError(
                f"attention_mask must have shape [B, T], got {tuple(attention_mask.shape)}"
            )

        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"attention_mask must have shape {(batch_size, seq_len)}, "
                f"got {tuple(attention_mask.shape)}"
            )

        return attention_mask

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0 + self.eps

    def _debug_attention_weights(
        self,
        q_phi: torch.Tensor,
        k_phi: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, _, T, _ = q_phi.shape
        scores = torch.matmul(q_phi, k_phi.transpose(-2, -1))

        causal_allowed = torch.tril(torch.ones(T, T, device=q_phi.device, dtype=torch.bool))
        allowed = causal_allowed[None, None, :, :].expand(B, self.n_heads, T, T)
        if attention_mask is not None:
            key_allowed = attention_mask[:, None, None, :].to(device=q_phi.device, dtype=torch.bool)
            allowed = allowed & key_allowed

        scores = scores * allowed.to(dtype=scores.dtype)
        denom = scores.sum(dim=-1, keepdim=True)
        return torch.where(
            denom > 0,
            scores / denom.clamp_min(torch.finfo(scores.dtype).tiny),
            torch.zeros_like(scores),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.dim() != 3:
            raise ValueError(
                f"CausalLinearAttention expects x with shape [B, T, d_model], "
                f"got {tuple(x.shape)}"
            )

        B, T, C = x.shape
        if C != self.d_model:
            raise ValueError(f"Expected x.shape[-1] == d_model={self.d_model}, got {C}")

        if T > self.max_seq_len:
            raise ValueError(f"Sequence length T={T} exceeds max_seq_len={self.max_seq_len}")

        if attention_mask is not None:
            attention_mask = self._validate_attention_mask(
                attention_mask=attention_mask,
                batch_size=B,
                seq_len=T,
            )

        q = self._shape_projection(self.q_proj(x))
        k = self._shape_projection(self.k_proj(x))
        v = self._shape_projection(self.v_proj(x))

        if self.rope is not None:
            q = self.rope(q, position_ids=position_ids, start_pos=start_pos)
            k = self.rope(k, position_ids=position_ids, start_pos=start_pos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q_phi = self._feature_map(q)
        k_phi = self._feature_map(k)

        if attention_mask is not None:
            key_mask = attention_mask[:, None, :, None].to(device=x.device, dtype=k_phi.dtype)
            k_phi = k_phi * key_mask
            v = v * key_mask

        q_acc = q_phi.float()
        k_acc = k_phi.float()
        v_acc = v.float()

        kv = torch.einsum("bhtd,bhte->bhtde", k_acc, v_acc).cumsum(dim=2)
        k_prefix = k_acc.cumsum(dim=2)

        context = torch.einsum("bhtd,bhtde->bhte", q_acc, kv)
        denom = torch.einsum("bhtd,bhtd->bht", q_acc, k_prefix).unsqueeze(-1)
        context = torch.where(
            denom > 0,
            context / denom.clamp_min(torch.finfo(context.dtype).tiny),
            torch.zeros_like(context),
        ).to(dtype=x.dtype)

        context = self.attention_dropout(context)
        context = context.transpose(1, 2).contiguous().view(B, T, self.inner_dim)
        out = self.out_proj(context)
        out = self.residual_dropout(out)

        if need_weights:
            return out, self._debug_attention_weights(q_phi, k_phi, attention_mask)

        return out
