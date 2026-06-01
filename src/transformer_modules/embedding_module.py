"""
Token Embedding 模块
将 token id 映射为稠密向量，支持 weight tying（与 LM head 共享权重）。
不包含位置编码——位置信息由后续 RoPE 在 attention 中注入。
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from typing import List, Dict, Tuple, Union
import torch.nn as nn


@dataclass
class EmbeddingConfig:
    vocab_size: int                     # 词表大小 V
    d_model: int                        # 嵌入维度 D

    pad_token_id: Optional[int] = None  # padding token 的 id，其嵌入向量固定为零
    max_seq_len: int = 1024             # 最大序列长度

    embedding_dropout: float = 0.0
    scale_embeddings: bool = False      # 是否乘以 sqrt(d_model)，某些模型使用此缩放
    
    init_std: float = 0.02
    tie_word_embeddings: bool = True    # 是否与 LM head 共享权重

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")

        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")

        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {self.max_seq_len}")

        if not (0.0 <= self.embedding_dropout < 1.0):
            raise ValueError(
                "embedding_dropout must satisfy 0 <= embedding_dropout < 1, "
                f"got {self.embedding_dropout}")

        if self.init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {self.init_std}")

        if self.pad_token_id is not None:
            if not (0 <= self.pad_token_id < self.vocab_size):
                raise ValueError(
                    "pad_token_id must satisfy 0 <= pad_token_id < vocab_size, "
                    f"got pad_token_id={self.pad_token_id}, "
                    f"vocab_size={self.vocab_size}")


class TokenEmbedding(nn.Module):
    """
    Token 嵌入层。
    
    功能: input_ids [B, T] -> hidden_states [B, T, d_model]
    
    特性:
    - 支持 padding_idx: 该位置的嵌入固定为零且不更新梯度
    - 支持 scale_embeddings: 乘以 sqrt(d_model)
    - 支持 weight tying: 通过 .weight 属性暴露参数供 LM head 复用
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__()

        config.validate()

        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.pad_token_id = config.pad_token_id
        self.max_seq_len = config.max_seq_len
        self.scale_embeddings = config.scale_embeddings

        # nn.Embedding: 查表操作，将整数 id 映射为 d_model 维向量
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,)

        self.dropout = nn.Dropout(config.embedding_dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """GPT 风格初始化: N(0, init_std)，pad 行强制为零"""
        nn.init.normal_(
            self.token_embedding.weight,
            mean=0.0,
            std=self.config.init_std,)

        if self.pad_token_id is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.pad_token_id].zero_()

    def forward(
          self,
          input_ids: Union[torch.Tensor, Dict[str, torch.Tensor]],) -> torch.Tensor:
          """
          输入: input_ids [B, T] 或包含 "input_ids" 键的字典
          输出: hidden_states [B, T, d_model]
          """

          # 兼容字典输入（如 HuggingFace DataLoader 的输出）
          if isinstance(input_ids, dict):
              if "input_ids" not in input_ids:
                  raise KeyError(
                      "TokenEmbedding received a dict batch but it does not contain "
                      f"'input_ids'. Available keys: {list(input_ids.keys())}"
                  )

              input_ids = input_ids["input_ids"]

          # 形状检查
          if not torch.is_tensor(input_ids):
              raise TypeError(
                  "TokenEmbedding expects either a tensor [B, T] or a dict containing "
                  f"'input_ids'. Got type: {type(input_ids)}"
              )

          if input_ids.dim() != 2:
              raise ValueError(
                  f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}"
              )

          _, seq_len = input_ids.shape

          if seq_len > self.max_seq_len:
              raise ValueError(
                  f"Sequence length T={seq_len} exceeds max_seq_len={self.max_seq_len}"
              )

          # 类型检查: 必须是整数类型
          if input_ids.dtype not in (torch.long, torch.int64, torch.int32):
              raise TypeError(
                  "input_ids must contain integer token indices; "
                  f"got dtype={input_ids.dtype}"
              )

          if input_ids.dtype != torch.long:
              input_ids = input_ids.long()

          # 范围检查
          if torch.any(input_ids < 0):
              min_id = int(input_ids.min().item())
              raise ValueError(
                  f"input_ids contain negative token ids. Minimum id found: {min_id}"
              )

          if torch.any(input_ids >= self.vocab_size):
              max_id = int(input_ids.max().item())
              raise ValueError(
                  "input_ids contain token ids >= vocab_size. "
                  f"Maximum id found: {max_id}, vocab_size={self.vocab_size}"
              )

          # 核心操作: 查表得到嵌入向量
          hidden_states = self.token_embedding(input_ids)

          # 可选缩放: 乘以 sqrt(d_model)，补偿嵌入值的方差
          if self.scale_embeddings:
              hidden_states = hidden_states * math.sqrt(self.d_model)

          hidden_states = self.dropout(hidden_states)

          return hidden_states

    @property
    def weight(self) -> nn.Parameter:
        """暴露嵌入权重，用于 LM head 的 weight tying"""
        return self.token_embedding.weight
