import pytest
import torch

from src.transformer_modules.linear_attention import *


def make_linear_config(**overrides):
    cfg = dict(
        d_model=64,
        n_heads=4,
        head_dim=16,
        attention_dropout=0.0,
        residual_dropout=0.0,
        use_bias=False,
        use_rope=True,
        rope_theta=10000.0,
        rotary_dim=16,
        max_seq_len=128,
        init_std=0.02,
    )
    cfg.update(overrides)
    return CausalLinearAttentionConfig(**cfg)


def make_linear_attention(**overrides):
    return CausalLinearAttention(make_linear_config(**overrides))


def make_input(B=2, T=8, D=64):
    return torch.randn(B, T, D)


def test_valid_linear_attention_config_builds():
    attn = make_linear_attention(d_model=64, n_heads=4, head_dim=16)

    assert attn.d_model == 64
    assert attn.n_heads == 4
    assert attn.head_dim == 16
    assert attn.inner_dim == 64


@pytest.mark.parametrize("field,value", [("feature_map", "relu"), ("eps", 0.0)])
def test_invalid_linear_attention_specific_config_raises(field, value):
    with pytest.raises(ValueError):
        make_linear_attention(**{field: value})


def test_linear_attention_output_shape_matches_input():
    attn = make_linear_attention()
    x = make_input(B=2, T=8, D=64)

    out = attn(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_linear_attention_returns_debug_weights_when_requested():
    attn = make_linear_attention(attention_dropout=0.0, residual_dropout=0.0)
    attn.eval()
    x = make_input(B=2, T=8, D=64)

    out, weights = attn(x, need_weights=True)

    assert out.shape == x.shape
    assert weights.shape == (2, 4, 8, 8)
    assert torch.isfinite(weights).all()


def test_linear_attention_debug_weights_are_causal():
    attn = make_linear_attention(attention_dropout=0.0, residual_dropout=0.0)
    attn.eval()
    x = make_input(B=2, T=8, D=64)

    _, weights = attn(x, need_weights=True)

    future_mask = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    assert torch.allclose(weights[:, :, future_mask], torch.zeros_like(weights[:, :, future_mask]))


def test_changing_future_tokens_does_not_change_past_outputs():
    attn = make_linear_attention(attention_dropout=0.0, residual_dropout=0.0)
    attn.eval()

    B, T, D = 2, 10, 64
    cut = 5
    x1 = make_input(B=B, T=T, D=D)
    x2 = x1.clone()
    x2[:, cut:, :] = torch.randn_like(x2[:, cut:, :])

    out1 = attn(x1)
    out2 = attn(x2)

    assert torch.allclose(out1[:, :cut, :], out2[:, :cut, :], atol=1e-5, rtol=1e-5)


def test_attention_mask_blocks_padding_keys_in_debug_weights():
    attn = make_linear_attention(attention_dropout=0.0, residual_dropout=0.0)
    attn.eval()
    x = make_input(B=2, T=8, D=64)
    attention_mask = torch.ones(2, 8)
    attention_mask[0, 3] = 0
    attention_mask[1, 5] = 0

    _, weights = attn(x, attention_mask=attention_mask, need_weights=True)

    assert torch.allclose(weights[0, :, :, 3], torch.zeros_like(weights[0, :, :, 3]))
    assert torch.allclose(weights[1, :, :, 5], torch.zeros_like(weights[1, :, :, 5]))
