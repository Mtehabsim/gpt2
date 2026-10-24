import torch as t

from components import Attention, Config


def test_attention_shape():
    cfg = Config()
    attention = Attention(cfg)

    x = t.randn(2, 5, cfg.d_model)
    output = attention(x)

    assert output.shape == x.shape
    assert t.isfinite(output).all()


def test_attention_is_causal():
    cfg = Config()
    attention = Attention(cfg)

    x = t.randn(1, 5, cfg.d_model)

    modified_x = x.clone()
    modified_x[:, -1, :] += 100

    original_output = attention(x)
    modified_output = attention(modified_x)

    assert t.allclose(
        original_output[:, :-1, :],
        modified_output[:, :-1, :],
        atol=1e-5,
    )
