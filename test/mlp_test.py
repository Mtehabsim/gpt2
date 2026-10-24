import torch as t

from components import MLP, Config


def test_mlp_shape():
    cfg = Config()
    mlp = MLP(cfg)

    x = t.randn(2, 4, cfg.d_model)
    output = mlp(x)

    assert output.shape == x.shape
    assert t.isfinite(output).all()


def test_mlp_gradients():
    cfg = Config()
    mlp = MLP(cfg)

    x = t.randn(2, 4, cfg.d_model)
    output = mlp(x)

    output.sum().backward()

    assert mlp.W_in.grad is not None
    assert mlp.W_out.grad is not None
