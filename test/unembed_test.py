import torch as t

from components import Config, Unembed


def test_unembed_shape():
    cfg = Config()
    unembed = Unembed(cfg)

    x = t.randn(2, 5, cfg.d_model)
    output = unembed(x)

    assert output.shape == (2, 5, cfg.d_vocab)
    assert t.isfinite(output).all()
