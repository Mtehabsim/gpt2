import torch as t

from components.config import Config
from components.demo_transformer import DemoTransformer


def test_demo_transformer():
    cfg = Config()
    model = DemoTransformer(cfg)

    tokens = t.tensor(
        [
            [10, 25, 7],
            [42, 10, 9],
        ]
    )

    with t.no_grad():
        logits = model(tokens)

    assert logits.shape == (2, 3, cfg.d_vocab)
    assert t.isfinite(logits).all()
