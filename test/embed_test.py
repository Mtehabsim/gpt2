import torch as t

from components.config import Config
from components.embed import Embed


def test_embed_shape():
    cfg = Config()
    embed = Embed(cfg)

    tokens = t.tensor(
        [
            [5, 10, 2],
            [8, 5, 1],
        ]
    )

    output = embed(tokens)

    assert output.shape == (2, 3, cfg.d_model)


def test_embed_lookup():
    cfg = Config()
    embed = Embed(cfg)

    tokens = t.tensor(
        [
            [5, 10, 2],
            [8, 5, 1],
        ]
    )

    output = embed(tokens)

    # Token ID 5 should retrieve row 5 of W_E
    assert t.equal(output[0, 0], embed.W_E[5])

    # Token ID 10 should retrieve row 10
    assert t.equal(output[0, 1], embed.W_E[10])

    # The same token ID should have the same embedding
    assert t.equal(output[0, 0], output[1, 1])
