import torch as t

from components import Config, PosEmbed


def test_pos_embed():
    cfg = Config()
    pos_embed = PosEmbed(cfg)

    tokens = t.tensor(
        [
            [5, 10, 2],
            [8, 5, 1],
        ]
    )

    output = pos_embed(tokens)

    assert output.shape == (2, 3, cfg.d_model)

    # Position 0 must have the same embedding in both sequences
    assert t.equal(output[0, 0], output[1, 0])

    # Position 1 must retrieve row 1 of W_pos
    assert t.equal(output[0, 1], pos_embed.W_pos[1])
