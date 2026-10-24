import torch as t
import torch.nn.functional as F

from components import Config, LayerNorm

cfg = Config()
layer_norm = LayerNorm(cfg)

x = t.randn(2, 5, cfg.d_model)

your_output = layer_norm(x)

expected = F.layer_norm(
    x,
    normalized_shape=(cfg.d_model,),
    weight=layer_norm.w,
    bias=layer_norm.b,
    eps=cfg.layer_norm_eps,
)

print(t.allclose(your_output, expected, atol=1e-5))
