import torch as t
from jaxtyping import Float
from torch import Tensor, nn

from .config import Config


class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(
        self,
        residual: Float[Tensor, "batch posn d_model"],  # type: ignore
    ) -> Float[Tensor, "batch posn d_model"]:  # type: ignore
        mean_tensor = residual.mean(dim=-1, keepdim=True)
        varience_tensor = residual.var(dim=-1, keepdim=True, correction=0)
        semi_final_tensor = (residual - mean_tensor) / t.sqrt(
            varience_tensor + self.cfg.layer_norm_eps
        )
        final = (semi_final_tensor * self.w) + self.b
        return final
