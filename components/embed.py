import torch as t
from jaxtyping import Float, Int
from torch import Tensor, nn

from .config import Config


class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(
        self,
        tokens: Int[Tensor, "batch position"],  # type: ignore
    ) -> Float[Tensor, "batch position d_model"]:  # type: ignore
        return self.W_E[tokens]
