import torch as t
from jaxtyping import Float, Int
from torch import Tensor, nn

from .config import Config


class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(
        self,
        tokens: Int[Tensor, "batch position"],  # type: ignore
    ) -> Float[Tensor, "batch position d_model"]:  # type: ignore
        positions = (
            t.arange(tokens.shape[1], device=tokens.device)
            .unsqueeze(0)
            .expand(tokens.shape[0], -1)
        )

        return self.W_pos[positions]
