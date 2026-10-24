import torch as t
from einops import einsum
from jaxtyping import Float
from torch import Tensor, nn

from .config import Config


class Unembed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab)), requires_grad=False)

    def forward(
        self,
        normalized_resid_final: Float[Tensor, "batch position d_model"],  # type: ignore
    ) -> Float[Tensor, "batch position d_vocab"]:  # type: ignore
        return (
            einsum(
                normalized_resid_final,  # fro every batch we have tokens, for each one we have a d_model vector
                self.W_U,  # for every d_model dimension we have a d_vocab vector
                # as a result we want for each batch, every position a vocab vector
                "batch position d_model, d_model d_vocab -> batch position d_vocab",
            )
            + self.b_U
        )
