import torch as t
import torch.nn.functional as F
from einops import einsum
from jaxtyping import Float
from torch import Tensor, nn

from .config import Config


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self,
        normalized_resid_mid: Float[Tensor, "batch posn d_model"],  # type: ignore
    ) -> Float[Tensor, "batch posn d_model"]:  # type: ignore
        scale_up = (
            einsum(
                normalized_resid_mid,  # residual stream, every batch has tokens(positions) and each one has a d_model vector
                self.W_in,  # weights which are for every d_model dimension it has a d_mlp weights
                # as output we want for every batch, every position with a d_mlp
                "batch posn d_model, d_model d_mlp -> batch posn d_mlp",
            )
            + self.b_in
        )
        non_linear_step = F.gelu(scale_up, approximate="tanh")
        scale_down = (
            einsum(
                non_linear_step,  # every batch has tokens(positions) and each one has a d_mlp vector
                self.W_out,  # weights which are for every d_model dimension it has a d_mlp weights
                # as output we want for every batch, every position with a d_mlp
                "batch posn d_mlp, d_mlp d_model -> batch posn d_model",
            )
            + self.b_out
        )
        return scale_down
