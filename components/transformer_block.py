from jaxtyping import Float
from torch import Tensor, nn

from .attention import Attention
from .config import Config
from .layernorm import LayerNorm
from .mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self,
        resid_pre: Float[Tensor, "batch position d_model"],  # type: ignore
    ) -> Float[Tensor, "batch position d_model"]:  # type: ignore

        normalized_resid_pre = self.ln1.forward(resid_pre)

        attention_output = self.attn.forward(normalized_resid_pre)

        resid_mid = attention_output + resid_pre

        normalized_mid_block = self.ln2.forward(resid_mid)

        post_mlp = self.mlp.forward(normalized_mid_block)

        resid_post = post_mlp + resid_mid
        return resid_post
