import torch as t
from einops import einsum
from jaxtyping import Float, Int
from torch import Tensor, nn

from .config import Config
from .embed import Embed
from .layernorm import LayerNorm
from .posembed import PosEmbed
from .transformer_block import TransformerBlock
from .unembed import Unembed


class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(
        self,
        tokens: Int[Tensor, "batch position"],  # type: ignore
    ) -> Float[Tensor, "batch position d_vocab"]:  # type: ignore

        embeddings = self.embed(tokens) + self.pos_embed(tokens)

        residual_stream = embeddings
        for block in self.blocks:
            residual_stream = block(residual_stream)

        normalized_stream = self.ln_final(residual_stream)

        logits = self.unembed(normalized_stream)
        return logits
