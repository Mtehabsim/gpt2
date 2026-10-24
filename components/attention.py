import torch as t
from einops import einsum
from jaxtyping import Float
from torch import Tensor, nn

from .config import Config


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]  # type: ignore

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32))

    def prepare_matrix(
        self,
        normalized_resid_pre: Float[Tensor, "batch posn d_model"],  # type: ignore
        learned,
        bias,
    ):
        return (
            einsum(
                normalized_resid_pre,
                learned,
                "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head",
            )
            + bias
        )

    def forward(
        self,
        normalized_resid_pre: Float[Tensor, "batch posn d_model"],  # type: ignore
    ) -> Float[Tensor, "batch posn d_model"]:  # type: ignore
        key = self.prepare_matrix(normalized_resid_pre, self.W_K, self.b_K)
        query = self.prepare_matrix(normalized_resid_pre, self.W_Q, self.b_Q)
        value = self.prepare_matrix(normalized_resid_pre, self.W_V, self.b_V)

        # now each of them is [batch posn n_heads d_head]

        attention_score_matrix = einsum(
            query,
            key,
            "batch q_posn n_heads d_head, batch k_posn n_heads d_head -> batch n_heads q_posn k_posn",
        )
        scaled_scores = attention_score_matrix / (self.cfg.d_head**0.5)
        masked_scores = self.apply_causal_mask(scaled_scores)
        attention = t.softmax(masked_scores, dim=-1)
        heads_output = einsum(
            value,  # for every batch, every token, every head, a k dimention vector
            attention,  # for every batch, every head, every query token, a keys tokens
            # the output we want is for every batch, every query token, every head, a d_model matrix
            "batch key_pos n_heads d_head, batch n_heads query_pos key_pos -> batch query_pos n_heads d_head",
        )
        output = (
            einsum(
                heads_output,
                self.W_O,
                "batch pos n_heads d_head, n_heads d_head d_model-> batch pos d_model",
            )
            + self.b_O
        )
        return output

    def apply_causal_mask(
        self,
        attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"],  # type: ignore
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:  # type: ignore
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        size = attn_scores.shape[2]
        causal_mask = t.triu(
            t.ones(size, size, device=attn_scores.device, dtype=t.bool), diagonal=1
        )
        return attn_scores.masked_fill(causal_mask, self.IGNORE)
