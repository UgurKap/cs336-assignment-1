import torch
from torch.nn import Module, init, Parameter
from einops import einsum
from math import sqrt
from jaxtyping import Float, Int
from torch import Tensor


class Linear(Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        std = sqrt(2 / (in_features + out_features))
        self.W = Parameter(
            init.trunc_normal_(
                torch.empty(size=(out_features, in_features), dtype=dtype, device=device),
                mean=0,
                std=std,
                a=(-3 * std),
                b=(3 * std),
            )
        )

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embeddings = Parameter(
            init.trunc_normal_(
                torch.empty(size=(num_embeddings, embedding_dim), dtype=dtype, device=device), mean=0, std=1, a=-3, b=3
            )
        )

    def forward(self, x: Int[Tensor, "batch_size seq_len"]) -> Float[Tensor, "batch_size seq_len embedding_dim"]:
        return self.embeddings[x]
