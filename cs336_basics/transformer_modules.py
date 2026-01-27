import torch
from torch.nn import Module, init, Parameter
from einops import einsum, reduce, rearrange, repeat
from math import sqrt, log
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


class RMSNorm(Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.g = Parameter(torch.ones(size=(d_model,), device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Float[Tensor, "batch_size seq_len d_model"]) -> Float[Tensor, "batch_size seq_len d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = (x / torch.sqrt(reduce(x * x, "... d_model -> ... 1", "mean") + self.eps)) * self.g
        return result.to(in_dtype)


class SiLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... in_features"]:
        return x * torch.sigmoid(x)


class SwiGLU(Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.silu = SiLU()
        self.W1 = Linear(in_features=d_model, out_features=d_ff)
        self.W2 = Linear(in_features=d_ff, out_features=d_model)
        self.W3 = Linear(in_features=d_model, out_features=d_ff)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        op1 = self.silu(self.W1(x))
        op2 = self.W3(x)
        return self.W2(op1 * op2)


class RotaryPositionalEmbedding(Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        assert d_k % 2 == 0, "d_k should be divisible by 2"
        freqs = torch.exp(torch.tensor([-2 * k * log(theta) / (d_k) for k in range(d_k // 2)], device=device))
        positions = torch.arange(0, max_seq_len, device=device)
        self.register_buffer(
            "cos_matrix",
            tensor=torch.cos(rearrange(positions, "seq_len -> seq_len 1") * rearrange(freqs, "d -> 1 d")),
            persistent=False,
        )
        self.register_buffer(
            "sin_matrix",
            tensor=torch.sin(rearrange(positions, "seq_len -> seq_len 1") * rearrange(freqs, "d -> 1 d")),
            persistent=False,
        )

    def forward(
        self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"]
    ) -> Float[Tensor, "... seq_len d_k"]:
        el1 = repeat(self.cos_matrix[token_positions, ...], "... seq_len d -> ... seq_len (d 2)") * x
        el2 = repeat(self.sin_matrix[token_positions, ...], "... seq_len d -> ... seq_len (d 2)") * rearrange(
            rearrange(x, "... seq_len (d_k f) -> ... seq_len d_k f", f=2).flip(dims=[-1])
            * torch.tensor([-1, 1], device=x.device, dtype=x.dtype),
            "... seq_len d_k f -> ... seq_len (d_k f)",
        )
        return el1 + el2


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    x = x - torch.max(x, dim).values.unsqueeze(dim)
    x = torch.exp(x) / torch.sum(torch.exp(x), dim, keepdim=True)
    return x
