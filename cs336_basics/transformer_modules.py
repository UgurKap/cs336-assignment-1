import torch
from torch.nn import Module, init, Parameter, Sequential
from einops import einsum, reduce, rearrange, repeat
from math import sqrt, log
from jaxtyping import Float, Int, Bool
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


class SiLUFF(Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.silu = SiLU()
        self.W1 = Linear(in_features=d_model, out_features=d_ff)
        self.W2 = Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.W2(self.silu(self.W1(x)))


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
        self.register_buffer("flip_ones", tensor=torch.tensor([-1, 1], device=device), persistent=False)

    def forward(
        self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"]
    ) -> Float[Tensor, "... seq_len d_k"]:
        el1 = repeat(self.cos_matrix[token_positions, ...], "... seq_len d -> ... seq_len (d 2)") * x
        el2 = repeat(self.sin_matrix[token_positions, ...], "... seq_len d -> ... seq_len (d 2)") * rearrange(
            rearrange(x, "... seq_len (d_k f) -> ... seq_len d_k f", f=2).flip(dims=[-1]) * self.flip_ones,
            "... seq_len d_k f -> ... seq_len (d_k f)",
        )
        return el1 + el2


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    exp_x = torch.exp(x - torch.max(x, dim).values.unsqueeze(dim))
    return exp_x / torch.sum(exp_x, dim, keepdim=True)


def scaled_dot_product_attention(
    key: Float[Tensor, "batch_size ... seq_len d_k"],
    value: Float[Tensor, "batch_size ... seq_len d_v"],
    query: Float[Tensor, "batch_size ... seq_len d_k"],
    mask: Bool[Tensor, "seq_len seq_len"] | None = None,
) -> Float[Tensor, "batch_size ... seq_len d_v"]:
    dot_product = einsum(query, key, "... s1 d_k, ... s2 d_k -> ... s1 s2") / sqrt(key.shape[-1])
    if mask is not None:
        dot_product[..., ~mask] = -torch.inf
    attention_weights = softmax(dot_product, -1)
    return einsum(attention_weights, value, "b ... s1 s2, b ... s2 d_v -> b ... s1 d_v")


class MultiHeadSelfAttention(Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 1024, device: torch.device | None = None):
        super().__init__()
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device)), persistent=False
        )
        d_k = d_v = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = Linear(d_model, num_heads * d_k, device=device)
        self.W_k = Linear(d_model, num_heads * d_k, device=device)
        self.W_v = Linear(d_model, num_heads * d_v, device=device)
        self.output_projection = Linear(num_heads * d_v, d_model, device=device)

    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        seq_len = x.shape[-2]
        k = rearrange(self.W_k(x), "b s (h d) -> b h s d", h=self.num_heads)
        q = rearrange(self.W_q(x), "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(self.W_v(x), "b s (h d) -> b h s d", h=self.num_heads)
        out = rearrange(
            scaled_dot_product_attention(key=k, query=q, value=v, mask=self.mask[:seq_len, :seq_len]),
            "batch head seq_len d_v -> batch seq_len (head d_v)",
        )
        return self.output_projection(out)


class MultiHeadSelfAttentionRoPE(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 1024,
        theta: float = 10_000,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device)), persistent=False
        )
        d_k = d_v = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = Linear(d_model, num_heads * d_k, device=device)
        self.W_k = Linear(d_model, num_heads * d_k, device=device)
        self.W_v = Linear(d_model, num_heads * d_v, device=device)
        self.rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=device)
        self.output_projection = Linear(num_heads * d_v, d_model, device=device)

    def forward(
        self, x: Float[Tensor, "... seq_len d_model"], pos_vector: Int[Tensor, "... seq_len"] | None = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        seq_len = x.shape[-2]
        if pos_vector is None:
            pos_vector = torch.arange(0, seq_len, device=x.device)
        k = self.rope(rearrange(self.W_k(x), "b s (h d) -> b h s d", h=self.num_heads), pos_vector)
        q = self.rope(rearrange(self.W_q(x), "b s (h d) -> b h s d", h=self.num_heads), pos_vector)
        v = rearrange(self.W_v(x), "b s (h d) -> b h s d", h=self.num_heads)
        out = rearrange(
            scaled_dot_product_attention(key=k, query=q, value=v, mask=self.mask[:seq_len, :seq_len]),
            "batch head seq_len d_v -> batch seq_len (head d_v)",
        )
        return self.output_projection(out)


class TransformerBlock(Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 1024, theta: float = 10_000):
        super().__init__()
        self.rms1 = RMSNorm(d_model)
        self.rms2 = RMSNorm(d_model)
        self.mha = MultiHeadSelfAttentionRoPE(d_model, num_heads, max_seq_len, theta)
        self.ff = SwiGLU(d_model, d_ff)

    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        out1 = x + self.mha(self.rms1(x))
        out2 = out1 + self.ff(self.rms2(out1))
        return out2


class TransformerBlockSiLU(Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 1024, theta: float = 10_000):
        super().__init__()
        self.rms1 = RMSNorm(d_model)
        self.rms2 = RMSNorm(d_model)
        self.mha = MultiHeadSelfAttentionRoPE(d_model, num_heads, max_seq_len, theta)
        self.ff = SiLUFF(d_model, d_ff)

    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        out1 = x + self.mha(self.rms1(x))
        out2 = out1 + self.ff(self.rms2(out1))
        return out2


class TransformerBlockNoNorm(Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 1024, theta: float = 10_000):
        super().__init__()
        self.mha = MultiHeadSelfAttentionRoPE(d_model, num_heads, max_seq_len, theta)
        self.ff = SwiGLU(d_model, d_ff)

    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        out1 = x + self.mha(x)
        out2 = out1 + self.ff(out1)
        return out2


class TransformerBlockPostNorm(Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 1024, theta: float = 10_000):
        super().__init__()
        self.rms1 = RMSNorm(d_model)
        self.rms2 = RMSNorm(d_model)
        self.mha = MultiHeadSelfAttentionRoPE(d_model, num_heads, max_seq_len, theta)
        self.ff = SwiGLU(d_model, d_ff)

    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        out1 = self.rms1(x + self.mha(x))
        out2 = self.rms2(out1 + self.ff(out1))
        return out2


class TransformerBlockNoPE(Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 1024):
        super().__init__()
        self.rms1 = RMSNorm(d_model)
        self.rms2 = RMSNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model, num_heads, max_seq_len)
        self.ff = SwiGLU(d_model, d_ff)

    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        out1 = x + self.mha(self.rms1(x))
        out2 = out1 + self.ff(self.rms2(out1))
        return out2


class TransformerLM(Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: int,
    ):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer_blocks = Sequential(
            *[
                TransformerBlock(d_model, num_heads, d_ff, max_seq_len=context_length, theta=rope_theta)
                for _ in range(num_layers)
            ]
        )
        self.rms = RMSNorm(d_model)
        self.out_proj = Linear(d_model, vocab_size)

    def forward(self, x: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len vocab_size"]:
        return self.out_proj(self.rms(self.transformer_blocks(self.token_embeddings(x))))


class TransformerLMNoNorm(Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: int,
    ):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer_blocks = Sequential(
            *[
                TransformerBlockNoNorm(d_model, num_heads, d_ff, max_seq_len=context_length, theta=rope_theta)
                for _ in range(num_layers)
            ]
        )
        self.out_proj = Linear(d_model, vocab_size)

    def forward(self, x: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len vocab_size"]:
        return self.out_proj(self.transformer_blocks(self.token_embeddings(x)))


class TransformerLMPostNorm(Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: int,
    ):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer_blocks = Sequential(
            *[
                TransformerBlockPostNorm(d_model, num_heads, d_ff, max_seq_len=context_length, theta=rope_theta)
                for _ in range(num_layers)
            ]
        )
        self.out_proj = Linear(d_model, vocab_size)

    def forward(self, x: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len vocab_size"]:
        return self.out_proj(self.transformer_blocks(self.token_embeddings(x)))


class TransformerLMNoPE(Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: int,
    ):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer_blocks = Sequential(
            *[
                TransformerBlockNoPE(d_model, num_heads, d_ff, max_seq_len=context_length, theta=rope_theta)
                for _ in range(num_layers)
            ]
        )
        self.rms = RMSNorm(d_model)
        self.out_proj = Linear(d_model, vocab_size)

    def forward(self, x: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len vocab_size"]:
        return self.out_proj(self.rms(self.transformer_blocks(self.token_embeddings(x))))


class TransformerLMSiLU(Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: int,
    ):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer_blocks = Sequential(
            *[
                TransformerBlockSiLU(d_model, num_heads, d_ff, max_seq_len=context_length, theta=rope_theta)
                for _ in range(num_layers)
            ]
        )
        self.rms = RMSNorm(d_model)
        self.out_proj = Linear(d_model, vocab_size)

    def forward(self, x: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len vocab_size"]:
        return self.out_proj(self.rms(self.transformer_blocks(self.token_embeddings(x))))
