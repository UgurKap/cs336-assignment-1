import torch
from torch.nn import Module, init, Parameter, Sequential
from einops import einsum, reduce, rearrange, repeat
from math import sqrt, log
from jaxtyping import Float, Int, Bool
from torch import Tensor


def cross_entropy_loss(
    predicted_logits: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]
) -> Float[Tensor, "1"]:
    logits = predicted_logits - predicted_logits.max(dim=-1, keepdim=True).values
    batch_index = torch.arange(logits.shape[0], device=logits.device)
    loss = torch.log(torch.sum(torch.exp(logits), -1)) - logits[batch_index, targets]
    return torch.mean(loss)
