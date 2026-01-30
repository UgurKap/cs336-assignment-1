import torch
from torch.optim import Optimizer
from math import sqrt
from jaxtyping import Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

ParamsT: TypeAlias = (
    Iterable[Float[Tensor, "..."]] | Iterable[dict[str, Any]] | Iterable[tuple[str, Float[Tensor, "..."]]]
)


def cross_entropy_loss(
    predicted_logits: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, "1"]:
    logits = predicted_logits - predicted_logits.max(dim=-1, keepdim=True).values
    batch_index = torch.arange(logits.shape[0], device=logits.device)
    loss = torch.log(torch.sum(torch.exp(logits), -1)) - logits[batch_index, targets]
    return torch.mean(loss)


class SGD(Optimizer):
    def __init__(self, params: ParamsT, lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss


class AdamW(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0:
            raise ValueError("Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad.data
                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * grad * grad
                lr_t = lr * sqrt(1 - b2**t) / (1 - b1**t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss
