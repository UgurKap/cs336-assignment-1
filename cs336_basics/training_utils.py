import torch
from torch.optim import Optimizer
from math import sqrt, cos, pi, sin
from jaxtyping import Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias
import numpy as np
from numpy.typing import NDArray

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


def lr_cosine_schedule(step: int, max_lr: float, min_lr: float, warmup_steps: int, cosine_cycle_steps: int) -> float:
    if step < warmup_steps:
        return (step / warmup_steps) * max_lr
    elif step <= cosine_cycle_steps:
        return (
            min_lr
            + ((1 + cos(pi * (step - warmup_steps) / (cosine_cycle_steps - warmup_steps))) * (max_lr - min_lr)) / 2
        )
    else:
        return min_lr


def lr_cosine_schedule_sine_warmup(
    step: int, max_lr: float, min_lr: float, warmup_steps: int, cosine_cycle_steps: int
) -> float:
    if step < warmup_steps:
        return sin(pi / 2 * (step / warmup_steps)) * max_lr
    elif step <= cosine_cycle_steps:
        return (
            min_lr
            + ((1 + cos(pi * (step - warmup_steps) / (cosine_cycle_steps - warmup_steps))) * (max_lr - min_lr)) / 2
        )
    else:
        return min_lr


def gradient_clipping(params: ParamsT, max_norm: float, eps: float = 1e-6):
    norms = torch.tensor([torch.norm(p.grad) for p in params if p.grad is not None])
    total_norm = torch.norm(norms)
    if total_norm > max_norm:
        for p in params:
            if p.grad is not None:
                p.grad *= max_norm / (total_norm + eps)


def get_batch(
    x: Int[NDArray, " "], batch_size: int, context_length: int, device: str, seed: None | int = None
) -> tuple[Int[Tensor, " batch context"], Int[Tensor, " batch context"]]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(low=0, high=x.shape[-1] - context_length, size=batch_size)
    batch = torch.tensor(
        np.stack([x[start_ind : start_ind + context_length + 1] for start_ind in indices]),
        dtype=torch.int16,
        device=device,
    )
    return batch[..., 0:context_length], batch[..., 1 : context_length + 1]
