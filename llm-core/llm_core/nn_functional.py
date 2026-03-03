from __future__ import annotations

from typing import Iterable

import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x = x - x_max
    return x - torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    neg_log_probs = -log_softmax(logits)
    return torch.mean(torch.gather(neg_log_probs, -1, targets.unsqueeze(-1)))


def clip_gradient(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = 0.0
    for g in grads:
        total_norm += (g**2).sum()
    total_norm = torch.sqrt(total_norm)
    clip_coef = min(1.0, max_norm / (total_norm + 1e-6))
    for g in grads:
        g *= clip_coef
