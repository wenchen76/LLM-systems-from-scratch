"""Correctness tests for the fused Triton AdamW kernel.

The kernel is checked against a pure-PyTorch implementation of the *same* update
(matching llm_core.optimizer.AdamW: bias-corrected step size alpha_t + decoupled
weight decay applied after the Adam step). Matching that reference confirms the
fused kernel computes the intended math.

Note: the very first step() also exercises autotune, which re-runs the kernel
many times to benchmark configs. Because the kernel updates param/m/v in place,
a correct result here also validates that restore_value=[...] is set properly —
without it those trial runs would corrupt the moments.

CUDA + Triton only; skipped elsewhere.
"""
import math

import pytest

triton = pytest.importorskip("triton")
import torch

if not torch.cuda.is_available():
    pytest.skip("Triton AdamW kernel requires CUDA", allow_module_level=True)

from llm_systems.kernels.triton_adamw import FusedAdamW


class ReferenceAdamW:
    """Pure-PyTorch reference matching the kernel's update, step for step."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr, (self.b1, self.b2), self.eps, self.wd = lr, betas, eps, weight_decay
        self.m = [torch.zeros_like(p, dtype=torch.float32) for p in self.params]
        self.v = [torch.zeros_like(p, dtype=torch.float32) for p in self.params]
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        alpha_t = self.lr * math.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t)
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad.float()
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            p32 = p.float()
            p32 = p32 - alpha_t * m / (v.sqrt() + self.eps)
            p32 = p32 - self.lr * self.wd * p32  # decoupled, on the updated param
            p.copy_(p32)  # round back to param dtype, mirroring the kernel's store


def tolerance(dtype):
    return {torch.float32: 2e-5, torch.float16: 2e-2, torch.bfloat16: 3e-2}[dtype]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(4096,), (1000,), (33, 50), (7, 13, 5)])
def test_matches_reference_trajectory(dtype, shape):
    """Run several steps with fresh random grads each step; params and moments
    must track the reference within tolerance for the dtype."""
    torch.manual_seed(0)
    base = torch.randn(shape, device="cuda", dtype=dtype)
    p_fused = base.clone().requires_grad_(True)
    p_ref = base.clone().requires_grad_(True)

    fused = FusedAdamW([p_fused], lr=1e-2, weight_decay=0.1)
    ref = ReferenceAdamW([p_ref], lr=1e-2, weight_decay=0.1)

    atol = tolerance(dtype)
    gen = torch.Generator(device="cuda").manual_seed(1)
    for _ in range(5):
        grad = torch.randn(shape, device="cuda", dtype=dtype, generator=gen)
        p_fused.grad = grad.clone()
        p_ref.grad = grad.clone()
        fused.step()
        ref.step()
        assert torch.allclose(p_fused.float(), p_ref.float(), atol=atol, rtol=1e-3), \
            (p_fused.float() - p_ref.float()).abs().max().item()

    # Moments (always fp32) should match tightly.
    assert torch.allclose(fused.state[p_fused]["m"], ref.m[0], atol=1e-5)
    assert torch.allclose(fused.state[p_fused]["v"], ref.v[0], atol=1e-5)
    assert fused.state[p_fused]["t"] == ref.t == 5


def test_zero_weight_decay_matches_reference():
    torch.manual_seed(0)
    base = torch.randn(2048, device="cuda")
    p_fused = base.clone().requires_grad_(True)
    p_ref = base.clone().requires_grad_(True)
    fused = FusedAdamW([p_fused], lr=3e-4, weight_decay=0.0)
    ref = ReferenceAdamW([p_ref], lr=3e-4, weight_decay=0.0)

    for _ in range(10):
        g = torch.randn_like(base)
        p_fused.grad = g.clone()
        p_ref.grad = g.clone()
        fused.step()
        ref.step()
    assert torch.allclose(p_fused, p_ref, atol=2e-5)


def test_minimizes_quadratic():
    """End-to-end sanity: AdamW should drive a simple quadratic toward its minimum."""
    torch.manual_seed(0)
    target = torch.randn(512, device="cuda")
    x = torch.zeros(512, device="cuda", requires_grad=True)
    opt = FusedAdamW([x], lr=1e-1)

    start = (x.detach() - target).pow(2).mean().item()
    for _ in range(200):
        opt.zero_grad()
        loss = (x - target).pow(2).mean()
        loss.backward()
        opt.step()
    end = (x.detach() - target).pow(2).mean().item()
    assert end < start * 1e-2  # converged by ~2 orders of magnitude


def test_first_step_correct_under_autotune():
    """The first step triggers autotune (many in-place trial runs). If
    restore_value were missing/wrong, this single step would already diverge."""
    torch.manual_seed(0)
    base = torch.randn(4096, device="cuda")
    p_fused = base.clone().requires_grad_(True)
    p_ref = base.clone().requires_grad_(True)
    g = torch.randn_like(base)
    p_fused.grad = g.clone()
    p_ref.grad = g.clone()

    FusedAdamW([p_fused], lr=1e-2, weight_decay=0.1).step()
    ReferenceAdamW([p_ref], lr=1e-2, weight_decay=0.1).step()
    assert torch.allclose(p_fused, p_ref, atol=2e-5)
