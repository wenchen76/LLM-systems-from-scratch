"""Fused AdamW optimizer using a single Triton kernel per parameter.

Why fuse?
---------
The standard PyTorch AdamW performs 4 separate operations per parameter, each
launching its own CUDA kernel and making its own pass over global memory:

    m  = β₁·m  + (1-β₁)·g          # pass 1: read m,g  → write m     (3N)
    v  = β₂·v  + (1-β₂)·g²         # pass 2: read v,g  → write v     (3N)
    p -= αₜ · m / (√v + ε)          # pass 3: read p,m,v → write p    (4N)
    p -= lr · λ · p                  # pass 4: read p     → write p    (2N)
                                     # ─────────────────────────────────────
                                     # Total: 12N memory transactions

This module fuses ALL four operations into ONE Triton kernel that makes a
SINGLE pass over memory:

    read p, g, m, v  (4N)  ─┐
    all math in registers    │  ONE kernel, ONE pass
    write p, m, v    (3N)  ─┘  Total: 7N memory transactions

That's a ~1.7× reduction in memory traffic, which matters because optimizer
steps are memory-bandwidth-bound (pure element-wise ops, no data reuse).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel — the ENTIRE AdamW update in a single pass
# ---------------------------------------------------------------------------

@triton.jit
def _fused_adamw_kernel(
    # --- 4 pointers: the only global memory we touch ---
    param_ptr,       # [N] parameters (read + write)
    grad_ptr,        # [N] gradients  (read only)
    exp_avg_ptr,     # [N] 1st moment m (read + write), always fp32
    exp_avg_sq_ptr,  # [N] 2nd moment v (read + write), always fp32
    # --- scalar hyperparameters (passed as kernel args, NOT constexpr) ---
    # Not constexpr because lr changes every step with a schedule;
    # constexpr would trigger Triton recompilation each step.
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    # Bias-corrected step size, precomputed on host to avoid pow() in kernel:
    #   alpha_t = lr * sqrt(1 - β₂ᵗ) / (1 - β₁ᵗ)
    alpha_t,
    # --- size ---
    n_elements,
    # --- tile size (must be constexpr for tl.arange) ---
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance handles one BLOCK_SIZE tile of the flat parameter.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # =====================================================================
    # SINGLE READ PASS — load all 4 tensors from global memory into registers
    # =====================================================================
    p = tl.load(param_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
    m = tl.load(exp_avg_ptr + offsets, mask=mask)      # already fp32
    v = tl.load(exp_avg_sq_ptr + offsets, mask=mask)    # already fp32

    # =====================================================================
    # ALL MATH IN REGISTERS — zero additional global memory access
    # =====================================================================
    # (1) Update biased first moment:   m ← β₁·m + (1-β₁)·g
    m = beta1 * m + (1.0 - beta1) * g

    # (2) Update biased second moment:  v ← β₂·v + (1-β₂)·g²
    v = beta2 * v + (1.0 - beta2) * g * g

    # (3) Bias-corrected Adam update:   p ← p - αₜ · m/(√v + ε)
    #     where αₜ = lr · √(1-β₂ᵗ) / (1-β₁ᵗ)  (precomputed on host)
    p = p - alpha_t * m / (tl.sqrt(v) + eps)

    # (4) Decoupled weight decay:       p ← p - lr·λ·p
    #     Applied AFTER the Adam step, on the already-updated p,
    #     exactly matching the reference optimizer.py line 80.
    p = p - lr * weight_decay * p

    # =====================================================================
    # SINGLE WRITE PASS — store all 3 updated tensors back to global memory
    # (grad is read-only, so only 3 stores vs 4 loads)
    # =====================================================================
    # Cast param back to its original dtype (supports fp16/bf16 mixed precision)
    tl.store(param_ptr + offsets, p, mask=mask)
    tl.store(exp_avg_ptr + offsets, m, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, v, mask=mask)


# ---------------------------------------------------------------------------
# Optimizer class — drop-in replacement for cs336_basics.optimizer.AdamW
# ---------------------------------------------------------------------------

class FusedAdamW(torch.optim.Optimizer):
    """AdamW with a fused Triton kernel.

    Identical interface to cs336_basics.optimizer.AdamW.
    Each call to step() launches ONE kernel per parameter (not 4).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FusedAdamW does not support sparse gradients")

                state = self.state[p]

                # Lazy state initialization — m and v are always fp32
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    state["v"] = torch.zeros_like(p, dtype=torch.float32)

                state["t"] += 1
                t = state["t"]

                # Precompute bias-corrected step size on host.
                # This avoids pow/sqrt inside the GPU kernel and matches
                # optimizer.py line 77:
                #   alpha_t = lr * sqrt(1 - beta_2^t) / (1 - beta_1^t)
                alpha_t = lr * math.sqrt(1.0 - beta2 ** t) / (1.0 - beta1 ** t)

                # Flatten to 1-D for the kernel (contiguous view, no copy)
                param_flat = p.view(-1)
                grad_flat = grad.contiguous().view(-1)
                m_flat = state["m"].view(-1)
                v_flat = state["v"].view(-1)
                n_elements = param_flat.numel()

                # Launch ONE kernel that does the ENTIRE update
                BLOCK_SIZE = 1024
                grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

                _fused_adamw_kernel[grid](
                    param_flat, grad_flat, m_flat, v_flat,
                    lr, beta1, beta2, eps, weight_decay,
                    alpha_t,
                    n_elements,
                    BLOCK_SIZE=BLOCK_SIZE,
                )

        return loss
