"""Fused cross-entropy loss using a Triton kernel.

Computes forward loss + in-place gradient in a single pass over the logits,
avoiding materialisation of the full softmax probability matrix.
Uses the online softmax algorithm (Milakov & Gimelshein, 2018).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


MAX_FUSED_SIZE = 65536 // 2


# ---------------------------------------------------------------------------
# Triton kernel – one program per row (sample)
# ---------------------------------------------------------------------------

@triton.jit
def _cross_entropy_fwd_kernel(
    X_ptr,          # [BT, V] logits — overwritten with gradients
    X_stride,       # stride between rows
    Y_ptr,          # [BT]    targets
    Y_stride,
    loss_ptr,       # [BT]    per-row loss
    loss_stride,
    n_cols,         # V (vocab size)
    n_non_ignore,   # number of non-ignored rows
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
    HAS_GRADIENTS: tl.constexpr,
):
    row_id = tl.program_id(0).to(tl.int64)

    # ---- load target ---------------------------------------------------
    Y_ptr += row_id * Y_stride
    y = tl.load(Y_ptr)

    X_ptr += row_id * X_stride

    # If this row is ignored, zero out logits (gradient) and return
    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            offs = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + offs, 0.0, mask=offs < n_cols)
        return

    loss_ptr += row_id * loss_stride

    # ---- first pass: online max + sum (logsumexp) ----------------------
    m = float("-inf")
    d = 0.0
    ori_X_y = tl.load(X_ptr + y).to(tl.float32)

    for i in range(0, n_cols, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + offs, mask=offs < n_cols, other=float("-inf")).to(tl.float32)
        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    # log(sum(e^x_i)) = log(e^m * sum(e^(x_i - m))) = m + log(d)
    lse = m + tl.log(d)

    # ---- second pass: write gradients into X ---------------------------
    # grad_i = softmax(x_i) / N           for i != y
    # grad_y = (softmax(x_y) - 1) / N
    if HAS_GRADIENTS:
        for i in range(0, n_cols, BLOCK_SIZE):
            offs = i + tl.arange(0, BLOCK_SIZE)
            X_block = tl.load(X_ptr + offs, mask=offs < n_cols, other=float("-inf")).to(tl.float32)
            grad = tl.exp(X_block - m) / d          # softmax
            grad = tl.where(offs != y, grad, grad - 1.0)
            grad = grad / n_non_ignore               # mean reduction
            tl.store(X_ptr + offs, grad, mask=offs < n_cols)

    tl.debug_barrier()

    # ---- loss ----------------------------------------------------------
    loss = (lse - ori_X_y) / n_non_ignore
    tl.store(loss_ptr, loss)


# ---------------------------------------------------------------------------
# Forward / backward Python wrappers
# ---------------------------------------------------------------------------

def _cross_entropy_forward(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100):
    BT, V = x.shape
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    loss_1d = torch.zeros(BT, dtype=x.dtype, device=x.device)

    n_non_ignore = (target != ignore_index).sum().item()
    if n_non_ignore == 0:
        # Nothing to compute — zero loss, zero grads
        x.zero_()
        return torch.tensor(0.0, dtype=x.dtype, device=x.device), x

    if x.stride(-1) != 1:
        x = x.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    _cross_entropy_fwd_kernel[(BT,)](
        X_ptr=x,
        X_stride=x.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-1),
        loss_ptr=loss_1d,
        loss_stride=loss_1d.stride(-1),
        n_cols=V,
        n_non_ignore=n_non_ignore,
        ignore_index=ignore_index,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_GRADIENTS=x.requires_grad,
        num_warps=32,
    )

    loss = loss_1d.sum()
    return loss, x


def _cross_entropy_backward(x: torch.Tensor, grad_output: torch.Tensor):
    # When cross entropy is the last layer, grad_output is 1.0 — skip the
    # elementwise multiply to save a full kernel launch + memory pass.
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return x
    # scalar grad_output (mean/sum reduction)
    BT, V = x.shape
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    _element_mul_kernel[(BT,)](
        x,
        x.stride(-2),
        grad_output,
        V,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )
    return x


@triton.jit
def _element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Multiply every element in a row by a scalar (grad_output)."""
    row_id = tl.program_id(0).to(tl.int64)
    X_ptr += row_id * X_stride
    g = tl.load(grad_output_ptr)
    for i in range(0, n_cols, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X_ptr + offs, mask=offs < n_cols)
        tl.store(X_ptr + offs, x * g, mask=offs < n_cols)


# ---------------------------------------------------------------------------
# Autograd function
# ---------------------------------------------------------------------------

class TritonCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, target: torch.Tensor):
        requires_grad = x.requires_grad
        loss, x = _cross_entropy_forward(x, target)
        if requires_grad:
            ctx.save_for_backward(x.detach())
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        x = _cross_entropy_backward(x, grad_output)
        return x, None


# ---------------------------------------------------------------------------
# Public API — drop-in replacement for cross_entropy(inputs, targets)
# ---------------------------------------------------------------------------

def triton_cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Triton-fused cross-entropy loss.

    Same interface as ``cs336_basics.nn_utils.cross_entropy``:
        inputs:  (B*T, V)  float logits
        targets: (B*T,)    int64 class indices
    Returns scalar mean loss.
    """
    return TritonCrossEntropyFunction.apply(inputs, targets)
