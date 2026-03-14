"""Fused RMSNorm using Triton kernels for forward and backward passes.

Fuses the entire RMSNorm computation into a single kernel per direction,
avoiding multiple global-memory passes that the naive PyTorch implementation
would require (cast → square → mean → rsqrt → multiply → multiply → cast).

Reference kernels adapted from Liger-Kernel / Unsloth, simplified to:
- Always use learnable weight (elementwise_affine=True, no offset)
- Always compute in fp32 and cast back (GEMMA-style casting)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

TORCH_TO_TRITON_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    RSTD_ptr,
    W_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    y_i = (x_i / RMS) * w_i,  RMS = sqrt(sum(x_i^2) / N)

    Each program instance processes one row.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x_base = X_ptr + row_idx * X_row_stride
    y_base = Y_ptr + row_idx * Y_row_stride

    # Load row and weight
    X_row = tl.load(x_base + col_offsets, mask=mask, other=0)
    X_row_dtype = X_row.dtype
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    # Cast to fp32 for numerical stability
    X_row = X_row.to(tl.float32)
    W_row = W_row.to(tl.float32)

    # Compute rstd = 1 / RMS
    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(mean_square + eps)

    # Cache rstd for backward
    tl.store(RSTD_ptr + row_idx, rstd)

    # Normalize and apply weight
    Y_row = (X_row * rstd) * W_row

    # Cast back to original dtype
    tl.store(y_base + col_offsets, Y_row.to(X_row_dtype), mask=mask)


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------

@triton.jit
def _rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    RSTD_ptr,
    W_ptr,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    rows_per_program,
    BLOCK_SIZE: tl.constexpr,
):
    """
    dx = rstd * [dy * w - (1/N) * rstd^2 * (dy * w · x) * x]
    dw = sum(dy * x * rstd)  — summed over the batch dimension
    """
    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Accumulator for dW partial sum (this program's rows)
    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Load weight once
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)

    for row_idx in range(row_start, row_end):
        dY_row = tl.load(
            dY_ptr + row_idx * dY_row_stride + col_offsets, mask=mask, other=0.0
        )
        X_row = tl.load(
            X_ptr + row_idx * X_row_stride + col_offsets, mask=mask, other=0.0
        )
        rstd_row = tl.load(RSTD_ptr + row_idx)

        # Upcast to fp32
        X_row = X_row.to(tl.float32)
        dY_row = dY_row.to(tl.float32)

        # m = dy * w
        m = dY_row * W_row

        # dx = rstd * m - rstd * (1/N) * rstd^2 * dot(m, x) * x
        dX_row = rstd_row * m
        dX_row += rstd_row * (
            -(1.0 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0) * X_row
        )

        # dw += dy * (x * rstd)  — accumulate over rows
        dW_row += dY_row * (X_row * rstd_row)

        tl.store(
            dX_ptr + row_idx * dX_row_stride + col_offsets,
            dX_row.to(X_dtype),
            mask=mask,
        )

    # Store this program's partial dW
    tl.store(
        dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row, mask=mask
    )


_SM_COUNT_CACHE: dict[torch.device, int] = {}


def _get_sm_count(device: torch.device) -> int:
    if device not in _SM_COUNT_CACHE:
        _SM_COUNT_CACHE[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SM_COUNT_CACHE[device]


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------

class TritonRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float):
        # Flatten leading dims: (..., D) -> (M, D)
        orig_shape = x.shape
        x = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x.shape

        # Allocate outputs
        y = torch.empty_like(x)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        _rms_norm_forward_kernel[(n_rows,)](
            y, y.stride(0),
            x, x.stride(0),
            rstd,
            weight,
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(x, weight, rstd)
        ctx.n_rows = n_rows
        ctx.n_cols = n_cols
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.orig_shape = orig_shape

        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight, rstd = ctx.saved_tensors
        n_rows = ctx.n_rows
        n_cols = ctx.n_cols
        BLOCK_SIZE = ctx.BLOCK_SIZE

        dy = dy.contiguous().view(n_rows, n_cols)

        # Allocate outputs
        dx = torch.empty_like(x)

        # Number of programs for backward — use SM count or n_rows, whichever is smaller
        sm_count = _get_sm_count(x.device)
        n_programs = min(sm_count, n_rows)
        rows_per_program = (n_rows + n_programs - 1) // n_programs

        # Each program writes a partial dW; we sum them at the end
        dW_partial = torch.empty(
            n_programs, n_cols, dtype=torch.float32, device=x.device
        )

        tl_dtype = TORCH_TO_TRITON_DTYPE[x.dtype]

        _rms_norm_backward_kernel[(n_programs,)](
            dy, dy.stride(0),
            dx, dx.stride(0),
            x, x.stride(0),
            tl_dtype,
            rstd,
            weight,
            dW_partial, dW_partial.stride(0),
            n_rows,
            n_cols,
            rows_per_program,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Sum partial dW across programs
        dw = dW_partial.sum(dim=0).to(weight.dtype)

        return dx.view(ctx.orig_shape), dw, None


# ---------------------------------------------------------------------------
# nn.Module — drop-in replacement for RMSNorm
# ---------------------------------------------------------------------------

class TritonRMSNorm(nn.Module):
    """
    Triton-fused RMSNorm. Same interface as cs336_basics.model.RMSNorm.

    Args:
        hidden_size: Dimensionality of the input to normalize.
        eps: A value added to the denominator for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TritonRMSNormFunction.apply(x, self.weight, self.eps)

    def extra_repr(self):
        return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"
