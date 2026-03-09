"""Fused SwiGLU FFN using Triton kernels.

Phase 1: Fuse SiLU + element-wise multiply into a single kernel.
Phase 2: Merge gate (w1) and up (w3) projections into one matmul.
Phase 3A: Epilogue fusion — SiLU+mul in the matmul epilogue (future).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from llm_core.model import Linear


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _silu_mul_fwd_kernel(
    gate_ptr, up_ptr, out_ptr,
    stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)

    gate_ptr += row_idx * stride
    up_ptr += row_idx * stride
    out_ptr += row_idx * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires float32 for numerical stability
    gate_row = tl.load(gate_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    up_row = tl.load(up_ptr + col_offsets, mask=mask, other=0)
    out_row = _silu(gate_row).cast(up_row.dtype) * up_row
    tl.store(out_ptr + col_offsets, out_row, mask=mask)


@triton.jit
def _silu_mul_bwd_kernel(
    dout_ptr, gate_ptr, up_ptr,
    stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward for out = silu(gate) * up, where silu(x) = x * sigmoid(x).

    d_up   = d_out * silu(gate)
    d_gate = d_out * up * [silu(gate) * (1 - sigmoid(gate)) + sigmoid(gate)]
           = d_out * up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
    """
    row_idx = tl.program_id(0).to(tl.int64)

    dout_ptr += row_idx * stride
    gate_ptr += row_idx * stride
    up_ptr += row_idx * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dout_row = tl.load(dout_ptr + col_offsets, mask=mask, other=0)
    # sigmoid requires float32
    gate_row = tl.load(gate_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    up_row = tl.load(up_ptr + col_offsets, mask=mask, other=0)

    # recomputation to save memory
    sig_gate = tl.sigmoid(gate_row)
    silu_gate = gate_row * sig_gate
    dup_row = dout_row * silu_gate
    dgate_row = dout_row * (silu_gate * (1 - sig_gate) + sig_gate) * up_row

    # in-place: overwrite gate, up buffers with gradients
    tl.store(gate_ptr + col_offsets, dgate_row, mask=mask)
    tl.store(up_ptr + col_offsets, dup_row, mask=mask)


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length()


class TritonSiLUMulFunction(torch.autograd.Function):
    """Fused silu(gate) * up — forward and backward via Triton."""

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        # gate = w1(x), up = w3(x) — matmul results already computed
        assert gate.shape == up.shape
        assert gate.is_contiguous() and up.is_contiguous()

        orig_shape = gate.shape
        # flatten to 2-D: (n_rows, n_cols)
        gate_2d = gate.reshape(-1, gate.shape[-1])
        up_2d = up.reshape(-1, up.shape[-1])
        n_rows, n_cols = gate_2d.shape
        out_2d = torch.empty_like(up_2d)

        BLOCK_SIZE = _next_power_of_2(n_cols)

        _silu_mul_fwd_kernel[(n_rows,)](
            gate_2d, up_2d, out_2d,
            stride=gate_2d.stride(0),
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Save gate, up for backward (recomputation of sigmoid inside kernel)
        ctx.save_for_backward(gate, up)
        return out_2d.reshape(orig_shape)

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        gate, up = ctx.saved_tensors

        # Clone because the backward kernel writes in-place
        dgate = gate.clone().contiguous()
        dup = up.clone().contiguous()
        dout = dout.contiguous()

        orig_shape = dgate.shape
        dgate_2d = dgate.reshape(-1, dgate.shape[-1])
        dup_2d = dup.reshape(-1, dup.shape[-1])
        dout_2d = dout.reshape(-1, dout.shape[-1])
        n_rows, n_cols = dgate_2d.shape

        BLOCK_SIZE = _next_power_of_2(n_cols)

        _silu_mul_bwd_kernel[(n_rows,)](
            dout_2d, dgate_2d, dup_2d,
            stride=dgate_2d.stride(0),
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return dgate_2d.reshape(orig_shape), dup_2d.reshape(orig_shape)


triton_silu_mul = TritonSiLUMulFunction.apply


# ---------------------------------------------------------------------------
# Module: drop-in replacement for SwiGLU
# ---------------------------------------------------------------------------

class FusedSwiGLU(nn.Module):
    """Fused SwiGLU FFN with merged gate+up projection.

    Phase 1: SiLU + element-wise mul fused into one Triton kernel.
    Phase 2: w1 (gate) and w3 (up) merged into a single (d_model, 2*d_ff) matmul.

    Interface is identical to the original SwiGLU(d_model, d_ff).
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # Phase 2: merged gate+up projection  — Linear(d_model, 2*d_ff)
        # First d_ff outputs = gate (w1), last d_ff outputs = up (w3)
        self.w_gate_up = Linear(d_model, 2 * d_ff)

        # Down projection
        self.w_down = Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_model)
        Returns:
            (batch, seq, d_model)
        """
        # Phase 2: single matmul for gate + up projections
        gate_up = self.w_gate_up(x)                        # (..., 2*d_ff)
        gate, up = gate_up.chunk(2, dim=-1)                # each (..., d_ff), free split

        # Phase 1: fused silu(gate) * up via Triton kernel
        hidden = triton_silu_mul(gate.contiguous(), up.contiguous())  # (..., d_ff)

        # Down projection
        return self.w_down(hidden)                         # (..., d_model)
