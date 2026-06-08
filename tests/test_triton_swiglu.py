"""Correctness tests for the fused Triton SwiGLU (silu(gate) * up) kernel.

The autotuned kernels live at the silu_mul level, so they are checked there
against the eager reference silu(gate) * up — output plus both gradients
(w.r.t. gate and up). The backward overwrites gate/up in place, and the first
call per n_cols triggers autotune (many in-place trial runs), so a correct
gradient here also validates restore_value=["gate_ptr","up_ptr"].

CUDA + Triton only; skipped elsewhere.
"""
import pytest

triton = pytest.importorskip("triton")
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("Triton SwiGLU kernel requires CUDA", allow_module_level=True)

from llm_systems.kernels.triton_swiglu import FusedSwiGLU, triton_silu_mul


def reference_silu_mul(gate, up):
    return F.silu(gate) * up


def tolerance(dtype):
    return {torch.float32: 2e-5, torch.float16: 2e-2, torch.bfloat16: 3e-2}[dtype]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows,d", [(16, 1344), (8, 1000), (32, 4096), (4, 2048), (64, 1344)])
def test_silu_mul_forward_and_grads(dtype, rows, d):
    torch.manual_seed(0)
    gate = torch.randn(rows, d, device="cuda", dtype=dtype)
    up = torch.randn(rows, d, device="cuda", dtype=dtype)
    grad_out = torch.randn(rows, d, device="cuda", dtype=dtype)
    atol = tolerance(dtype)

    g_t = gate.clone().requires_grad_(True)
    u_t = up.clone().requires_grad_(True)
    out_t = triton_silu_mul(g_t, u_t)
    out_t.backward(grad_out)

    g_r = gate.clone().requires_grad_(True)
    u_r = up.clone().requires_grad_(True)
    out_r = reference_silu_mul(g_r, u_r)
    out_r.backward(grad_out)

    assert torch.allclose(out_t.float(), out_r.float(), atol=atol, rtol=1e-3), \
        (out_t.float() - out_r.float()).abs().max().item()
    assert torch.allclose(g_t.grad.float(), g_r.grad.float(), atol=atol, rtol=1e-3), \
        ("dgate", (g_t.grad.float() - g_r.grad.float()).abs().max().item())
    assert torch.allclose(u_t.grad.float(), u_r.grad.float(), atol=atol, rtol=1e-3), \
        ("dup", (u_t.grad.float() - u_r.grad.float()).abs().max().item())


def test_fused_swiglu_matches_unfused_module():
    """The full FusedSwiGLU should equal an unfused gate/up/down made of the same
    weights (its merged w_gate_up is [gate; up] stacked)."""
    torch.manual_seed(0)
    d_model, d_ff = 256, 688
    module = FusedSwiGLU(d_model, d_ff).cuda()
    x = torch.randn(4, 16, d_model, device="cuda")

    gate_w, up_w = module.w_gate_up.weight.chunk(2, dim=0)  # each (d_ff, d_model)
    expected = module.w_down(F.silu(x @ gate_w.T) * (x @ up_w.T))
    assert torch.allclose(module(x), expected, atol=1e-4)
