"""Correctness tests for the fused Triton RMSNorm kernel.

Checked against llm_core.model.RMSNorm (same formula: fp32 upcast, rsqrt of the
mean square plus eps, scale by weight, cast back). Output and both gradients
(w.r.t. the input and the weight) must match. The first call per n_cols also
triggers autotune; a correct result confirms the autotuned configs are sound.

CUDA + Triton only; skipped elsewhere.
"""
import pytest

triton = pytest.importorskip("triton")
import torch

if not torch.cuda.is_available():
    pytest.skip("Triton RMSNorm kernel requires CUDA", allow_module_level=True)

from llm_core.model import RMSNorm
from llm_systems.kernels.triton_rms_norm import TritonRMSNorm


def tolerance(dtype):
    return {torch.float32: 2e-5, torch.float16: 2e-2, torch.bfloat16: 3e-2}[dtype]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows,d", [(16, 512), (8, 1000), (32, 4096), (4, 768), (64, 2048)])
def test_forward_and_grads_match_reference(dtype, rows, d):
    torch.manual_seed(0)
    x = torch.randn(rows, d, device="cuda", dtype=dtype)
    weight = torch.randn(d, device="cuda")  # shared weight for both modules
    grad_out = torch.randn(rows, d, device="cuda", dtype=dtype)
    atol = tolerance(dtype)

    triton_norm = TritonRMSNorm(d).cuda()
    ref_norm = RMSNorm(d).cuda()
    with torch.no_grad():
        triton_norm.weight.copy_(weight)
        ref_norm.weight.copy_(weight)

    x_triton = x.clone().requires_grad_(True)
    y_triton = triton_norm(x_triton)
    y_triton.backward(grad_out)

    x_ref = x.clone().requires_grad_(True)
    y_ref = ref_norm(x_ref)
    y_ref.backward(grad_out)

    assert torch.allclose(y_triton.float(), y_ref.float(), atol=atol, rtol=1e-3), \
        (y_triton.float() - y_ref.float()).abs().max().item()
    assert torch.allclose(x_triton.grad.float(), x_ref.grad.float(), atol=atol, rtol=1e-3), \
        ("dx", (x_triton.grad.float() - x_ref.grad.float()).abs().max().item())
    assert torch.allclose(triton_norm.weight.grad.float(), ref_norm.weight.grad.float(), atol=1e-2, rtol=1e-3), \
        ("dw", (triton_norm.weight.grad.float() - ref_norm.weight.grad.float()).abs().max().item())


def test_unit_weight_normalizes():
    """With weight=1, each row should have unit RMS after normalization."""
    torch.manual_seed(0)
    x = torch.randn(32, 1024, device="cuda")
    out = TritonRMSNorm(1024).cuda()(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones(32, device="cuda"), atol=1e-3)
