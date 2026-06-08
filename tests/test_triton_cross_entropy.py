"""Correctness tests for the fused Triton cross-entropy kernel.

Checked against llm_core.nn_functional.cross_entropy (mean over rows of the
negative log-prob at the target). Both the loss and the gradient w.r.t. the
logits must match.

Note: the forward kernel writes the gradient back into the logits in place, and
the first call for each vocab size triggers autotune (many in-place trial runs).
A correct loss/grad here therefore also validates that restore_value=["X_ptr"]
is set — otherwise the trials would corrupt the logits.

CUDA + Triton only; skipped elsewhere.
"""
import pytest

triton = pytest.importorskip("triton")
import torch

if not torch.cuda.is_available():
    pytest.skip("Triton cross-entropy kernel requires CUDA", allow_module_level=True)

from llm_core.nn_functional import cross_entropy as reference_cross_entropy
from llm_systems.kernels.triton_cross_entropy import triton_cross_entropy


def tolerance(dtype):
    return {torch.float32: (1e-4, 1e-5), torch.float16: (3e-2, 3e-3), torch.bfloat16: (5e-2, 5e-3)}[dtype]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bt,vocab", [(8, 1000), (4, 50257), (16, 32000), (2, 999), (32, 4096)])
def test_loss_and_grad_match_reference(dtype, bt, vocab):
    torch.manual_seed(0)
    logits = torch.randn(bt, vocab, device="cuda", dtype=dtype)
    targets = torch.randint(0, vocab, (bt,), device="cuda")
    loss_atol, grad_atol = tolerance(dtype)

    x_triton = logits.clone().requires_grad_(True)
    loss_triton = triton_cross_entropy(x_triton, targets)
    loss_triton.backward()

    x_ref = logits.clone().requires_grad_(True)
    loss_ref = reference_cross_entropy(x_ref, targets)
    loss_ref.backward()

    assert torch.allclose(loss_triton.float(), loss_ref.float(), atol=loss_atol, rtol=1e-3), \
        (loss_triton.item(), loss_ref.item())
    assert torch.allclose(x_triton.grad.float(), x_ref.grad.float(), atol=grad_atol, rtol=1e-3), \
        (x_triton.grad.float() - x_ref.grad.float()).abs().max().item()


def test_forward_only_matches_reference():
    """Loss without requires_grad (kernel skips the gradient pass)."""
    torch.manual_seed(0)
    logits = torch.randn(64, 50257, device="cuda")
    targets = torch.randint(0, 50257, (64,), device="cuda")

    loss_triton = triton_cross_entropy(logits.clone(), targets)
    loss_ref = reference_cross_entropy(logits.clone(), targets)
    assert torch.allclose(loss_triton, loss_ref, atol=1e-4)


def test_gradient_is_normalized_softmax():
    """Sanity: grad equals (softmax - onehot) / N and each row sums to ~0."""
    torch.manual_seed(0)
    bt, vocab = 16, 4096
    logits = torch.randn(bt, vocab, device="cuda")
    targets = torch.randint(0, vocab, (bt,), device="cuda")

    x = logits.clone().requires_grad_(True)
    triton_cross_entropy(x, targets).backward()

    expected = torch.softmax(logits, dim=-1)
    expected[torch.arange(bt, device="cuda"), targets] -= 1.0
    expected /= bt
    assert torch.allclose(x.grad, expected, atol=1e-5)
    assert torch.allclose(x.grad.sum(dim=-1), torch.zeros(bt, device="cuda"), atol=1e-5)
