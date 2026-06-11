"""Correctness test for the Triton varlen prefill (FlashAttention-2) kernel.

The fused single-launch kernel must match eager causal attention run on each
sequence separately, including sequences longer than the tile (multi-tile causal)
and lengths that are not tile multiples.

CUDA + Triton only; skipped elsewhere.
"""
import pytest

triton = pytest.importorskip("triton")
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("Triton flash attention requires CUDA", allow_module_level=True)

from llm_systems.kernels.triton_flash_attention import flash_prefill_varlen


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flash_prefill_varlen_matches_eager(dtype):
    torch.manual_seed(0)
    heads, d = 4, 64
    lengths = [5, 100, 3, 64]  # includes multi-tile (100) and exact-tile (64)
    starts = [0]
    for L in lengths[:-1]:
        starts.append(starts[-1] + L)
    total = starts[-1] + lengths[-1]

    Q = torch.randn(heads, total, d, device="cuda", dtype=dtype)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    out = torch.empty_like(V)

    flash_prefill_varlen(
        Q, K, V,
        torch.tensor(starts, device="cuda", dtype=torch.int32),
        torch.tensor(lengths, device="cuda", dtype=torch.int32),
        out,
    )

    atol = 1e-4 if dtype == torch.float32 else 2e-2
    for s, L in zip(starts, lengths):
        ref = F.scaled_dot_product_attention(Q[:, s:s + L], K[:, s:s + L], V[:, s:s + L], is_causal=True)
        got = out[:, s:s + L]
        assert torch.allclose(got.float(), ref.float(), atol=atol), \
            (s, L, (got.float() - ref.float()).abs().max().item())
