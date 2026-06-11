"""Correctness test for the Triton decode attention kernel.

A single query per sequence attends over that sequence's full cache (read from a
concatenated K/V buffer via k_starts/k_lens). The fused kernel must match eager
attention run per sequence, across short and long (multi-tile) cache lengths.

CUDA + Triton only; skipped elsewhere.
"""
import pytest

triton = pytest.importorskip("triton")
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("Triton flash decode requires CUDA", allow_module_level=True)

from llm_systems.kernels.triton_flash_attention import flash_decode


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flash_decode_matches_eager(dtype):
    torch.manual_seed(0)
    heads, d = 4, 64
    kv_lens = [1, 50, 200, 17]  # single key, multi-tile (200), and non-tile lengths
    num_decode = len(kv_lens)
    total_kv = sum(kv_lens)

    Q = torch.randn(heads, num_decode, d, device="cuda", dtype=dtype)  # query per sequence
    K_flat = torch.randn(heads, total_kv, d, device="cuda", dtype=dtype)
    V_flat = torch.randn_like(K_flat)
    out = torch.empty_like(Q)

    starts = [0]
    for length in kv_lens[:-1]:
        starts.append(starts[-1] + length)

    flash_decode(
        Q, K_flat, V_flat, out,
        torch.arange(num_decode, device="cuda", dtype=torch.int32),  # q at position seq
        torch.tensor(starts, device="cuda", dtype=torch.int32),
        torch.tensor(kv_lens, device="cuda", dtype=torch.int32),
    )

    atol = 1e-3 if dtype == torch.float32 else 2e-2
    for i, (ks, L) in enumerate(zip(starts, kv_lens)):
        ref = F.scaled_dot_product_attention(Q[:, i:i + 1], K_flat[:, ks:ks + L], V_flat[:, ks:ks + L])
        got = out[:, i:i + 1]
        assert torch.allclose(got.float(), ref.float(), atol=atol), \
            (i, L, (got.float() - ref.float()).abs().max().item())
