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

from llm_core.paged_kv import BlockPool, PagedKVCache
from llm_systems.kernels.triton_flash_attention import flash_decode, paged_decode


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_paged_decode_matches_eager(dtype):
    """Decode reading KV from a paged pool via block tables == eager per sequence."""
    torch.manual_seed(0)
    heads, d, block_size = 4, 64, 16
    kv_lens = [1, 50, 200, 17]  # exercise multi-block and ragged tails
    num_decode = len(kv_lens)

    pool = BlockPool(128, block_size, heads, d, dtype=dtype, device="cuda")
    caches, ref_k, ref_v = [], [], []
    for L in kv_lens:
        cache = PagedKVCache(pool)
        k = torch.randn(heads, L, d, device="cuda", dtype=dtype)
        v = torch.randn(heads, L, d, device="cuda", dtype=dtype)
        cache.append(k, v)
        caches.append(cache)
        ref_k.append(k)
        ref_v.append(v)

    Q = torch.randn(heads, num_decode, d, device="cuda", dtype=dtype)
    out = torch.empty_like(Q)
    max_blocks = max(len(c.block_table) for c in caches)
    block_tables = torch.zeros(num_decode, max_blocks, dtype=torch.int32, device="cuda")
    for i, c in enumerate(caches):
        block_tables[i, : len(c.block_table)] = torch.tensor(c.block_table, dtype=torch.int32, device="cuda")

    paged_decode(
        Q, pool.k, pool.v, out,
        torch.arange(num_decode, dtype=torch.int32, device="cuda"),
        block_tables,
        torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
        block_size,
    )

    atol = 1e-3 if dtype == torch.float32 else 2e-2
    for i, L in enumerate(kv_lens):
        ref = F.scaled_dot_product_attention(Q[:, i:i + 1], ref_k[i], ref_v[i])
        assert torch.allclose(out[:, i:i + 1].float(), ref.float(), atol=atol), \
            (i, L, (out[:, i:i + 1].float() - ref.float()).abs().max().item())
