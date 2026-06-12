"""Tests for the paged KV block allocator."""
import pytest
import torch

from llm_core.paged_kv import BlockPool, PagedKVCache, append_decode


def make_pool(num_blocks=8, block_size=4, num_heads=2, d_head=16):
    return BlockPool(num_blocks, block_size, num_heads, d_head)


def test_pool_shapes_and_initial_free():
    pool = make_pool(num_blocks=8, block_size=4, num_heads=2, d_head=16)
    assert pool.k.shape == (8, 2, 4, 16)
    assert pool.v.shape == (8, 2, 4, 16)
    assert pool.num_free == 8


def test_allocate_hands_out_distinct_blocks():
    pool = make_pool(num_blocks=8)
    ids = [pool.allocate() for _ in range(8)]
    assert sorted(ids) == list(range(8))  # every block, exactly once
    assert pool.num_free == 0


def test_allocate_n_and_free_round_trip():
    pool = make_pool(num_blocks=8)
    block_ids = pool.allocate_n(5)
    assert len(block_ids) == 5
    assert pool.num_free == 3
    pool.free(block_ids)
    assert pool.num_free == 8
    # freed blocks can be handed out again
    assert pool.allocate() in block_ids


def test_exhaustion_raises():
    pool = make_pool(num_blocks=3)
    pool.allocate_n(3)
    with pytest.raises(RuntimeError):
        pool.allocate()
    with pytest.raises(RuntimeError):
        make_pool(num_blocks=3).allocate_n(4)


@pytest.mark.parametrize("n_tokens,expected", [(0, 0), (1, 1), (4, 1), (5, 2), (8, 2), (9, 3)])
def test_blocks_needed(n_tokens, expected):
    pool = make_pool(block_size=4)
    assert pool.blocks_needed(n_tokens) == expected


def test_paged_cache_matches_contiguous_reference():
    """append + materialize over a mix of prefill (chunk) and decode (1-token)
    steps, including block-boundary crossings, must equal a plain concatenation."""
    torch.manual_seed(0)
    heads, d, block_size = 2, 16, 4
    pool = BlockPool(num_blocks=16, block_size=block_size, num_heads=heads, d_head=d)
    cache = PagedKVCache(pool)

    ref_k, ref_v = [], []
    for n_new in [5, 1, 1, 1, 3, 1]:  # prefill 5, then decodes, crossing 4-token blocks
        k = torch.randn(heads, n_new, d)
        v = torch.randn(heads, n_new, d)
        cache.append(k, v)
        ref_k.append(k)
        ref_v.append(v)

    length = 12
    assert cache.length == length
    assert len(cache.block_table) == pool.blocks_needed(length) == 3  # cdiv(12, 4)

    mk, mv = cache.materialize()
    assert mk.shape == (heads, length, d)
    assert torch.allclose(mk, torch.cat(ref_k, dim=1))
    assert torch.allclose(mv, torch.cat(ref_v, dim=1))


def test_append_decode_batches_match_per_cache_append():
    """The batched decode append (one scatter for many caches) must leave every
    cache identical to calling append() on each one — across a block boundary, with
    caches at different lengths so they cross blocks on different steps."""
    torch.manual_seed(0)
    heads, d, block_size, R = 2, 16, 4, 5
    batched_pool = BlockPool(num_blocks=32, block_size=block_size, num_heads=heads, d_head=d)
    ref_pool = BlockPool(num_blocks=32, block_size=block_size, num_heads=heads, d_head=d)
    batched = [PagedKVCache(batched_pool) for _ in range(R)]
    ref = [PagedKVCache(ref_pool) for _ in range(R)]

    # Seed each cache with a different prefill length so steps cross blocks staggered.
    for i in range(R):
        n = i + 2
        k, v = torch.randn(heads, n, d), torch.randn(heads, n, d)
        batched[i].append(k.clone(), v.clone())
        ref[i].append(k.clone(), v.clone())

    for _ in range(6):  # decode steps, enough to cross block boundaries
        k = torch.randn(heads, R, d)  # column i -> cache i
        v = torch.randn(heads, R, d)
        append_decode(batched, k, v)
        for i in range(R):
            ref[i].append(k[:, i : i + 1], v[:, i : i + 1])

    for b, r in zip(batched, ref):
        assert b.length == r.length
        assert b.block_table == r.block_table
        bk, bv = b.materialize()
        rk, rv = r.materialize()
        assert torch.allclose(bk, rk) and torch.allclose(bv, rv)


def test_paged_cache_free_returns_blocks():
    pool = make_pool(num_blocks=8, block_size=4)
    cache = PagedKVCache(pool)
    cache.append(torch.randn(2, 9, 16), torch.randn(2, 9, 16))  # 9 tokens -> 3 blocks
    assert pool.num_free == 5
    cache.free()
    assert pool.num_free == 8
    assert cache.length == 0 and cache.block_table == []
