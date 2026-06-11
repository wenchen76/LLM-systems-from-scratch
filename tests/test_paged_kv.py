"""Tests for the paged KV block allocator."""
import pytest
import torch

from llm_core.paged_kv import BlockPool, PagedKVCache


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


def test_paged_cache_free_returns_blocks():
    pool = make_pool(num_blocks=8, block_size=4)
    cache = PagedKVCache(pool)
    cache.append(torch.randn(2, 9, 16), torch.randn(2, 9, 16))  # 9 tokens -> 3 blocks
    assert pool.num_free == 5
    cache.free()
    assert pool.num_free == 8
    assert cache.length == 0 and cache.block_table == []
