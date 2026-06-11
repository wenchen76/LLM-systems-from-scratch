"""Tests for the paged KV block allocator."""
import pytest
import torch

from llm_core.paged_kv import BlockPool


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
