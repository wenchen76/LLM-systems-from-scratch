"""Paged KV cache: a block-based KV store (PagedAttention, Kwon et al. 2023).

KV memory is one pre-allocated pool of fixed-size blocks. Each request holds a
block table (logical position -> physical block), so its cache grows a block at
a time from a shared free list instead of needing a contiguous per-request
allocation — no fragmentation, no over-reservation, and the attention kernel can
read scattered blocks via the table instead of a concatenated buffer.
"""
from __future__ import annotations

import torch


class BlockPool:
    """A pool of fixed-size KV blocks with a free list.

    K and V are stored as (num_blocks, num_heads, block_size, d_head): each
    physical block holds block_size tokens for every head, laid out so one
    block-head slice is contiguous (what the attention kernel streams).
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        d_head: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.k = torch.zeros(num_blocks, num_heads, block_size, d_head, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self._free: list[int] = list(reversed(range(num_blocks)))  # pop() hands out 0, 1, 2, ...

    @property
    def num_free(self) -> int:
        return len(self._free)

    def blocks_needed(self, num_tokens: int) -> int:
        """How many blocks a sequence of num_tokens occupies."""
        return (num_tokens + self.block_size - 1) // self.block_size

    def allocate(self) -> int:
        """Take one free block; raises if the pool is exhausted."""
        if not self._free:
            raise RuntimeError("BlockPool exhausted: no free KV blocks")
        return self._free.pop()

    def allocate_n(self, n: int) -> list[int]:
        """Take n free blocks at once (e.g. for a prompt's prefill)."""
        if len(self._free) < n:
            raise RuntimeError(f"BlockPool exhausted: requested {n}, {len(self._free)} free")
        return [self._free.pop() for _ in range(n)]

    def free(self, block_ids: list[int]) -> None:
        """Return blocks to the pool (e.g. when a request finishes)."""
        self._free.extend(block_ids)
