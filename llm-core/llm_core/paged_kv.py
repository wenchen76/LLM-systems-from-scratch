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
        self._decode_scratch: dict[str, torch.Tensor] = {}  # reused per-step decode-metadata buffers
        self.max_seq_blocks: int | None = None  # fixed block-table width for graph capture (engine sets it)

    def decode_buffer(self, name: str, rows: int, cols: int | None = None,
                      dtype: torch.dtype = torch.int64) -> torch.Tensor:
        """A reusable buffer for one decode step's metadata (block ids, offsets,
        lengths, ...), grown on demand and sliced to the requested shape. Reusing one
        buffer instead of allocating a fresh tensor each step keeps the address stable
        between same-shape steps — the prerequisite for CUDA-graph capture / compile of
        the decode path. Callers overwrite the whole slice (copy_) before reading it."""
        buf = self._decode_scratch.get(name)
        fits = buf is not None and buf.shape[0] >= rows and (cols is None or (buf.dim() == 2 and buf.shape[1] >= cols))
        if not fits:
            cap_rows = rows if buf is None else max(rows, buf.shape[0])
            if cols is None:
                buf = torch.empty(cap_rows, dtype=dtype, device=self.k.device)
            else:
                cap_cols = cols if (buf is None or buf.dim() < 2) else max(cols, buf.shape[1])
                buf = torch.empty(cap_rows, cap_cols, dtype=dtype, device=self.k.device)
            self._decode_scratch[name] = buf
        return buf[:rows] if cols is None else buf[:rows, :cols]

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


class PagedKVCache:
    """One request's KV cache, stored as a block table into a shared BlockPool.

    Mirrors KVCache's role (length / append) but writes tokens into pool blocks,
    growing one block at a time. materialize() gathers the cached tokens back
    into a contiguous (heads, length, d_head) view for eager attention and tests;
    the paged attention kernel will instead read the pool via the block table.
    """

    is_paged = True

    def __init__(self, pool: BlockPool):
        self.pool = pool
        self.block_table: list[int] = []
        self.length = 0

    def _physical(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map absolute token positions to (physical block id, in-block offset)."""
        table = torch.tensor(self.block_table, device=self.pool.k.device)
        block_idx = positions // self.pool.block_size
        return table[block_idx], positions % self.pool.block_size

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append new tokens. k, v: (heads, n_new, d_head)."""
        n_new = k.size(1)
        end = self.length + n_new
        while len(self.block_table) < self.pool.blocks_needed(end):
            self.block_table.append(self.pool.allocate())

        positions = torch.arange(self.length, end, device=self.pool.k.device)
        phys, offsets = self._physical(positions)
        # advanced index on (block, offset) -> (n_new, heads, d_head) on the LHS
        self.pool.k[phys, :, offsets, :] = k.permute(1, 0, 2).to(self.pool.k.dtype)
        self.pool.v[phys, :, offsets, :] = v.permute(1, 0, 2).to(self.pool.v.dtype)
        self.length = end

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather the cached K, V into contiguous (heads, length, d_head)."""
        positions = torch.arange(self.length, device=self.pool.k.device)
        phys, offsets = self._physical(positions)
        k = self.pool.k[phys, :, offsets, :].permute(1, 0, 2)  # (heads, length, d_head)
        v = self.pool.v[phys, :, offsets, :].permute(1, 0, 2)
        return k, v

    # KVCache-compatible views so eager attention can use a paged cache unchanged.
    @property
    def k(self) -> torch.Tensor:
        return self.materialize()[0]

    @property
    def v(self) -> torch.Tensor:
        return self.materialize()[1]

    def free(self) -> None:
        """Release this request's blocks back to the pool."""
        self.pool.free(self.block_table)
        self.block_table = []
        self.length = 0


def append_decode(caches: list[PagedKVCache], k_new: torch.Tensor, v_new: torch.Tensor) -> None:
    """Append exactly one new token to each of several caches sharing one pool, in
    a single batched scatter. k_new, v_new are (heads, R, d_head); column i belongs
    to caches[i].

    This is the decode hot path: it replaces R per-cache append() calls — each of
    which would rebuild a block-table tensor and copy it to the GPU — with two small
    index transfers (block id + offset) and one scatter, regardless of R.
    """
    pool = caches[0].pool
    bs = pool.block_size
    for cache in caches:  # grow a table by one block only when the new token spills over (rare)
        if cache.length // bs >= len(cache.block_table):
            cache.block_table.append(pool.allocate())
    n = len(caches)
    phys = pool.decode_buffer("phys", n)  # reused buffers -> stable address, no per-step alloc
    offset = pool.decode_buffer("offset", n)
    phys.copy_(torch.tensor([c.block_table[c.length // bs] for c in caches]))
    offset.copy_(torch.tensor([c.length % bs for c in caches]))
    pool.k[phys, :, offset, :] = k_new.permute(1, 0, 2).to(pool.k.dtype)  # (R, heads, d_head)
    pool.v[phys, :, offset, :] = v_new.permute(1, 0, 2).to(pool.v.dtype)
    for cache in caches:
        cache.length += 1
