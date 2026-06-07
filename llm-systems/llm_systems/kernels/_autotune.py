"""Shared @triton.autotune config builders for the custom kernels.

Autotune benchmarks a kernel across these configs the first time it sees each
new `key` shape, caches the winning config, and reuses it thereafter. Two shapes
of tuning show up across our kernels:

- tile_configs: the tile size BLOCK_SIZE is a free knob (1-D elementwise kernels,
  or column-tiled kernels that loop over BLOCK_SIZE chunks).
- warp_configs: BLOCK_SIZE is pinned by the caller (e.g. next_pow2(n_cols), one
  row per program), so only num_warps / num_stages are tuned.

IMPORTANT — in-place kernels: autotune RE-RUNS the kernel many times to time it.
Any kernel that mutates its inputs in place (AdamW updates params/moments,
cross-entropy writes grads back into the logits, SwiGLU backward writes into
gate/up) MUST pass restore_value=[...] (the mutated arg names) to
@triton.autotune, or the benchmark trials corrupt state and results go wrong.
"""
from __future__ import annotations

import triton


def tile_configs(
    block_sizes: tuple[int, ...],
    warps: tuple[int, ...] = (4, 8),
    stages: tuple[int, ...] = (2, 3),
) -> list[triton.Config]:
    """Configs that tune the tile size BLOCK_SIZE together with warps/stages."""
    return [
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in block_sizes
        for w in warps
        for s in stages
    ]


def warp_configs(
    warps: tuple[int, ...] = (2, 4, 8, 16),
    stages: tuple[int, ...] = (2, 3),
) -> list[triton.Config]:
    """Configs that tune only num_warps / num_stages (tile size fixed by caller)."""
    return [triton.Config({}, num_warps=w, num_stages=s) for w in warps for s in stages]
