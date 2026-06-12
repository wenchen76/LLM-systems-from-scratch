"""Triton FlashAttention-2 forward, variable-length (prefill).

Computes causal self-attention for a flat batch of ragged sequences in a single
launch — the fused replacement for the per-sequence Python loop. Each sequence's
tokens occupy a contiguous span [start, start + length) of the flat token axis;
one program handles (query tile, head, sequence) and streams keys/values with
online softmax (Dao, 2023).

Forward only (inference): no logsumexp is saved and there is no backward, since
the serving path runs under no_grad.

Layout: Q, K, V, O are (heads, total_tokens, d_head), contiguous, with RoPE
already applied. K is read transposed via strides. d_head must be a power of two
and >= 16 (a tl.dot constraint).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _prefill_configs():
    # Tune the query/key tile shapes and warps/stages; keyed on head dim D, which
    # is what mainly drives the best tiling. No restore_value: O is written fresh
    # and Q/K/V are read-only, so autotune's trial reruns are side-effect free.
    return [
        triton.Config({"Q_TILE": qt, "K_TILE": kt}, num_warps=w, num_stages=s)
        for qt in (16, 32, 64, 128)
        for kt in (32, 64, 128)
        for w in (4, 8)
        for s in (2, 3)
    ]


@triton.autotune(configs=_prefill_configs(), key=["D"])
@triton.jit
def _flash_prefill_varlen_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    starts_ptr, lengths_ptr,
    stride_h, stride_t, stride_d,
    num_seqs,
    scale,
    D: tl.constexpr,
    Q_TILE: tl.constexpr,
    K_TILE: tl.constexpr,
):
    query_tile = tl.program_id(0)
    hs = tl.program_id(1)
    seq = hs % num_seqs
    head = hs // num_seqs

    start = tl.load(starts_ptr + seq)
    q_len = tl.load(lengths_ptr + seq)
    if query_tile * Q_TILE >= q_len:
        return  # this query tile lies past the end of the sequence

    base = head * stride_h + start * stride_t

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + base, shape=(q_len, D), strides=(stride_t, stride_d),
        offsets=(query_tile * Q_TILE, 0), block_shape=(Q_TILE, D), order=(1, 0),
    )
    Kt_block_ptr = tl.make_block_ptr(
        K_ptr + base, shape=(D, q_len), strides=(stride_d, stride_t),
        offsets=(0, 0), block_shape=(D, K_TILE), order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + base, shape=(q_len, D), strides=(stride_t, stride_d),
        offsets=(0, 0), block_shape=(K_TILE, D), order=(1, 0),
    )

    q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q_tile = (q_tile * scale).to(q_tile.dtype)

    m_i = tl.full((Q_TILE,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE,), dtype=tl.float32)
    o_i = tl.zeros((Q_TILE, D), dtype=tl.float32)

    q_pos = query_tile * Q_TILE + tl.arange(0, Q_TILE)
    max_key = (query_tile + 1) * Q_TILE  # causal: no query attends past its own position

    for k_tile in range(tl.cdiv(max_key, K_TILE)):
        kt = tl.load(Kt_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        s = tl.dot(q_tile, kt)  # (Q_TILE, K_TILE), fp32 accumulate
        k_pos = k_tile * K_TILE + tl.arange(0, K_TILE)
        s = tl.where(q_pos[:, None] < k_pos[None, :], -1e6, s)  # mask future keys

        m_new = tl.maximum(m_i, tl.max(s, axis=-1))
        p = tl.math.exp(s - m_new[:, None])
        alpha = tl.math.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=-1)
        o_i = o_i * alpha[:, None]
        o_i = tl.dot(p.to(v.dtype), v, acc=o_i)
        m_i = m_new

        Kt_block_ptr = Kt_block_ptr.advance((0, K_TILE))
        V_block_ptr = V_block_ptr.advance((K_TILE, 0))

    o_i = o_i / l_i[:, None]

    O_block_ptr = tl.make_block_ptr(
        O_ptr + base, shape=(q_len, D), strides=(stride_t, stride_d),
        offsets=(query_tile * Q_TILE, 0), block_shape=(Q_TILE, D), order=(1, 0),
    )
    tl.store(O_block_ptr, o_i.to(O_ptr.dtype.element_ty), boundary_check=(0, 1))


def _decode_configs():
    return [
        triton.Config({"K_TILE": kt}, num_warps=w, num_stages=s)
        for kt in (32, 64, 128, 256)
        for w in (2, 4, 8)
        for s in (2, 3)
    ]


@triton.autotune(configs=_decode_configs(), key=["D"])
@triton.jit
def _flash_decode_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    q_pos_ptr, k_starts_ptr, k_lens_ptr,
    stride_qh, stride_qt, stride_qd,
    stride_kh, stride_kt, stride_kd,
    num_decode,
    scale,
    D: tl.constexpr,
    K_TILE: tl.constexpr,
):
    hs = tl.program_id(0)
    seq = hs % num_decode
    head = hs // num_decode

    q_pos = tl.load(q_pos_ptr + seq)
    k_start = tl.load(k_starts_ptr + seq)
    k_len = tl.load(k_lens_ptr + seq)

    d_range = tl.arange(0, D)
    q = tl.load(Q_ptr + head * stride_qh + q_pos * stride_qt + d_range * stride_qd).to(tl.float32) * scale

    K_block_ptr = tl.make_block_ptr(
        K_ptr + head * stride_kh + k_start * stride_kt, shape=(k_len, D),
        strides=(stride_kt, stride_kd), offsets=(0, 0), block_shape=(K_TILE, D), order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + head * stride_kh + k_start * stride_kt, shape=(k_len, D),
        strides=(stride_kt, stride_kd), offsets=(0, 0), block_shape=(K_TILE, D), order=(1, 0),
    )

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((D,), dtype=tl.float32)

    # The decode query sits at the newest position, so it sees every cached key —
    # no causal mask, just the boundary mask for the ragged tail.
    for k_tile in range(tl.cdiv(k_len, K_TILE)):
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        offs = k_tile * K_TILE + tl.arange(0, K_TILE)
        s = tl.sum(q[None, :] * k, axis=1)  # (K_TILE,) — matrix-vector, q_len == 1
        s = tl.where(offs < k_len, s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        p = tl.math.exp(s - m_new)
        alpha = tl.math.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

        K_block_ptr = K_block_ptr.advance((K_TILE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE, 0))

    acc = acc / l_i
    tl.store(O_ptr + head * stride_qh + q_pos * stride_qt + d_range * stride_qd,
             acc.to(O_ptr.dtype.element_ty))


def flash_decode(
    Q: torch.Tensor,
    K_flat: torch.Tensor,
    V_flat: torch.Tensor,
    out: torch.Tensor,
    q_positions: torch.Tensor,
    k_starts: torch.Tensor,
    k_lens: torch.Tensor,
) -> None:
    """Single-query decode attention (no split-K) for a batch of sequences.

    Q, out: (heads, total_tokens, d_head). K_flat, V_flat: (heads, total_kv,
    d_head) — the running sequences' caches concatenated along the key axis.
    q_positions: (num_seqs,) flat-token index of each decode query. k_starts /
    k_lens: (num_seqs,) each sequence's span in K_flat. One program per
    (head, seq) attends over that sequence's full cache; out is written at
    q_positions.
    """
    heads, _, d = Q.shape
    num_decode = q_positions.numel()
    grid = (heads * num_decode,)
    _flash_decode_kernel[grid](
        Q, K_flat, V_flat, out,
        q_positions, k_starts, k_lens,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K_flat.stride(0), K_flat.stride(1), K_flat.stride(2),
        num_decode,
        1.0 / (d ** 0.5),
        D=d,
    )


def _paged_decode_configs():
    return [triton.Config({}, num_warps=w, num_stages=s) for w in (2, 4, 8) for s in (2, 3)]


@triton.autotune(configs=_paged_decode_configs(), key=["D"])
@triton.jit
def _paged_decode_kernel(
    Q_ptr, K_pool_ptr, V_pool_ptr, O_ptr,
    q_pos_ptr, block_tables_ptr, seq_lens_ptr,
    stride_qh, stride_qt, stride_qd,
    stride_blk, stride_ph, stride_bs, stride_pd,
    stride_bt_seq, stride_bt_blk,
    num_decode,
    scale,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    hs = tl.program_id(0)
    seq = hs % num_decode
    head = hs // num_decode

    q_pos = tl.load(q_pos_ptr + seq)
    seq_len = tl.load(seq_lens_ptr + seq)

    d_range = tl.arange(0, D)
    q = tl.load(Q_ptr + head * stride_qh + q_pos * stride_qt + d_range * stride_qd).to(tl.float32) * scale

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((D,), dtype=tl.float32)
    slot = tl.arange(0, BLOCK_SIZE)

    # Walk this sequence's blocks via its block table; a freed-and-reused block
    # may hold stale tokens past seq_len, so mask by absolute key position.
    for b in range(tl.cdiv(seq_len, BLOCK_SIZE)):
        block_id = tl.load(block_tables_ptr + seq * stride_bt_seq + b * stride_bt_blk)
        base = block_id * stride_blk + head * stride_ph
        ptrs = base + slot[:, None] * stride_bs + d_range[None, :] * stride_pd
        k = tl.load(K_pool_ptr + ptrs).to(tl.float32)  # (BLOCK_SIZE, D)
        v = tl.load(V_pool_ptr + ptrs).to(tl.float32)

        key_pos = b * BLOCK_SIZE + slot
        s = tl.sum(q[None, :] * k, axis=1)  # (BLOCK_SIZE,)
        s = tl.where(key_pos < seq_len, s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        p = tl.math.exp(s - m_new)
        alpha = tl.math.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    acc = acc / l_i
    tl.store(O_ptr + head * stride_qh + q_pos * stride_qt + d_range * stride_qd, acc.to(O_ptr.dtype.element_ty))


def paged_decode(
    Q: torch.Tensor,
    K_pool: torch.Tensor,
    V_pool: torch.Tensor,
    out: torch.Tensor,
    q_positions: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
) -> None:
    """Decode attention reading the KV directly from a paged pool.

    Q, out: (heads, total_tokens, d_head). K_pool, V_pool: (num_blocks, heads,
    block_size, d_head). q_positions: (num_seqs,) flat-token index per query.
    block_tables: (num_seqs, max_blocks) physical block ids; seq_lens: (num_seqs,)
    cache length. One program per (head, seq) walks the sequence's blocks.
    """
    heads, _, d = Q.shape
    num_decode = q_positions.numel()
    grid = (heads * num_decode,)
    _paged_decode_kernel[grid](
        Q, K_pool, V_pool, out,
        q_positions, block_tables, seq_lens,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K_pool.stride(0), K_pool.stride(1), K_pool.stride(2), K_pool.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        num_decode,
        1.0 / (d ** 0.5),
        D=d, BLOCK_SIZE=block_size,
    )


def flash_prefill_varlen(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Causal varlen self-attention written into `out` for the given sequences.

    Q, K, V, out: (heads, total_tokens, d_head), contiguous. starts/lengths:
    (num_seqs,) int — each sequence's span on the token axis. Only the spanned
    token slots of `out` are written.
    """
    heads, _, d = Q.shape
    num_seqs = starts.numel()
    max_len = int(lengths.max().item())
    # Q_TILE / K_TILE come from autotune, so the grid reads Q_TILE from the meta.
    grid = lambda meta: (triton.cdiv(max_len, meta["Q_TILE"]), heads * num_seqs)
    _flash_prefill_varlen_kernel[grid](
        Q, K, V, out,
        starts, lengths,
        Q.stride(0), Q.stride(1), Q.stride(2),
        num_seqs,
        1.0 / (d ** 0.5),
        D=d,
    )
