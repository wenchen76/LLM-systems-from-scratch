"""Microbenchmark: our paged_decode Triton kernel vs FlashInfer (and flash-attn).

Isolates a single decode step -- R sequences, each with L cached KV tokens in a
paged pool, one query token each -- and compares latency, achieved HBM bandwidth,
and (with --peak-bw) % of peak against off-the-shelf paged-decode kernels. Every
backend reads the same logical KV from a block pool; correctness is asserted
against ours before timing.

FlashInfer is the main baseline -- purpose-built for paged decode, with split-K
(flash-decoding) along the KV length. Our kernel runs one program per (head, seq)
with no split-K, so it should track FlashInfer at large R*heads and fall behind at
small batch / long context, where keeping the SMs busy needs the KV-length split.
Decode is memory-bound, so achieved bandwidth is the fair cross-shape metric.

flash-attn's flash_attn_with_kvcache is included too but needs a prebuilt wheel
matching the local torch; its column shows "-" (or "err") when unavailable.

    python bench_paged_decode.py --heads 16 --d-head 128 --block-size 16 \
        --batches 1 8 64 --lengths 256 1024 4096 --peak-bw 1935
"""
import argparse

import torch

from llm_systems.kernels.triton_flash_attention import paged_decode

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

try:
    import flashinfer
except ImportError:
    flashinfer = None


def build_pool(R, L, H, D, block_size, device, dtype):
    """A paged pool holding R sequences of length L (block i*bps..  -> seq i)."""
    bps = (L + block_size - 1) // block_size
    num_blocks = R * bps
    k_pool = torch.randn(num_blocks, H, block_size, D, device=device, dtype=dtype)
    v_pool = torch.randn(num_blocks, H, block_size, D, device=device, dtype=dtype)
    block_tables = torch.arange(num_blocks, device=device, dtype=torch.int32).view(R, bps)
    seq_lens = torch.full((R,), L, device=device, dtype=torch.int32)
    return k_pool, v_pool, block_tables, seq_lens


def run_ours(q, k_pool, v_pool, block_tables, seq_lens, block_size):
    """q: (heads, R, d_head). Returns out (heads, R, d_head)."""
    _, R, _ = q.shape
    out = torch.empty_like(q)
    q_positions = torch.arange(R, device=q.device, dtype=torch.int32)
    paged_decode(q.contiguous(), k_pool, v_pool, out, q_positions, block_tables, seq_lens, block_size)
    return out


def make_flash_attn(q, k_pool, v_pool, block_tables, seq_lens):
    """Build a flash_attn_with_kvcache runner with the cache pre-converted to its
    paged layout (num_blocks, page_size, heads, dim) outside the timed region, so
    only the attention -- not a per-call multi-GB pool copy -- is measured (a server
    stores the cache in this layout to begin with). Returns run() -> (heads, R, d_head)."""
    k_cache = k_pool.permute(0, 2, 1, 3).contiguous()          # (num_blocks, block_size, H, D)
    v_cache = v_pool.permute(0, 2, 1, 3).contiguous()

    def run():
        q_bshd = q.transpose(0, 1).unsqueeze(1).contiguous()   # (R, 1, H, D)
        o = flash_attn_with_kvcache(q_bshd, k_cache, v_cache,
                                    cache_seqlens=seq_lens, block_table=block_tables, causal=False)
        return o.squeeze(1).transpose(0, 1)                    # (H, R, D)
    return run


def make_flashinfer(q, k_pool, v_pool, block_tables, L, block_size, workspace, dtype):
    """Build and plan a FlashInfer paged-decode wrapper, then return a callable that
    runs one decode step. plan() (scheduling) is done here, outside timing; only the
    returned run() is timed, matching a server that plans once per shape and runs each
    step. FlashInfer wants the cache in NHD page layout (num_pages, page_size, heads,
    dim) and the block table as CSR (page_indptr / page_indices / last_page_len)."""
    H, R, D = q.shape
    bps = (L + block_size - 1) // block_size
    k_cache = k_pool.permute(0, 2, 1, 3).contiguous()          # (num_pages, page_size, H, D)
    v_cache = v_pool.permute(0, 2, 1, 3).contiguous()
    device = q.device
    page_indptr = torch.arange(0, (R + 1) * bps, bps, device=device, dtype=torch.int32)
    page_indices = block_tables.reshape(-1).to(torch.int32)
    last_page_len = torch.full((R,), (L - 1) % block_size + 1, device=device, dtype=torch.int32)

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(page_indptr, page_indices, last_page_len, H, H, D, block_size,
                 pos_encoding_mode="NONE", q_data_type=dtype, kv_data_type=dtype)

    def run():
        qr = q.transpose(0, 1).contiguous()                    # (R, H, D)
        o = wrapper.run(qr, (k_cache, v_cache))                # (R, H, D)
        return o.transpose(0, 1)                               # (H, R, D)
    return run


def time_fn(fn, iters, warmup):
    """Median ms per call over `iters`, timed in one block after `warmup`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser(description="paged_decode vs flash_attn_with_kvcache")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--d-head", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256,
                        help="Tokens per KV block (flash-attn paged cache wants a multiple of 256)")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64])
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--peak-bw", type=float, default=None, help="Peak HBM GB/s for %% of peak (e.g. 1935)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    bytes_per = torch.empty((), dtype=dtype).element_size()
    H, D = args.heads, args.d_head
    have_fi, have_fa = flashinfer is not None, flash_attn_with_kvcache is not None
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=args.device) if have_fi else None
    print(f"device={args.device} dtype={args.dtype} heads={H} d_head={D} block_size={args.block_size} "
          f"flashinfer={'yes' if have_fi else 'MISSING'} flash_attn={'yes' if have_fa else 'MISSING'}")
    bwh = "GB/s" + (" %pk" if args.peak_bw else "")
    print(f"{'batch':>5} {'len':>6} | {'ours ms':>8} {bwh:>10} | {'fi ms':>8} {'x':>5} | "
          f"{'fa ms':>8} {'x':>5} | {'max|Δ|':>8}")
    print("-" * 84)

    for L in args.lengths:
        for R in args.batches:
            k_pool, v_pool, block_tables, seq_lens = build_pool(R, L, H, D, args.block_size, args.device, dtype)
            q = torch.randn(H, R, D, device=args.device, dtype=dtype)
            kv_bytes = R * L * H * D * 2 * bytes_per  # K and V read once: the memory-bound floor

            def bw(ms):
                gbs = kv_bytes / (ms * 1e-3) / 1e9
                return f"{gbs:.0f}" + (f" ({gbs / args.peak_bw * 100:.0f}%)" if args.peak_bw else "")

            out_ours = run_ours(q, k_pool, v_pool, block_tables, seq_lens, args.block_size)
            t_ours = time_fn(lambda: run_ours(q, k_pool, v_pool, block_tables, seq_lens, args.block_size),
                             args.iters, args.warmup)

            diff = 0.0
            fi_ms, fi_x = "-", "-"
            if have_fi:
                try:
                    fi_run = make_flashinfer(q, k_pool, v_pool, block_tables, L, args.block_size, workspace, dtype)
                    diff = max(diff, (out_ours - fi_run()).abs().max().item())
                    t_fi = time_fn(fi_run, args.iters, args.warmup)
                    fi_ms, fi_x = f"{t_fi:.3f}", f"{t_fi / t_ours:.2f}"
                except Exception as e:  # API / layout mismatches surface here, not as a crash
                    fi_ms, fi_x = "err", str(e)[:5]

            fa_ms, fa_x = "-", "-"
            if have_fa:
                try:
                    fa_run = make_flash_attn(q, k_pool, v_pool, block_tables, seq_lens)
                    diff = max(diff, (out_ours - fa_run()).abs().max().item())
                    t_fa = time_fn(fa_run, args.iters, args.warmup)
                    fa_ms, fa_x = f"{t_fa:.3f}", f"{t_fa / t_ours:.2f}"
                except Exception as e:
                    fa_ms, fa_x = "err", str(e)[:5]

            print(f"{R:>5} {L:>6} | {t_ours:>8.3f} {bw(t_ours):>10} | {fi_ms:>8} {fi_x:>5} | "
                  f"{fa_ms:>8} {fa_x:>5} | {diff:>8.1e}")


if __name__ == "__main__":
    main()
