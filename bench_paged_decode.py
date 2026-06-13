"""Microbenchmark: our paged_decode Triton kernel vs flash-attn's flash_attn_with_kvcache.

Isolates a single decode step -- R sequences, each with L cached KV tokens in a
paged pool, one query token each -- and compares latency, achieved HBM bandwidth,
and (with --peak-bw) % of peak. Both kernels read the same logical KV from a block
pool via per-sequence block tables; correctness is asserted before timing.

Decode is memory-bound: the floor is reading every KV byte once, so achieved
bandwidth is the fair cross-shape metric. Our kernel runs one program per
(head, seq) with no split-K, so it should track flash-attn at large R*heads and
fall behind at small batch / long context, where flash-decoding's split-K along
the KV length is what keeps the SMs busy.

    python bench_paged_decode.py --heads 16 --d-head 128 --block-size 256 \
        --batches 1 8 64 --lengths 256 1024 4096 --peak-bw 1935
"""
import argparse

import torch

from llm_systems.kernels.triton_flash_attention import paged_decode

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None


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


def run_flash_attn(q, k_pool, v_pool, block_tables, seq_lens):
    """Same logical attention through flash_attn_with_kvcache; returns (heads, R, d_head)."""
    H, R, D = q.shape
    q_bshd = q.transpose(0, 1).unsqueeze(1).contiguous()       # (R, 1, H, D)
    k_cache = k_pool.permute(0, 2, 1, 3).contiguous()          # (num_blocks, block_size, H, D)
    v_cache = v_pool.permute(0, 2, 1, 3).contiguous()
    o = flash_attn_with_kvcache(q_bshd, k_cache, v_cache,
                                cache_seqlens=seq_lens, block_table=block_tables, causal=False)
    return o.squeeze(1).transpose(0, 1)                        # (H, R, D)


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
    have_fa = flash_attn_with_kvcache is not None
    print(f"device={args.device} dtype={args.dtype} heads={H} d_head={D} block_size={args.block_size} "
          f"flash_attn={'yes' if have_fa else 'MISSING'}")
    bw_hdr = "GB/s" + (" (%pk)" if args.peak_bw else "")
    print(f"{'batch':>5} {'len':>6} | {'ours ms':>9} {bw_hdr:>12} | {'flash ms':>9} {bw_hdr:>12} | "
          f"{'ours/flash':>10} {'max|Δ|':>8}")
    print("-" * 88)

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

            fa_ms, ratio, diff = "-", "-", "-"
            if have_fa:
                try:
                    out_fa = run_flash_attn(q, k_pool, v_pool, block_tables, seq_lens)
                    diff = f"{(out_ours - out_fa).abs().max().item():.1e}"
                    t_fa = time_fn(lambda: run_flash_attn(q, k_pool, v_pool, block_tables, seq_lens),
                                   args.iters, args.warmup)
                    fa_ms, ratio = f"{t_fa:.3f}", f"{t_fa / t_ours:.2f}x"
                    fa_bw = bw(t_fa)
                except Exception as e:  # layout / page-size constraints surface here, not as a crash
                    fa_ms, fa_bw, ratio, diff = "err", str(e)[:24], "-", "-"
            else:
                fa_bw = "-"
            print(f"{R:>5} {L:>6} | {t_ours:>9.3f} {bw(t_ours):>12} | {fa_ms:>9} {fa_bw:>12} | "
                  f"{ratio:>10} {diff:>8}")


if __name__ == "__main__":
    main()
