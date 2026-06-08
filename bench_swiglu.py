"""Performance benchmark for the fused Triton SwiGLU (silu(gate) * up) kernel.

The fused kernel computes silu(gate) * up in one pass and recomputes sigmoid in
the backward (no saved activation), versus the eager path that materializes the
silu activation and the product. PyTorch has no single fused silu-gated-multiply
op, so the baseline is the eager reference. We report forward+backward time and
peak memory across d_ff sizes.

CUDA + Triton only.

    python bench_swiglu.py
    python bench_swiglu.py --rows 32768 --dtype bfloat16
"""
import argparse

import torch
import torch.nn.functional as F
import triton

from llm_systems.kernels.triton_swiglu import triton_silu_mul


def reference_silu_mul(gate, up):
    return F.silu(gate) * up


METHODS = {"triton": triton_silu_mul, "ref": reference_silu_mul}


def make_fwd_bwd(fn, gate, up, grad_out):
    def run():
        gate.grad = None
        up.grad = None
        fn(gate, up).backward(grad_out)
    return run


def peak_memory_mb(run):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    run()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


def main():
    parser = argparse.ArgumentParser(description="Fused SwiGLU benchmark")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--rows", type=int, default=16384, help="rows = batch*seq")
    parser.add_argument("--dffs", type=int, nargs="+", default=[1344, 2048, 4096, 8192, 11008])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA.")

    dtype = getattr(torch, args.dtype)
    print(f"device={torch.cuda.get_device_name()} dtype={args.dtype} rows={args.rows}")
    print(f"{'d_ff':>6} | {'triton ms':>9} {'ref ms':>8} | {'x_ref':>6} | "
          f"{'triton MB':>9} {'ref MB':>8} {'mem save':>8}")
    print("-" * 70)

    for d_ff in args.dffs:
        torch.manual_seed(0)
        gate = torch.randn(args.rows, d_ff, device="cuda", dtype=dtype, requires_grad=True)
        up = torch.randn(args.rows, d_ff, device="cuda", dtype=dtype, requires_grad=True)
        grad_out = torch.randn(args.rows, d_ff, device="cuda", dtype=dtype)

        ms = {name: triton.testing.do_bench(make_fwd_bwd(fn, gate, up, grad_out)) for name, fn in METHODS.items()}
        mem_triton = peak_memory_mb(make_fwd_bwd(triton_silu_mul, gate, up, grad_out))
        mem_ref = peak_memory_mb(make_fwd_bwd(reference_silu_mul, gate, up, grad_out))

        print(f"{d_ff:>6} | {ms['triton']:>9.3f} {ms['ref']:>8.3f} | "
              f"{ms['ref'] / ms['triton']:>5.2f}x | "
              f"{mem_triton:>9.1f} {mem_ref:>8.1f} {mem_ref / mem_triton:>7.2f}x")


if __name__ == "__main__":
    main()
