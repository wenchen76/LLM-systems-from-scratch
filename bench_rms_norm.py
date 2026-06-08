"""Performance benchmark for the fused Triton RMSNorm kernel.

RMSNorm is memory-bound (elementwise + a row reduction). The fused kernel does
the forward in one pass and the backward in one pass, versus the several
intermediate tensors and kernel launches the eager path makes. We report
forward+backward time and peak memory across model dims, comparing:
  - triton: the fused kernel
  - ref:    llm_core.model.RMSNorm (eager)
  - torchF: torch.nn.functional.rms_norm (PyTorch's fused path), if available

CUDA + Triton only.

    python bench_rms_norm.py
    python bench_rms_norm.py --rows 32768 --dtype bfloat16
"""
import argparse

import torch
import torch.nn.functional as F
import triton

from llm_core.model import RMSNorm
from llm_systems.kernels.triton_rms_norm import TritonRMSNorm

HAS_TORCH_RMS = hasattr(F, "rms_norm")


def make_fwd_bwd(forward, x, grad_out):
    def run():
        x.grad = None
        forward(x).backward(grad_out)
    return run


def peak_memory_mb(run):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    run()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


def main():
    parser = argparse.ArgumentParser(description="Fused RMSNorm benchmark")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--rows", type=int, default=16384, help="rows = batch*seq")
    parser.add_argument("--dims", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA.")

    dtype = getattr(torch, args.dtype)
    print(f"device={torch.cuda.get_device_name()} dtype={args.dtype} rows={args.rows} "
          f"torchF={'yes' if HAS_TORCH_RMS else 'unavailable'}")
    print(f"{'d':>6} | {'triton ms':>9} {'ref ms':>8} {'torchF ms':>9} | "
          f"{'x_ref':>6} {'x_torchF':>8} | {'triton MB':>9} {'torchF MB':>9}")
    print("-" * 86)

    for d in args.dims:
        torch.manual_seed(0)
        x = torch.randn(args.rows, d, device="cuda", dtype=dtype, requires_grad=True)
        grad_out = torch.randn(args.rows, d, device="cuda", dtype=dtype)
        weight = torch.randn(d, device="cuda", requires_grad=True)

        triton_norm = TritonRMSNorm(d).cuda()
        ref_norm = RMSNorm(d).cuda()
        forwards = {
            "triton": triton_norm,
            "ref": ref_norm,
            "torchF": (lambda x: F.rms_norm(x, (d,), weight, 1e-5)) if HAS_TORCH_RMS else None,
        }

        ms = {}
        for name, fwd in forwards.items():
            ms[name] = triton.testing.do_bench(make_fwd_bwd(fwd, x, grad_out)) if fwd else float("nan")
        mem_triton = peak_memory_mb(make_fwd_bwd(triton_norm, x, grad_out))
        mem_torch = peak_memory_mb(make_fwd_bwd(forwards["torchF"], x, grad_out)) if HAS_TORCH_RMS else float("nan")

        print(f"{d:>6} | {ms['triton']:>9.3f} {ms['ref']:>8.3f} {ms['torchF']:>9.3f} | "
              f"{ms['ref'] / ms['triton']:>5.2f}x {ms['torchF'] / ms['triton']:>7.2f}x | "
              f"{mem_triton:>9.1f} {mem_torch:>9.1f}")


if __name__ == "__main__":
    main()
