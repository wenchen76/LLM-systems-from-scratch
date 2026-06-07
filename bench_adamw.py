"""Performance benchmark for the fused Triton AdamW kernel.

AdamW is memory-bandwidth-bound (pure elementwise, no reuse), so the figure of
merit is effective bandwidth: the kernel moves 7N transactions (read p,g,m,v;
write p,m,v) versus the 12N of the unfused update. We report ms and GB/s for
three optimizers:
  - fused: our Triton FusedAdamW
  - ref:   llm_core.optimizer.AdamW, the pure-PyTorch (eager) implementation the
           kernel replaces — same algorithm, so "x_ref" isolates fusion + the
           launch/temporary overhead the eager version pays.
  - torchF: torch.optim.AdamW(fused=True), PyTorch's own fused CUDA kernel — the
           honest "production-grade" comparison ("x_torchF").

CUDA + Triton only.

    python bench_adamw.py
    python bench_adamw.py --dtype bfloat16
"""
import argparse

import torch
import triton

from llm_core.optimizer import AdamW as ReferenceAdamW
from llm_systems.kernels.triton_adamw import FusedAdamW


def bytes_per_element(param_dtype: torch.dtype) -> int:
    """Read p,g,m,v + write p,m,v. p,g are param dtype; m,v are always fp32."""
    pg = torch.tensor([], dtype=param_dtype).element_size()
    f32 = 4
    reads = 2 * pg + 2 * f32
    writes = pg + 2 * f32
    return reads + writes


def make_step(build_opt, n, dtype):
    p = torch.randn(n, device="cuda", dtype=dtype, requires_grad=True)
    p.grad = torch.randn_like(p)
    opt = build_opt([p])

    def step():
        opt.step()

    return step


def main():
    parser = argparse.ArgumentParser(description="Fused AdamW benchmark")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[1 << e for e in (16, 18, 20, 22, 24)])  # 64K .. 16M
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA.")

    dtype = getattr(torch, args.dtype)
    elem_bytes = bytes_per_element(dtype)
    print(f"device={torch.cuda.get_device_name()} dtype={args.dtype} "
          f"bytes/elem={elem_bytes}")
    print(f"{'N':>10} | {'fused ms':>8} {'GB/s':>7} | {'ref ms':>8} {'GB/s':>7} | "
          f"{'torchF ms':>9} {'GB/s':>7} | {'x_ref':>6} {'x_torchF':>8}")
    print("-" * 88)

    for n in args.sizes:
        fused_ms = triton.testing.do_bench(make_step(lambda ps: FusedAdamW(ps), n, dtype))
        ref_ms = triton.testing.do_bench(make_step(lambda ps: ReferenceAdamW(ps), n, dtype))
        torch_ms = triton.testing.do_bench(make_step(lambda ps: torch.optim.AdamW(ps, fused=True), n, dtype))
        gbps = lambda ms: n * elem_bytes / (ms * 1e-3) / 1e9  # noqa: E731
        print(f"{n:>10} | {fused_ms:>8.3f} {gbps(fused_ms):>7.1f} | "
              f"{ref_ms:>8.3f} {gbps(ref_ms):>7.1f} | "
              f"{torch_ms:>9.3f} {gbps(torch_ms):>7.1f} | "
              f"{ref_ms / fused_ms:>5.2f}x {torch_ms / fused_ms:>7.2f}x")


if __name__ == "__main__":
    main()
