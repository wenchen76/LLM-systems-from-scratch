"""Performance benchmark for the fused Triton cross-entropy kernel.

The fused kernel computes the loss and writes the gradient in a single pass over
the logits, never materializing the (BT, V) softmax/log-softmax that the eager
path allocates. So the two figures of merit are forward+backward time and peak
memory. We compare three implementations across vocab sizes:
  - triton: the fused kernel
  - ref:    llm_core.nn_functional.cross_entropy (eager, materializes log_softmax)
  - torchF: torch.nn.functional.cross_entropy (PyTorch's own fused-ish ATen path)

Each timed iteration clones the logits first (the kernel overwrites them with the
gradient in place); all three pay the same clone, so the comparison stays fair.

CUDA + Triton only.

    python bench_cross_entropy.py
    python bench_cross_entropy.py --bt 8192 --dtype bfloat16
"""
import argparse

import torch
import triton
import torch.nn.functional as F

from llm_core.nn_functional import cross_entropy as reference_cross_entropy
from llm_systems.kernels.triton_cross_entropy import triton_cross_entropy

METHODS = {
    "triton": triton_cross_entropy,
    "ref": reference_cross_entropy,
    "torchF": F.cross_entropy,
}


def make_fwd_bwd(loss_fn, base, targets):
    def run():
        x = base.clone().requires_grad_(True)
        loss_fn(x, targets).backward()
    return run


def peak_memory_mb(run):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    run()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


def main():
    parser = argparse.ArgumentParser(description="Fused cross-entropy benchmark")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--bt", type=int, default=4096, help="rows = batch*seq")
    parser.add_argument("--vocabs", type=int, nargs="+", default=[10000, 32000, 50257, 128000])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA.")

    dtype = getattr(torch, args.dtype)
    print(f"device={torch.cuda.get_device_name()} dtype={args.dtype} rows(BT)={args.bt}")
    print(f"{'V':>8} | {'triton ms':>9} {'ref ms':>8} {'torchF ms':>9} | "
          f"{'x_ref':>6} {'x_torchF':>8} | {'triton MB':>9} {'torchF MB':>9} {'mem save':>8}")
    print("-" * 92)

    for vocab in args.vocabs:
        torch.manual_seed(0)
        base = torch.randn(args.bt, vocab, device="cuda", dtype=dtype)
        targets = torch.randint(0, vocab, (args.bt,), device="cuda")

        ms = {name: triton.testing.do_bench(make_fwd_bwd(fn, base, targets)) for name, fn in METHODS.items()}
        mem_triton = peak_memory_mb(make_fwd_bwd(triton_cross_entropy, base, targets))
        mem_torch = peak_memory_mb(make_fwd_bwd(F.cross_entropy, base, targets))

        print(f"{vocab:>8} | {ms['triton']:>9.3f} {ms['ref']:>8.3f} {ms['torchF']:>9.3f} | "
              f"{ms['ref'] / ms['triton']:>5.2f}x {ms['torchF'] / ms['triton']:>7.2f}x | "
              f"{mem_triton:>9.1f} {mem_torch:>9.1f} {mem_torch / mem_triton:>7.2f}x")


if __name__ == "__main__":
    main()
