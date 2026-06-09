"""Weak-scaling efficiency benchmark for the custom DDP and FSDP wrappers.

Measures how throughput scales as GPUs are added, with a FIXED per-GPU (local)
batch — the standard data-parallel metric. For each N in --gpus it spawns N
workers that build the model, wrap it in the repo's DDP or FSDP, and run the same
step as train.py (forward -> cross-entropy -> backward -> grad sync -> clip ->
AdamW) on random data, with no checkpoint / validation / data-file I/O.

Scaling efficiency:

    E(N) = TPS(N) / (N * TPS(1))

where TPS is global tokens/sec at fixed local batch. Equivalently E(N) =
step_time(1) / step_time(N): perfect overlap keeps the step time flat as GPUs are
added; communication overhead stretches it and drops the efficiency.

CUDA + NCCL, multi-GPU node.

    python bench_scaling.py --mode ddp  --gpus 1 2 4 8 --local-batch 2 --amp --flash-attn
    python bench_scaling.py --mode fsdp --gpus 1 2 4 8 --local-batch 4 --amp --flash-attn
"""
import argparse
import json
import os
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from llm_core.model import TransformerLM
from llm_core.nn_functional import clip_gradient, cross_entropy
from llm_core.optimizer import AdamW


def worker(rank, world_size, mode, model_cfg, optim_cfg, local_batch, amp, flash, warmup, iters, result_path):
    distributed = world_size > 1
    if distributed:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        local_rank = rank % torch.cuda.device_count()
    else:
        local_rank = 0
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    ctx, vocab = model_cfg["context_length"], model_cfg["vocab_size"]
    model = TransformerLM(
        vocab_size=vocab,
        context_length=ctx,
        d_model=model_cfg["d_model"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        d_ff=model_cfg["d_ff"],
        rope_theta=model_cfg["rope_theta"],
        use_flash_attn=flash,
    ).to(device)
    model.train()
    if distributed and mode == "ddp":
        from llm_systems.parallelism.ddp import DDP
        model = DDP(model)
    elif distributed and mode == "fsdp":
        from llm_systems.parallelism.fsdp_zero3 import FSDP
        model = FSDP(model)

    opt = AdamW(model.parameters(), lr=float(optim_cfg["learning_rate_max"]),
                weight_decay=float(optim_cfg["weight_decay"]))
    max_grad_norm = float(optim_cfg["max_grad_norm"])

    x = torch.randint(0, vocab, (local_batch, ctx), device=device)
    y = torch.randint(0, vocab, (local_batch, ctx), device=device)

    def step():
        model.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        if distributed:
            model.finish_gradient_synchronization()
        clip_gradient(model.parameters(), max_grad_norm)
        opt.step()

    for _ in range(warmup):  # NCCL/alloc warmup, optimizer-state allocation
        step()
    if distributed:
        dist.barrier()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    if distributed:
        dist.barrier()
    step_time = (time.time() - t0) / iters

    if rank == 0:
        global_batch = world_size * local_batch
        with open(result_path, "w") as f:
            json.dump({
                "global_tps": global_batch * ctx / step_time,
                "step_ms": step_time * 1e3,
                "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            }, f)
    if distributed:
        dist.destroy_process_group()


def run_n(n, args, model_cfg, optim_cfg):
    """Spawn n workers (always via mp.spawn so each run is an isolated process)."""
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
        result_path = tf.name
    mp.spawn(
        worker,
        args=(n, args.mode, model_cfg, optim_cfg, args.local_batch,
              args.amp, args.flash_attn, args.warmup, args.iters, result_path),
        nprocs=n,
        join=True,
    )
    with open(result_path) as f:
        result = json.load(f)
    os.unlink(result_path)
    return result


def main():
    parser = argparse.ArgumentParser(description="DDP / FSDP weak-scaling efficiency benchmark")
    parser.add_argument("--mode", choices=["ddp", "fsdp"], required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 4, 8], help="GPU counts to sweep")
    parser.add_argument("--local-batch", type=int, default=2, help="Per-GPU batch (held fixed across N)")
    parser.add_argument("--config", default="configures/gpt3xl.yaml")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA + NCCL.")
    n_visible = torch.cuda.device_count()
    if max(args.gpus) > n_visible:
        raise SystemExit(f"--gpus requests {max(args.gpus)} but only {n_visible} GPUs are visible.")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg, optim_cfg = cfg["model"], cfg["optimizer"]
    print(f"mode={args.mode} local_batch={args.local_batch} amp={args.amp} flash={args.flash_attn} "
          f"model: d_model={model_cfg['d_model']} layers={model_cfg['num_layers']} ctx={model_cfg['context_length']}")

    results = {}
    for n in args.gpus:
        print(f"running N={n} ({n} x {args.local_batch} = {n * args.local_batch} global batch)...")
        results[n] = run_n(n, args, model_cfg, optim_cfg)

    tps1 = results[args.gpus[0]]["global_tps"]  # baseline = smallest N (ideally 1)
    base_n = args.gpus[0]
    print(f"\n{'GPUs':>5} | {'global batch':>12} | {'step ms':>8} | {'global tok/s':>12} | "
          f"{'efficiency':>10} | {'peak GB':>8}")
    print("-" * 74)
    for n in args.gpus:
        r = results[n]
        eff = (r["global_tps"] / (n / base_n * tps1)) * 100
        print(f"{n:>5} | {n * args.local_batch:>12} | {r['step_ms']:>8.1f} | {r['global_tps']:>12,.0f} | "
              f"{eff:>9.1f}% | {r['peak_gb']:>8.2f}")
    print(f"\nefficiency = TPS(N) / (N/{base_n} * TPS({base_n})); baseline N={base_n}.")


if __name__ == "__main__":
    main()
