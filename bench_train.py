"""Benchmark --custom-triton vs the PyTorch baseline for a full training step.

Replicates train.py's step (forward -> cross-entropy -> backward -> AdamW.step)
with random data, sweeping batch sizes, for both paths:
  - baseline: eager TransformerLM + llm_core AdamW + llm_core cross_entropy
  - triton:   use_custom_triton model (RMSNorm/SwiGLU) + FusedAdamW + fused CE

It reports steady-state per-step time and peak memory so you can quote
"--custom-triton: N% faster, M% lower peak memory vs the PyTorch baseline".

Notes:
- flash attention is ON for both, so the delta isolates the RMSNorm/SwiGLU/CE
  kernels (and the fused optimizer), not attention.
- the memory win is dominated by the fused cross-entropy avoiding the
  (batch*seq, vocab) softmax materialization, so the gap grows with batch — the
  sweep makes that visible. Batches that OOM on the baseline are reported as OOM.

CUDA only (the triton path needs Triton/CUDA).

    python bench_train.py --config configures/gpt3xl.yaml --amp --batches 2 4 8
"""
import argparse
import time

import torch
import yaml

from llm_core.model import TransformerLM
from llm_core.nn_functional import cross_entropy
from llm_core.optimizer import AdamW


def build(setting, model_cfg, optim_cfg, device, compile_model=False):
    use_triton = setting == "triton"
    model = TransformerLM(
        vocab_size=model_cfg["vocab_size"],
        context_length=model_cfg["context_length"],
        d_model=model_cfg["d_model"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        d_ff=model_cfg["d_ff"],
        rope_theta=model_cfg["rope_theta"],
        use_flash_attn=True,            # fixed across all settings
        use_custom_triton=use_triton,
    ).to(device)
    model.train()
    lr = float(optim_cfg["learning_rate_max"])
    weight_decay = float(optim_cfg["weight_decay"])
    opt_params = model.parameters()
    if compile_model:
        # Only ever applied to the eager baseline — never the custom-kernel path,
        # whose autograd.Functions would graph-break under torch.compile.
        model = torch.compile(model)
    if use_triton:
        from llm_systems.kernels.triton_adamw import FusedAdamW
        from llm_systems.kernels.triton_cross_entropy import triton_cross_entropy
        return model, FusedAdamW(opt_params, lr=lr, weight_decay=weight_decay), triton_cross_entropy
    return model, AdamW(opt_params, lr=lr, weight_decay=weight_decay), cross_entropy


def run_step(model, opt, ce, x, y, amp, device):
    model.zero_grad()
    with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
        logits = model(x)
        loss = ce(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    opt.step()
    return loss


def bench_setting(setting, cfg, batches, amp, device, warmup, iters, compile_model=False):
    mc = cfg["model"]
    ctx, vocab = mc["context_length"], mc["vocab_size"]
    model, opt, ce = build(setting, mc, cfg["optimizer"], device, compile_model=compile_model)

    results = {}
    oom = False
    for b in batches:
        if oom:  # once a batch OOMs, all larger ones will too
            results[b] = "OOM"
            print(f"  [{setting} b={b}] skipped (a smaller batch already OOM'd)")
            continue
        try:
            x = torch.randint(0, vocab, (b, ctx), device=device)
            y = torch.randint(0, vocab, (b, ctx), device=device)
            for w in range(warmup):  # warm allocator + Triton autotune, allocate optimizer state
                run_step(model, opt, ce, x, y, amp, device)
                print(f"  [{setting} b={b}] warmup {w + 1}/{warmup}  mem {torch.cuda.memory_allocated() / 1e9:.1f} GB")
            torch.cuda.synchronize()
            
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for it in range(iters):
                run_step(model, opt, ce, x, y, amp, device)
                print(f"  [{setting} b={b}] step {it + 1}/{iters}  mem {torch.cuda.memory_allocated() / 1e9:.1f} GB")
            torch.cuda.synchronize()
            ms = (time.time() - t0) / iters * 1e3
            gb = torch.cuda.max_memory_allocated() / 1e9
            results[b] = (ms, gb)
            del x, y
        except torch.cuda.OutOfMemoryError:
            results[b] = "OOM"
            oom = True
            print(f"  [{setting} b={b}] OOM")
            torch.cuda.empty_cache()
    del model, opt
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark --custom-triton vs PyTorch baseline per training step")
    parser.add_argument("--config", default="configures/gpt3xl.yaml")
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--amp", action="store_true", help="bf16 autocast (matches train.py --amp)")
    parser.add_argument("--compile-baseline", action="store_true",
                        help="Also benchmark a torch.compile'd eager baseline (the 'beat the compiler' bar)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA.")
    device = "cuda"

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model"]
    print(f"device={torch.cuda.get_device_name()} amp={args.amp} "
          f"model: d_model={mc['d_model']} layers={mc['num_layers']} vocab={mc['vocab_size']} ctx={mc['context_length']}")

    # (label, build setting, compile?). Each is swept fully and freed before the
    # next, so the 1.5B models never coexist on the GPU.
    runs = [("baseline", "baseline", False)]
    if args.compile_baseline:
        runs.append(("compiled", "baseline", True))
    runs.append(("triton", "triton", False))

    results = {}
    for label, setting, comp in runs:
        results[label] = bench_setting(setting, cfg, args.batches, args.amp, device,
                                       args.warmup, args.iters, compile_model=comp)

    labels = [r[0] for r in runs]
    header = f"{'batch':>6} |"
    for lb in labels:
        header += f" {lb + ' ms':>10} {lb + ' GB':>9} |"
    header += f" {'x_base':>7}"
    if "compiled" in labels:
        header += f" {'x_comp':>7}"
    header += f" {'mem<base':>9}"
    print("\n" + header)
    print("-" * len(header))

    for b in args.batches:
        row = f"{b:>6} |"
        cells = {}
        for lb in labels:
            o = results[lb][b]
            cells[lb] = o if isinstance(o, tuple) else None
            row += (f" {o[0]:>10.1f} {o[1]:>9.2f} |" if isinstance(o, tuple)
                    else f" {str(o):>10} {'':>9} |")
        base, tri, comp = cells.get("baseline"), cells.get("triton"), cells.get("compiled")
        row += f" {(f'{base[0] / tri[0]:.2f}x' if base and tri else '-'):>7}"
        if "compiled" in labels:
            row += f" {(f'{comp[0] / tri[0]:.2f}x' if comp and tri else '-'):>7}"
        row += f" {(f'{(1 - tri[1] / base[1]) * 100:.1f}%' if base and tri else '-'):>9}"
        print(row)


if __name__ == "__main__":
    main()
