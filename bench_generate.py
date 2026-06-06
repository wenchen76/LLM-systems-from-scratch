"""Phase 0 baseline benchmark for autoregressive generation.

Measures decode throughput (tokens/sec) and peak memory for the current
no-KV-cache `TransformerLM.generate`. Perf is independent of weight quality,
so by default this builds a randomly-initialized model from a config and feeds
random token IDs — no checkpoint, vocab, or merges needed. The numbers it
prints are the baseline that the KV-cache / continuous-batching work (Phases
1-4) must beat. Keep the config fixed across runs so comparisons are valid.

Examples:
    python bench_generate.py
    python bench_generate.py --batch-size 8 --max-tokens 256
    python bench_generate.py --config model_config.json --checkpoint ckpt_final.pt
"""
import argparse
import json
import time

import torch

from llm_core.model import TransformerLM

# Small-but-representative default model used when no --config is given.
# Override with --config to benchmark whatever config you actually train.
DEFAULT_CONFIG = {
    "vocab_size": 10000,
    "context_length": 512,
    "d_model": 512,
    "num_layers": 8,
    "num_heads": 8,
    "d_ff": 2048,
    "rope_theta": 10000.0,
}


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    """Block until all queued device work finishes, so timing is accurate."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(device: str) -> float | None:
    """Peak allocated memory in MB, or None if the device can't report it."""
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    if device == "mps":
        return torch.mps.current_allocated_memory() / 1e6
    return None


def main():
    parser = argparse.ArgumentParser(description="Baseline generation benchmark")
    parser.add_argument("--config", type=str, default=None,
                        help="Model config JSON. If omitted, uses a built-in default config.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional checkpoint to load (weights don't affect speed).")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, mps")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=1, help="Number of sequences generated in parallel")
    parser.add_argument("--prompt-len", type=int, default=128, help="Prompt length per sequence")
    parser.add_argument("--max-tokens", type=int, default=128, help="New tokens to generate per sequence")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50, help="Exercises the top-k path; set 0 to disable")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs (averaged), after 1 warmup")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="bench_baseline.json", help="Where to append results")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)

    # Build the model: from a config, optionally loading real weights.
    if args.checkpoint and args.config:
        model = TransformerLM.from_pretrained(args.config, args.checkpoint)
        config = model.config
    else:
        if args.config:
            with open(args.config) as f:
                config = json.load(f)
        else:
            config = dict(DEFAULT_CONFIG)
        model = TransformerLM(**config)
    model = model.to(device=device, dtype=dtype).eval()

    vocab_size = config["vocab_size"]
    context_length = config["context_length"]
    if args.prompt_len > context_length:
        parser.error(f"--prompt-len {args.prompt_len} exceeds context_length {context_length}")

    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    # Fixed random prompt; seeded so every run/version sees the same input.
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    prompt = torch.randint(
        0, vocab_size, (args.batch_size, args.prompt_len), generator=gen, dtype=torch.long
    ).to(device)

    def run_once() -> torch.Tensor:
        with torch.no_grad():
            return model.generate(
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=top_k,
                eos_token_id=None,  # never stop early -> deterministic token count
            )

    print(f"device={device} dtype={args.dtype} batch={args.batch_size} "
          f"prompt_len={args.prompt_len} max_tokens={args.max_tokens} top_k={top_k}")
    print(f"params={model.get_num_params() / 1e6:.2f}M (non-embedding)")

    # Warmup (kernel compile, allocator warm, autotune for Triton/flash).
    run_once()
    sync(device)

    latencies = []
    reset_peak_memory(device)
    for i in range(args.runs):
        sync(device)
        t0 = time.perf_counter()
        out = run_once()
        sync(device)
        dt = time.perf_counter() - t0
        latencies.append(dt)
        gen_tokens = out.size(0) * out.size(1)
        print(f"  run {i + 1}: {dt:.3f}s  {gen_tokens / dt:,.1f} tok/s")

    avg = sum(latencies) / len(latencies)
    total_new_tokens = args.batch_size * args.max_tokens
    throughput = total_new_tokens / avg
    peak_mb = peak_memory_mb(device)

    print("-" * 60)
    print(f"avg latency : {avg:.3f}s over {args.runs} runs")
    print(f"throughput  : {throughput:,.1f} tokens/sec ({total_new_tokens} new tokens / run)")
    print(f"per-token   : {avg / args.max_tokens * 1e3:.2f} ms/step")
    print(f"peak memory : {f'{peak_mb:,.1f} MB' if peak_mb is not None else 'n/a (cpu)'}")

    record = {
        "label": "baseline-no-kv-cache",
        "device": device,
        "dtype": args.dtype,
        "config": config,
        "batch_size": args.batch_size,
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "top_k": top_k,
        "temperature": args.temperature,
        "runs": args.runs,
        "seed": args.seed,
        "avg_latency_s": avg,
        "throughput_tok_s": throughput,
        "ms_per_step": avg / args.max_tokens * 1e3,
        "peak_memory_mb": peak_mb,
    }
    with open(args.out, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nappended result to {args.out}")


if __name__ == "__main__":
    main()
