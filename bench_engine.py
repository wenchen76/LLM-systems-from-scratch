"""Phase 4 stress test: continuous batching vs static batching.

Both policies use the exact same LLMEngine and kernels; the ONLY difference is
admission:
  - continuous: a finished request frees its slot immediately and the next
    waiting request is admitted mid-flight, so the batch stays full.
  - static: requests are admitted in fixed waves; a wave must fully drain (wait
    for its longest sequence) before the next wave starts, so short requests
    hold slots idle.

With varied output lengths this exposes the static-batching bubble. Perf is
independent of weights, so we use a random-init model and random prompts.

    python bench_engine.py
    python bench_engine.py --requests 64 --batch 8 --device cpu
    python bench_engine.py --config configures/gpt3xl.yaml --device cuda --dtype bfloat16
"""
import argparse
import time

import torch
import yaml

from llm_core.engine import LLMEngine, SamplingParams
from llm_core.model import TransformerLM

DEFAULT_CONFIG = {
    "vocab_size": 10000,
    "context_length": 1024,
    "d_model": 512,
    "num_layers": 8,
    "num_heads": 8,
    "d_ff": 2048,
    "rope_theta": 10000.0,
}


def load_model_config(path):
    """Model dims from a training YAML's `model` section, or the built-in default."""
    if path is None:
        return dict(DEFAULT_CONFIG)
    with open(path) as f:
        return yaml.safe_load(f)["model"]


def make_requests(n, vocab_size, min_tokens=8, max_tokens=96, seed=0):
    """Varied prompt and output lengths. Output length is drawn from [min_tokens, max_tokens];
    raise both for long cache lengths (more pure-decode steps, where CUDA-graph replay helps)."""
    gen = torch.Generator().manual_seed(seed)
    reqs = []
    for _ in range(n):
        prompt_len = int(torch.randint(4, 32, (1,), generator=gen))
        out_len = int(torch.randint(min_tokens, max_tokens + 1, (1,), generator=gen))
        prompt = torch.randint(0, vocab_size, (prompt_len,), generator=gen).tolist()
        reqs.append((prompt, SamplingParams(max_tokens=out_len, top_k=50)))
    return reqs


def total_tokens(requests):
    return sum(sp.max_tokens for _, sp in requests)


def _log(label, steps, engine, done, total, log_every):
    if log_every and steps % log_every == 0:
        print(f"  [{label}] step {steps:4d}  running={len(engine.running):3d}  done={done}/{total}", flush=True)


def run_continuous(model, requests, batch, device, trace=None, label="continuous", log_every=0, engine_kwargs=None):
    engine = LLMEngine(model, device=device, max_running=batch, **(engine_kwargs or {}))
    for prompt, sp in requests:
        engine.add_request(prompt, sp)
    steps = done = 0
    while engine.has_work():
        if trace is not None:
            trace.append(len(engine.running))
        done += len(engine.step())
        steps += 1
        _log(label, steps, engine, done, len(requests), log_every)
    return steps


def run_static(model, requests, batch, device, trace=None, label="static", log_every=0, engine_kwargs=None):
    """Admit in fixed waves; the next wave waits until the current one fully drains."""
    engine = LLMEngine(model, device=device, max_running=batch, **(engine_kwargs or {}))
    i, steps, done = 0, 0, 0
    while i < len(requests) or engine.running:
        if not engine.running:  # current wave drained -> admit the next
            for prompt, sp in requests[i : i + batch]:
                engine.add_request(prompt, sp)
            i += batch
        if trace is not None:
            trace.append(len(engine.running))
        done += len(engine.step())
        steps += 1
        _log(label, steps, engine, done, len(requests), log_every)
    return steps


def sparkline(occupancy, batch):
    """Render per-step running-request count as a bar sparkline."""
    blocks = " ▁▂▃▄▅▆▇█"
    return "".join(blocks[min(len(blocks) - 1, round(n / batch * (len(blocks) - 1)))] for n in occupancy)


def time_run(fn, device):
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps = fn()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if device == "cuda":
        # fn built its own engine (and KV pools) internally; it is unreferenced now, so
        # reclaim its VRAM before the next phase allocates — matters on small GPUs.
        torch.cuda.empty_cache()
    return elapsed, steps


def main():
    parser = argparse.ArgumentParser(description="Continuous vs static batching stress test")
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--batch", type=int, default=8, help="Max concurrent requests (slots)")
    parser.add_argument("--min-tokens", type=int, default=8, help="Lower bound on generated tokens per request")
    parser.add_argument("--max-tokens", type=int, default=96,
                        help="Upper bound on generated tokens per request (raise both for long cache lengths)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, mps")
    parser.add_argument("--config", type=str, default=None,
                        help="Training YAML for model dims (e.g. configures/gpt3xl.yaml); default: small built-in")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Disable the custom Triton flash-attention kernels (use eager attention); on by default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace", action="store_true",
                        help="Print per-step batch occupancy (slot utilization) for both policies")
    parser.add_argument("--paged", action="store_true", help="Use a paged KV cache")
    parser.add_argument("--block-size", type=int, default=16, help="Paged KV block size")
    parser.add_argument("--num-blocks", type=int, default=2048, help="Paged KV blocks per layer")
    parser.add_argument("--cuda-graph", action="store_true",
                        help="Replay a captured CUDA graph per bucket for decode steps (paged)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model_cfg = load_model_config(args.config)
    dtype = getattr(torch, args.dtype)
    use_flash_attn = not args.no_flash_attn
    model = TransformerLM(**model_cfg, use_flash_attn=use_flash_attn).to(device=args.device, dtype=dtype).eval()
    requests = make_requests(args.requests, model_cfg["vocab_size"],
                             min_tokens=args.min_tokens, max_tokens=args.max_tokens, seed=args.seed)
    tokens = total_tokens(requests)
    engine_kwargs = dict(paged=args.paged, block_size=args.block_size, num_blocks=args.num_blocks,
                         cuda_graph=args.cuda_graph)

    print(f"device={args.device} dtype={args.dtype} d_model={model_cfg['d_model']} layers={model_cfg['num_layers']} "
          f"requests={args.requests} slots={args.batch} paged={args.paged} flash={use_flash_attn} tokens_to_generate={tokens}")
    print(f"output lengths: min={min(sp.max_tokens for _, sp in requests)} "
          f"max={max(sp.max_tokens for _, sp in requests)}")

    # Warm up kernels / allocator.
    print("warming up...", flush=True)
    time_run(lambda: run_continuous(model, requests[:args.batch], args.batch, args.device, engine_kwargs=engine_kwargs), args.device)

    print("running static batching...", flush=True)
    static_t, static_steps = time_run(
        lambda: run_static(model, requests, args.batch, args.device, label="static", log_every=20, engine_kwargs=engine_kwargs), args.device)
    print("running continuous batching...", flush=True)
    cont_t, cont_steps = time_run(
        lambda: run_continuous(model, requests, args.batch, args.device, label="continuous", log_every=20, engine_kwargs=engine_kwargs), args.device)

    print("-" * 60)
    print(f"static batching     : {static_t:6.2f}s  {static_steps:4d} steps  {tokens / static_t:8.1f} tok/s")
    print(f"continuous batching : {cont_t:6.2f}s  {cont_steps:4d} steps  {tokens / cont_t:8.1f} tok/s")
    print(f"speedup             : {static_t / cont_t:.2f}x  ({static_steps / cont_steps:.2f}x fewer steps)")

    if args.trace:
        st, ct = [], []
        run_static(model, requests, args.batch, args.device, trace=st, engine_kwargs=engine_kwargs)
        run_continuous(model, requests, args.batch, args.device, trace=ct, engine_kwargs=engine_kwargs)
        print(f"\nslot occupancy per step (each bar = running / {args.batch} slots):")
        print(f"  static     |{sparkline(st, args.batch)}|  (drains to empty between waves)")
        print(f"  continuous |{sparkline(ct, args.batch)}|  (stays full, refilled mid-flight)")


if __name__ == "__main__":
    main()
