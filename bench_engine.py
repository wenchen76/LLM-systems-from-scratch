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
"""
import argparse
import time

import torch

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


def make_requests(n, vocab_size, seed=0):
    """Varied prompt lengths and output lengths — the regime where static batching wastes the most."""
    gen = torch.Generator().manual_seed(seed)
    reqs = []
    for _ in range(n):
        prompt_len = int(torch.randint(4, 32, (1,), generator=gen))
        max_tokens = int(torch.randint(8, 96, (1,), generator=gen))  # wide spread
        prompt = torch.randint(0, vocab_size, (prompt_len,), generator=gen).tolist()
        reqs.append((prompt, SamplingParams(max_tokens=max_tokens, top_k=50)))
    return reqs


def total_tokens(requests):
    return sum(sp.max_tokens for _, sp in requests)


def run_continuous(model, requests, batch, device, trace=None):
    engine = LLMEngine(model, device=device, max_running=batch)
    for prompt, sp in requests:
        engine.add_request(prompt, sp)
    steps = 0
    while engine.has_work():
        if trace is not None:
            trace.append(len(engine.running))
        engine.step()
        steps += 1
    return steps


def run_static(model, requests, batch, device, trace=None):
    """Admit in fixed waves; the next wave waits until the current one fully drains."""
    engine = LLMEngine(model, device=device, max_running=batch)
    i, steps = 0, 0
    while i < len(requests) or engine.running:
        if not engine.running:  # current wave drained -> admit the next
            for prompt, sp in requests[i : i + batch]:
                engine.add_request(prompt, sp)
            i += batch
        if trace is not None:
            trace.append(len(engine.running))
        engine.step()
        steps += 1
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
    return time.perf_counter() - t0, steps


def main():
    parser = argparse.ArgumentParser(description="Continuous vs static batching stress test")
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--batch", type=int, default=8, help="Max concurrent requests (slots)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace", action="store_true",
                        help="Print per-step batch occupancy (slot utilization) for both policies")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = TransformerLM(**DEFAULT_CONFIG).to(args.device).eval()
    requests = make_requests(args.requests, DEFAULT_CONFIG["vocab_size"], seed=args.seed)
    tokens = total_tokens(requests)

    print(f"device={args.device} requests={args.requests} slots={args.batch} "
          f"tokens_to_generate={tokens}")
    print(f"output lengths: min={min(sp.max_tokens for _, sp in requests)} "
          f"max={max(sp.max_tokens for _, sp in requests)}")

    # Warm up kernels / allocator.
    time_run(lambda: run_continuous(model, requests[:args.batch], args.batch, args.device), args.device)

    static_t, static_steps = time_run(lambda: run_static(model, requests, args.batch, args.device), args.device)
    cont_t, cont_steps = time_run(lambda: run_continuous(model, requests, args.batch, args.device), args.device)

    print("-" * 60)
    print(f"static batching     : {static_t:6.2f}s  {static_steps:4d} steps  {tokens / static_t:8.1f} tok/s")
    print(f"continuous batching : {cont_t:6.2f}s  {cont_steps:4d} steps  {tokens / cont_t:8.1f} tok/s")
    print(f"speedup             : {static_t / cont_t:.2f}x  ({static_steps / cont_steps:.2f}x fewer steps)")

    if args.trace:
        st, ct = [], []
        run_static(model, requests, args.batch, args.device, trace=st)
        run_continuous(model, requests, args.batch, args.device, trace=ct)
        print(f"\nslot occupancy per step (each bar = running / {args.batch} slots):")
        print(f"  static     |{sparkline(st, args.batch)}|  (drains to empty between waves)")
        print(f"  continuous |{sparkline(ct, args.batch)}|  (stays full, refilled mid-flight)")


if __name__ == "__main__":
    main()
