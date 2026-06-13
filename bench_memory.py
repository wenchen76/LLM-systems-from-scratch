"""Memory-efficiency benchmark: paged vs contiguous KV under fixed concurrent load.

The workload is held constant -- `batch` identical requests, each a short prompt
plus `--max-tokens` generated tokens, all coexisting until they finish on the
same step -- and peak GPU memory is measured for both KV backends while sweeping
`batch` upward until one runs out of memory.

The contiguous KVCache grows each request with `torch.cat` and re-concatenates
every running cache into a flat buffer on each decode step, so it churns the
allocator and spikes on transients; the paged pool is allocated once and written
in place. The payoff of paging is that a fixed VRAM budget then sustains more
concurrency (or, fixing --batch and raising --max-tokens, longer context) before
OOM. Perf is weight-independent, so a random-init model is used.

    python bench_memory.py --config configures/gpt3xl.yaml --device cuda \
        --dtype bfloat16 --max-tokens 256 --batches 16 32 64 96 128 192
"""
import argparse

import torch

from bench_engine import load_model_config
from llm_core.engine import LLMEngine, SamplingParams
from llm_core.model import TransformerLM


def cdiv(a, b):
    return (a + b - 1) // b


def run_once(model, batch, prompt_len, gen_len, device, paged, block_size):
    """Run `batch` identical requests concurrently to completion.

    Returns (peak_reserved_gb, peak_allocated_gb), or (None, None) on OOM. For
    paged, the pool is sized to exactly cover this workload's worst case, so both
    backends are asked to hold the same number of KV tokens.
    """
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    num_blocks = batch * cdiv(prompt_len + gen_len, block_size)  # exact worst-case capacity
    sp = SamplingParams(max_tokens=gen_len, top_k=1)
    prompt = [1] * prompt_len
    engine = None
    try:
        engine = LLMEngine(model, device=device, max_running=batch, paged=paged,
                           block_size=block_size, num_blocks=num_blocks)
        for _ in range(batch):
            engine.add_request(list(prompt), sp)
        while engine.has_work():
            engine.step()
        if device != "cuda":
            return 0.0, 0.0
        torch.cuda.synchronize()
        return torch.cuda.max_memory_reserved() / 1e9, torch.cuda.max_memory_allocated() / 1e9
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise
        return None, None
    finally:
        del engine
        if device == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Paged vs contiguous KV memory benchmark")
    parser.add_argument("--config", type=str, default=None,
                        help="Training YAML for model dims (e.g. configures/gpt3xl.yaml); default: small built-in")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max-tokens", type=int, default=256, help="Tokens generated per request (KV depth)")
    parser.add_argument("--prompt-len", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--batches", type=int, nargs="+", default=[16, 32, 64, 96, 128],
                        help="Concurrency levels to sweep (stops once both backends OOM)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = load_model_config(args.config)
    dtype = getattr(torch, args.dtype)
    model = TransformerLM(**cfg, use_flash_attn=True).to(device=args.device, dtype=dtype).eval()

    depth = args.prompt_len + args.max_tokens
    weights_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"device={args.device} dtype={args.dtype} d_model={cfg['d_model']} layers={cfg['num_layers']} "
          f"weights={weights_gb:.2f}GB  prompt={args.prompt_len} gen={args.max_tokens} depth={depth} "
          f"block_size={args.block_size}")
    print(f"{'batch':>6} | {'contiguous resv/alloc GB':>26} | {'paged resv/alloc GB':>22} | note")
    print("-" * 78)

    np_max = pg_max = None
    last_common = None
    for batch in args.batches:
        np_resv, np_alloc = run_once(model, batch, args.prompt_len, args.max_tokens, args.device, False, args.block_size)
        pg_resv, pg_alloc = run_once(model, batch, args.prompt_len, args.max_tokens, args.device, True, args.block_size)
        np_str = "OOM" if np_resv is None else f"{np_resv:.2f}/{np_alloc:.2f}"
        pg_str = "OOM" if pg_resv is None else f"{pg_resv:.2f}/{pg_alloc:.2f}"
        note = ""
        if np_resv and pg_resv:  # both fit (and nonzero -> real cuda measurement)
            note = f"paged {(1 - pg_resv / np_resv) * 100:.0f}% lower peak"
            last_common = (batch, np_resv, pg_resv)
        elif np_resv is None and pg_resv is not None:
            note = "contiguous OOM, paged still fits"
        print(f"{batch:>6} | {np_str:>26} | {pg_str:>22} | {note}")
        if np_resv is not None:
            np_max = batch
        if pg_resv is not None:
            pg_max = batch
        if np_resv is None and pg_resv is None:
            break

    print("-" * 78)
    print(f"contiguous sustained up to batch {np_max}; paged up to batch {pg_max}.")
    if last_common:
        b, npk, pgk = last_common
        print(f"at batch {b} (both fit): paged peak {pgk:.2f} GB vs contiguous {npk:.2f} GB "
              f"({(1 - pgk / npk) * 100:.0f}% lower).")


if __name__ == "__main__":
    main()
