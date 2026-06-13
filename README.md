# LLM Systems from Scratch

An autoregressive Transformer language model built from scratch — plus the systems infrastructure that trains and serves it efficiently: custom Triton kernels, custom FSDP (ZeRO-3) and DDP wrappers, continuous batching, a BPE tokenizer, and a training loop wired up with mixed precision, `torch.compile`, FlashAttention, and Weights & Biases.

Every layer is reimplemented, so the interactions between modeling code, autograd, CUDA streams, NCCL collectives, and Triton kernels stay visible and modifiable.

## Technical depth

The codebase exposes the layers of an LLM training system that frameworks
usually hide behind their APIs:

- **Model internals** — annotated tensor shapes, hand-built RoPE caches, direct
  causal masking, and a self-contained sampling loop.
- **Tokenizer and data path** — a BPE tokenizer that trains from raw text,
  preserves special-token boundaries, and exports `uint32` streams for
  `np.memmap` training.
- **Training runtime** — one YAML-driven entry point compares plain PyTorch,
  BF16 AMP, `torch.compile`, FlashAttention, Triton kernels, DDP, and FSDP from a
  single code path.
- **Kernel engineering** — readable PyTorch reference code paired with Triton
  replacements for RMSNorm, SwiGLU, cross-entropy, and AdamW.
- **Distributed systems** — custom DDP overlaps bucketed all-reduce with
  backprop; custom FSDP/ZeRO-3 shards parameters, gradients, and optimizer state
  with explicit all-gather / reduce-scatter scheduling.
- **Inference serving** — a continuous-batching engine with per-request KV
  caches, iteration-level scheduling, PagedAttention-style block-paged KV storage 
  and custom Triton kernels cover variable-length prefill, flash decoding, and paged decoding.

## Repository layout

```
LLM-systems-from-scratch/
├── train.py                     # Training entry-point (flags toggle every system feature)
├── generate.py                  # Interactive continuous-batching demo (concurrent prompts)
├── configures/sample.yaml       # Model / optimizer / data config
├── tokenizer/BPETokenizer.py    # Byte-pair encoding tokenizer (train + encode/decode)
├── llm-core/                    # Modeling code — pure PyTorch reference implementation
│   └── llm_core/
│       ├── model.py             # TransformerLM: RoPE, RMSNorm, SwiGLU, causal MHA, KV cache, varlen forward
│       ├── engine.py            # Continuous-batching inference engine (iteration-level scheduler)
│       ├── paged_kv.py          # PagedAttention block pool + paged KV cache
│       ├── optimizer.py         # AdamW + cosine LR schedule with warmup
│       ├── nn_functional.py     # softmax, cross-entropy, gradient clipping
│       └── dataloader.py        # memmap-backed batch sampler with pinned-memory copy
└── llm-systems/                 # Systems code — accelerators and parallelism
    └── llm_systems/
        ├── kernels/             # Triton kernels
        │   ├── triton_adamw.py          # Fused AdamW optimizer update
        │   ├── triton_cross_entropy.py  # Fused logsumexp loss + in-place grad (online softmax)
        │   ├── triton_rms_norm.py       # Fused RMSNorm fwd/bwd
        │   ├── triton_swiglu.py         # Fused SiLU * up-proj
        │   └── triton_flash_attention.py # Varlen prefill, flash-decoding, and paged decode kernels
        └── parallelism/
            ├── ddp.py                   # Bucketed DDP with async all-reduce
            └── fsdp_zero3.py            # ZeRO-3 FSDP with prefetching + comm/compute overlap
```

## Training stack (`train.py`)

Every system feature is a CLI flag — turn one on at a time to isolate its effect:

| Flag | What it does |
|---|---|
| `--amp` | BF16 autocast around the forward/loss; weights stay fp32 |
| `--compile` | `torch.compile(model)` for fused, graph-captured execution |
| `--flash-attn` | Route attention through PyTorch SDPA (FlashAttention-2 on Ampere/Hopper/Blackwell) |
| `--custom-triton` | Swap RMSNorm / SwiGLU / cross-entropy / AdamW for in-tree Triton kernels |
| `--parallel ddp` | Custom bucketed DDP across `--world-size` GPUs |
| `--parallel fsdp` | Custom ZeRO-3 FSDP across `--world-size` GPUs |
| `--world-size N` | Number of GPUs for distributed training |
| `--config path` | Path to YAML config (default `configures/sample.yaml`) |

## Training benchmark ([bench_train.py](bench_train.py))

Usage:

```bash
uv run python bench_train.py --config configures/gpt3xl.yaml --amp --compile-baseline --batches 1 2 4 8 16 --warmup 10 --iters 100
```

End-to-end GPT-3 XL training-step benchmark using
[`configures/gpt3xl.yaml`](configures/gpt3xl.yaml).

Benchmark settings: NVIDIA A100-SXM4-80GB, FlashAttention enabled, BF16 AMP
enabled with `--amp`, and compiled baseline enabled with `--compile-baseline`.

Reduced % is the wall-time reduction from Triton relative to the baseline or
compiled baseline. Memory reduced compares Triton memory to the baseline.

| batch | baseline ms | baseline GB | compiled ms | compiled GB | triton ms | triton GB | reduced vs baseline | reduced vs compiled | memory reduced |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 314.0 | 28.02 | 268.0 | 25.43 | 198.9 | 24.94 | 36.7% | 25.8% | 11.0% |
| 2 | 458.8 | 36.08 | 367.3 | 30.95 | 318.1 | 29.43 | 30.7% | 13.4% | 18.5% |
| 4 | 727.3 | 52.08 | 559.0 | 41.96 | 551.3 | 38.35 | 24.2% | 1.4% | 26.4% |
| 8 | OOM | - | 958.7 | 64.06 | 1020.1 | 56.26 | - | -6.4% | - |
| 16 | OOM | - | OOM | - | OOM | - | - | - | - |

Overall reduced wall-time: 28.8% vs baseline on non-OOM baseline batches, 3.0%
vs compiled on non-OOM compiled batches.

Overall reduced memory: 20.2% vs baseline on non-OOM baseline batches.

## Custom Triton kernels (`--custom-triton`)

`--custom-triton` swaps the memory-bandwidth-bound pieces of the training step
for in-tree Triton kernels. The goal isn't to beat every optimized PyTorch
backend, but to make the tradeoffs explicit: which tensors are read, which
buffers are written, and where launches or intermediate activations can be cut.

### Fused cross-entropy ([triton_cross_entropy.py](llm-systems/llm_systems/kernels/triton_cross_entropy.py))

Uses online softmax (Milakov & Gimelshein, 2018) to compute logsumexp without
materializing the full `[B*T, V]` probability matrix. The forward kernel writes
the mean-reduced gradient in-place into the logits buffer; backward simply
returns it, scaling by `grad_output` only when needed.

#### Benchmark

Reduced % is the wall-time reduction from Triton relative to the reference or
Torch fused cross-entropy path. Mem reduced is the activation memory reduction
relative to the Torch fused path.

| V | triton ms | ref ms | torchF ms | reduced vs ref | reduced vs torchF | triton MB | torchF MB | mem reduced |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.650 | 3.017 | 0.831 | 78.5% | 21.8% | 327.8 | 819.2 | 60.0% |
| 32,000 | 1.576 | 9.173 | 3.350 | 82.8% | 53.0% | 1048.7 | 2621.5 | 60.0% |
| 50,257 | 2.515 | 14.344 | 5.397 | 82.5% | 53.4% | 1648.4 | 4120.9 | 60.0% |
| 128,000 | 6.394 | 36.258 | 13.496 | 82.4% | 52.6% | 4194.4 | 10485.8 | 60.0% |

Overall reduced wall-time: 82.3% vs ref, 51.7% vs Torch fused.

Overall reduced memory: 60.0% vs Torch fused.

### Fused AdamW ([triton_adamw.py](llm-systems/llm_systems/kernels/triton_adamw.py))

Does the entire update — both moments, the parameter step, and decoupled weight
decay — in a single launch instead of separate elementwise passes. Moment
buffers stay in fp32, the bias-corrected step size is precomputed on the host,
and `lr` is a runtime scalar so the cosine schedule never triggers
recompilation. Ships as a `torch.optim.Optimizer` subclass, so it drops into the
training loop unchanged.

#### Benchmark

Reduced % is the wall-time reduction relative to the reference AdamW update.

| N | fused ms | fused GB/s | ref ms | ref GB/s | reduced % |
|---:|---:|---:|---:|---:|---:|
| 65,536 | 0.011 | 160.3 | 0.109 | 16.9 | 89.5% |
| 262,144 | 0.012 | 622.7 | 0.054 | 136.4 | 78.1% |
| 1,048,576 | 0.026 | 1132.6 | 0.084 | 348.1 | 69.2% |
| 4,194,304 | 0.078 | 1508.5 | 0.324 | 362.4 | 76.0% |
| 16,777,216 | 0.278 | 1688.0 | 1.324 | 354.7 | 79.0% |

Overall reduced %: 78.6%

### Fused RMSNorm ([triton_rms_norm.py](llm-systems/llm_systems/kernels/triton_rms_norm.py))

Forward and backward as row-wise kernels. The reduction runs in fp32 for
numerical stability, the reciprocal RMS is cached for backward, and partial
weight gradients accumulate across programs before a final reduction in PyTorch.

#### Benchmark

Reduced % is the wall-time reduction from Triton relative to the reference or
Torch fused RMSNorm path. Memory reduced compares Triton memory to Torch fused;
negative values mean Triton used slightly more memory.

| d | triton ms | ref ms | torchF ms | reduced vs ref | reduced vs torchF | triton MB | torchF MB | memory reduced |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.739 | 1.325 | 0.330 | 44.2% | -123.9% | 269.0 | 268.5 | -0.2% |
| 2,048 | 0.692 | 2.499 | 0.701 | 72.3% | 1.3% | 537.9 | 537.0 | -0.2% |
| 4,096 | 0.836 | 4.862 | 1.569 | 82.8% | 46.7% | 1075.7 | 1073.9 | -0.2% |
| 8,192 | 1.587 | 9.535 | 3.086 | 83.4% | 48.6% | 2151.3 | 2147.8 | -0.2% |

Overall reduced wall-time: 78.9% vs ref, 32.2% vs Torch fused.

Overall reduced memory: -0.2% vs Torch fused.

### Fused SwiGLU ([triton_swiglu.py](llm-systems/llm_systems/kernels/triton_swiglu.py))

Merges the gate and up projections into one `Linear(d_model, 2 * d_ff)`, then
fuses `silu(gate) * up` into a custom autograd function. The down projection
stays a plain linear layer, leaving room for future epilogue fusion.

#### Benchmark

Reduced % is the wall-time reduction from Triton relative to the reference
SwiGLU path. Memory reduced compares Triton memory to the reference path.

| d_ff | triton ms | ref ms | reduced vs ref | triton MB | ref MB | memory reduced |
|---:|---:|---:|---:|---:|---:|---:|
| 1,344 | 0.622 | 0.720 | 13.6% | 528.5 | 616.6 | 14.3% |
| 2,048 | 0.935 | 1.081 | 13.5% | 805.3 | 939.5 | 14.3% |
| 4,096 | 1.842 | 2.127 | 13.4% | 1610.6 | 1879.0 | 14.3% |
| 8,192 | 3.673 | 4.223 | 13.0% | 3221.2 | 3758.1 | 14.3% |
| 11,008 | 4.939 | 5.663 | 12.8% | 4328.5 | 5049.9 | 14.3% |

Overall reduced wall-time: 13.1% vs ref.

Overall reduced memory: 14.3% vs ref.

## Custom DDP (`--parallel ddp`)

[`ddp.py`](llm-systems/llm_systems/parallelism/ddp.py) is a compact DDP wrapper
covering the mechanics that matter most: a rank-0 broadcast at init, gradient
bucketing, async all-reduce, and explicit sync before the optimizer step.

- Each rank holds a full model copy; weights broadcast from rank 0 at init so all ranks start identical.
- Parameters are iterated in **reverse order** so buckets fill in backward-pass order, maximizing comm/compute overlap.
- Gradients accumulate into fixed-size buckets (default 25 MB); when one fills, a `post_accumulate_grad_hook` flattens it and launches an async `all_reduce`.
- `finish_gradient_synchronization()` joins all in-flight handles and unflattens the averaged grads back into `param.grad`.

#### Benchmark

Environment: 8x A100 SXM4 80GB, GPT-3 XL config
([configures/gpt3xl.yaml](configures/gpt3xl.yaml)).

Efficiency is `TPS(N) / (N * TPS(1))`, using the 1-GPU run as the baseline.

**DDP Eager**

| GPUs | global batch | step ms | global tok/s | efficiency | peak GB |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 743.4 | 11,020 | 100.0% | 51.95 |
| 2 | 8 | 773.0 | 21,194 | 96.2% | 51.93 |
| 4 | 16 | 781.7 | 41,919 | 95.1% | 51.93 |
| 8 | 32 | 789.6 | 82,995 | 94.1% | 51.93 |

**DDP Triton**

| GPUs | global batch | step ms | global tok/s | efficiency | peak GB |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 564.2 | 14,520 | 100.0% | 38.30 |
| 2 | 8 | 593.7 | 27,596 | 95.0% | 38.30 |
| 4 | 16 | 599.5 | 54,662 | 94.1% | 38.30 |
| 8 | 32 | 605.7 | 108,192 | 93.1% | 38.30 |

## Custom FSDP / ZeRO-3 (`--parallel fsdp`)

[`fsdp_zero3.py`](llm-systems/llm_systems/parallelism/fsdp_zero3.py) is the most involved piece: full sharding of **parameters, gradients, and optimizer state**, with eager prefetching and a dedicated CUDA stream for collectives.

**Per-unit design (`FSDPUnit`).** Each transformer layer is one FSDP unit; embeddings + final norm + lm_head form one more. Every unit:

1. Flattens its parameters into one contiguous buffer, broadcasts rank 0's init, then keeps **only its 1/world_size shard** (`flat_shard`).
2. `all_gather_params` — allocates a transient `flat_full` buffer, runs `dist.all_gather` (async by default), then re-points the module's parameters as views into it.
3. `reduce_scatter_grads` — averages and reduce-scatters the unsharded gradient into a per-rank shard on the dedicated `comm_stream`, synchronized via CUDA events.
4. `discard_full_params` — frees the unsharded buffer as soon as the layer's compute is done.

**The hard part: backward scheduling.** Two custom `torch.autograd.Function`s are spliced into the forward graph so that, during backward, params are gathered just before they're needed and grads scattered the instant they're done:

```
Forward:  embedding → Hook_0 → layer_0 → ... → Hook_N → layer_N → EndHook → final_norm → lm_head
Backward: lm_head.bw → final_norm.bw → EndHook.bw → layer_N.bw → Hook_N.bw → ... → Hook_0.bw → embedding.bw
```

- `EndHook.bw` re-gathers layer N's params and prefetches layer N-1.
- `Hook_i.bw` runs **after** `layer_i.bw`: it reduce-scatters layer i's grads, makes sure layer i-1 is gathered, and prefetches layer i-2.
- Embedding/lm_head grads are reduce-scattered explicitly in `finish_gradient_synchronization()`, since no differentiable hook can sit before the integer token IDs.

**Checkpointing under FSDP.** At save time, every unit does a synchronous all-gather, rank 0 writes the full state dict, then each unit discards the unsharded buffer.

**Known limitation.** `--parallel fsdp` + `--compile` is explicitly blocked: the FSDP unit swaps `param.data` on every forward, invalidating `torch.compile`'s dynamo guards. Fixing this requires a persistent `flat_full` buffer per unit so param `data_ptr` stays stable.

#### Benchmark

Environment: 8x A100 SXM4 80GB, GPT-3 XL config
([configures/gpt3xl.yaml](configures/gpt3xl.yaml)).

Efficiency is `TPS(N) / (N * TPS(1))`, using the 1-GPU run as the baseline.

**FSDP Eager**

| GPUs | global batch | step ms | global tok/s | efficiency | peak GB |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 744.1 | 11,009 | 100.0% | 51.95 |
| 2 | 8 | 734.4 | 22,308 | 101.3% | 51.88 |
| 4 | 16 | 709.5 | 46,184 | 104.9% | 46.21 |
| 8 | 32 | 697.8 | 93,918 | 106.6% | 43.36 |

**FSDP Triton**

| GPUs | global batch | step ms | global tok/s | efficiency | peak GB |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 565.0 | 14,498 | 100.0% | 38.30 |
| 2 | 8 | 602.9 | 27,176 | 93.7% | 38.49 |
| 4 | 16 | 600.0 | 54,613 | 94.2% | 32.80 |
| 8 | 32 | 598.3 | 109,537 | 94.4% | 29.94 |

## Continuous batching ([engine.py](llm-core/llm_core/engine.py))

An iteration-level inference scheduler (Orca / vLLM style) built on the model's
KV cache and variable-length forward path. Static batching makes a whole batch
wait for its longest sequence; continuous batching schedules at the granularity
of a single decode step, so a finished request frees its slot immediately and a
queued request joins mid-flight — keeping the batch full instead of draining
between waves.

- `Request` tracks prompt tokens, generated tokens, per-layer `KVCache`, and
  `SamplingParams`.
- `TransformerLM.forward_varlen` packs mixed prefill and decode tokens into one
  ragged batch with `cu_seqlens`, avoiding padding while each sequence attends
  over its own cache.
- `LLMEngine.step()` admits requests, runs one mixed forward pass, samples one
  token per active request, and retires completed requests.

`add_request()` / `step()` are the public API. [generate.py](generate.py) is an
interactive demo for concurrent prompts, and [bench_engine.py](bench_engine.py)
compares continuous vs static admission on the same engine and kernels.

### Paged KV cache (`--paged`)

By default, each request owns a contiguous per-layer `KVCache`. With `--paged`,
the engine uses a PagedAttention-style block allocator
([paged_kv.py](llm-core/llm_core/paged_kv.py)): each layer has a pre-allocated
pool of fixed-size KV blocks, and each request keeps a *block table* mapping
logical token positions to physical blocks. Requests grow by taking blocks from
the shared free list, avoiding contiguous per-request reservations and most
external fragmentation.

| Flag | What it does |
|---|---|
| `--paged` | Enable the paged KV cache (off by default) |
| `--block-size N` | Tokens per KV block (default: 16) |
| `--num-blocks N` | Pre-allocated blocks per layer; this sets the pool capacity |

When a request is admitted, the engine estimates its worst-case block budget
(`prompt + max_tokens`) and admits it only if that budget fits in the remaining
pool capacity. The actual KV blocks are still allocated on demand as the
request's cache grows. On CUDA, `paged_decode` reads keys and values directly
from the paged pool through the request's block table. That avoids the
contiguous-cache path's per-step copy, where active request caches are
concatenated into a flat buffer before decoding. Both [generate.py](generate.py)
and [bench_engine.py](bench_engine.py) accept `--paged --block-size
--num-blocks`.

#### Benchmark

Environment: A100 PCIe 80GB, GPT-3 XL
([configures/gpt3xl.yaml](configures/gpt3xl.yaml)), BF16, and 512 requests with
prompt lengths from 4–32 tokens and output lengths from 8–95 tokens. Each row is
one [bench_engine.py](bench_engine.py) run. `speedup` is continuous-batching
throughput divided by static-batching throughput; paged KV and Triton flash
attention are toggled with `--paged` and `--no-flash-attn`.

| attention | KV cache | batch | static tok/s | continuous tok/s | speedup |
|---|---|---:|---:|---:|---:|
| eager | contiguous | 64 | 131.2 | 137.1 | 1.05x |
| Triton flash | contiguous | 64 | 548.8 | 693.7 | 1.26x |
| Triton flash | paged | 64 | 755.0 | 1044.6 | 1.38x |
| Triton flash | paged | 8 | 108.7 | 174.8 | 1.61x |

The main comparisons:

- **Continuous vs static** — iteration-level scheduling keeps the batch from
  draining between waves, giving 1.05–1.61x higher throughput in these runs. For
  this request mix, the fewer-steps factor is the speedup ceiling: about 1.62x at
  batch 64 and 1.67x at batch 8. Smaller batches get closer to that ceiling;
  larger batches spend more work per step on attention/KV traffic and, in the
  contiguous path, cache concatenation.
- **Triton flash vs eager** — with contiguous KV caches at batch 64, the custom
  variable-length prefill and flash-decoding kernels raise continuous throughput
  from 137.1 to 693.7 tok/s (5.1x). Static throughput rises from 131.2 to 548.8
  tok/s (4.2x), since the eager path handles sequences one at a time.
- **Paged vs contiguous** — with Triton flash attention at batch 64, paged KV
  raises continuous throughput from 693.7 to 1044.6 tok/s (1.5x) by reading KV
  directly from the block pool through request block tables, instead of
  flattening active caches before every decode step.
- **End-to-end** — the full serving stack (Triton flash + paged KV + continuous
  batching) reaches 1044.6 tok/s, compared with 131.2 tok/s for the baseline
  eager + contiguous + static path at batch 64: about **8x** faster.

#### Memory footprint ([bench_memory.py](bench_memory.py))

Environment: A100 40GB PCIe, GPT-3 XL, BF16, `batch` identical full-context
requests (16-token prompt + 2048 generated, depth 2064), block size 16, run
concurrently to completion. The paged pool is sized to exactly cover the
workload, so both backends hold the same KV; `resv` is the allocator's VRAM
footprint (what triggers OOM), `alloc` is the live KV.

| batch | contiguous resv / alloc GB | paged resv / alloc GB | paged peak |
|---:|---:|---:|---|
| 32 | 24.82 / 16.31 | 16.44 / 16.14 | 34% lower |
| 48 | 39.19 / 23.02 | 22.76 / 22.53 | 42% lower |
| 64 | 39.32 / 29.73 | 29.25 / 29.01 | 26% lower |
| 72 | 39.33 / 33.09 | 32.59 / 32.27 | 17% lower |
| 80 | 39.33 / 36.44 | 35.89 / 35.59 | 9% lower |
| 88 | OOM | 39.13 / 38.83 | contiguous OOM |
| 96 | OOM | OOM | — |

The contiguous cache's reserved memory pins near the card's ceiling (~39.3 GB)
from batch 48 on, while its live KV (`alloc`) stays far lower — at batch 48 the
gap is 16 GB, lost to `torch.cat` regrowth and the per-step concat of every
active cache. Paging holds the same KV with `resv ≈ alloc`, since the pool is
allocated once and written in place. That headroom is the difference between OOM
and serving: contiguous runs out at batch 88, paging still fits it (39.1 GB), so
the same 40 GB card sustains ~10% more concurrent full-context sequences and 9–42%
lower peak at matched batch. (The contiguous cache already grows lazily, so this
gap is fragmentation plus transients, not over-reservation; against a baseline
that pre-allocates to max context the gap would be wider.)

#### Custom paged-decode kernel vs FlashInfer / flash-attn ([bench_paged_decode.py](bench_paged_decode.py))

This microbenchmark isolates one single-token decode step over a paged KV cache:
`batch` sequences, each with `KV len` cached tokens, 16 attention heads, and
`d_head=128`. It compares this repo's hand-written `paged_decode` Triton kernel
([triton_flash_attention.py](llm-systems/llm_systems/kernels/triton_flash_attention.py))
with FlashInfer and flash-attn paged-decode kernels.

Environment: A100 40GB PCIe, BF16, block size 256. Layout conversion is done
before timing for every backend, and FlashInfer's `plan()` is also hoisted out
of the timed region, matching a serving setup where KV is already stored in the
backend's native layout and the decode plan is reused. Outputs match the library
kernels within BF16 tolerance (max absolute difference around `1e-3`). The ratio
columns report our throughput relative to each library; values above 1 mean our
kernel is faster. GB/s is the estimated KV read bandwidth, with percent of the
A100's 1555 GB/s peak in parentheses.

| batch | KV len | ours GB/s (% peak) | ours ÷ FlashInfer | ours ÷ flash-attn |
|---:|---:|---:|---:|---:|
| 1 | 256 | 24 (2%) | 0.36× | 0.33× |
| 64 | 256 | 1241 (80%) | 1.02× | 1.03× |
| 256 | 256 | 1335 (86%) | 0.96× | 0.98× |
| 1 | 1024 | 96 (6%) | 0.42× | 0.34× |
| 64 | 1024 | 1336 (86%) | 0.94× | 0.96× |
| 256 | 1024 | 1391 (89%) | 0.94× | 0.97× |
| 1 | 4096 | 345 (22%) | 0.38× | 0.34× |
| 64 | 4096 | 1327 (85%) | 0.91× | 0.91× |
| 256 | 4096 | 1350 (87%) | 0.90× | 0.92× |

At serving batch sizes (`batch >= 64`), the custom kernel reaches 80-89% of peak
HBM bandwidth and stays within roughly 10% of both libraries. Decode is
memory-bound at these shapes, so achieved bandwidth is the relevant comparison.
The gap is much larger at batch 1: one program per `(sequence, head)` leaves too
little parallelism to fill the GPU, while FlashInfer and flash-attn split work
across the KV length and run about 2.5-3x
faster.

## Quickstart

### Install

```bash
# uv (recommended)
uv sync

# or pip
pip install -e ./llm-core -e ./llm-systems -e .
```

Requires Python ≥ 3.11. On Linux, `torch >= 2.7` and `triton >= 3.3` are pulled in for Blackwell support; on macOS x86_64 the project falls back to `torch ~= 2.2.2` (no Triton, no FSDP).

### Prepare data

Tokenize your corpus with the BPE tokenizer and dump `train.bin` / `val.bin` as `uint32` token streams. Point `configures/sample.yaml` at them.

### Train

```bash
# Single-GPU, FP32
python train.py --config configures/sample.yaml

# Single-GPU, BF16 + compiled + FlashAttention
python train.py --amp --compile --flash-attn

# Single-GPU with custom Triton kernels swapped in
python train.py --amp --custom-triton

# 4-GPU DDP
python train.py --parallel ddp --world-size 4 --amp --flash-attn

# 4-GPU ZeRO-3 FSDP (note: --compile is currently incompatible)
python train.py --parallel fsdp --world-size 4 --amp --flash-attn
```

### Generate

`generate.py` loads a checkpoint into the continuous-batching engine. It enables
FlashAttention (PyTorch SDPA) by default for prefill and cached decode; pass
`--no-flash-attn` to fall back to the reference attention path.

```bash
python generate.py \
  --config checkpoints/sample/model_config.json \
  --checkpoint checkpoints/sample/ckpt_final.pt \
  --vocab vocab.json --merges merges.json \
  --max-tokens 200 --temperature 0.8 --top-k 40
```

Enter one prompt per line, then submit the batch with a blank line. Prompts in
the same batch generate concurrently; use `Ctrl+C` or `Ctrl+D` to exit.

Add `--paged` to serve with the block-paged KV cache, tuning the pool with
`--block-size` / `--num-blocks`:

```bash
python generate.py \
  --config checkpoints/sample/model_config.json \
  --checkpoint checkpoints/sample/ckpt_final.pt \
  --vocab vocab.json --merges merges.json \
  --max-tokens 200 --temperature 0.8 --top-k 40 \
  --paged --block-size 16 --num-blocks 2048
```

## Roadmap

- Persistent `flat_full` buffer in `FSDPUnit` so `--parallel fsdp` + `--compile` work together
- Tensor / pipeline parallelism
- Gradient accumulation
- Activation checkpointing
- Prefill-Decode Disaggregation
