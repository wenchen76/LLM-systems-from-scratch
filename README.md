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
  caches, iteration-level scheduling, and selective (variable-length) batching
  that mixes prefill and decode in a single forward.

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
│       ├── optimizer.py         # AdamW + cosine LR schedule with warmup
│       ├── nn_functional.py     # softmax, cross-entropy, gradient clipping
│       └── dataloader.py        # memmap-backed batch sampler with pinned-memory copy
└── llm-systems/                 # Systems code — accelerators and parallelism
    └── llm_systems/
        ├── kernels/             # Triton kernels
        │   ├── triton_adamw.py          # Fused AdamW optimizer update
        │   ├── triton_cross_entropy.py  # Fused logsumexp loss + in-place grad (online softmax)
        │   ├── triton_rms_norm.py       # Fused RMSNorm fwd/bwd
        │   └── triton_swiglu.py         # Fused SiLU * up-proj
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

## Roadmap

- Persistent `flat_full` buffer in `FSDPUnit` so `--parallel fsdp` + `--compile` work together
- Tensor / pipeline parallelism
- Gradient accumulation
- Activation checkpointing
- Prefill-Decode Disaggregation
