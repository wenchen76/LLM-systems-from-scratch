# LLM Systems from Scratch

A decoder-only Transformer language model built from scratch — plus the systems infrastructure that trains it efficiently: custom Triton kernels, custom FSDP (ZeRO-3) and DDP wrappers, a BPE tokenizer, and a training loop wired up with mixed precision, `torch.compile`, FlashAttention, and Weights & Biases.

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

## Repository layout

```
LLM-systems-from-scratch/
├── train.py                     # Training entry-point (flags toggle every system feature)
├── generate.py                  # Interactive REPL for sampling from a checkpoint
├── configures/sample.yaml       # Model / optimizer / data config
├── tokenizer/BPETokenizer.py    # Byte-pair encoding tokenizer (train + encode/decode)
├── llm-core/                    # Modeling code — pure PyTorch reference implementation
│   └── llm_core/
│       ├── model.py             # TransformerLM: RoPE, RMSNorm, SwiGLU, causal MHA
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

## Custom Triton kernels (`--custom-triton`)

`--custom-triton` swaps the memory-bandwidth-bound pieces of the training step
for in-tree Triton kernels. The goal isn't to beat every optimized PyTorch
backend, but to make the tradeoffs explicit: which tensors are read, which
buffers are written, and where launches or intermediate activations can be cut.

### Fused AdamW ([triton_adamw.py](llm-systems/llm_systems/kernels/triton_adamw.py))

Does the entire update — both moments, the parameter step, and decoupled weight
decay — in a single launch instead of separate elementwise passes. Moment
buffers stay in fp32, the bias-corrected step size is precomputed on the host,
and `lr` is a runtime scalar so the cosine schedule never triggers
recompilation. Ships as a `torch.optim.Optimizer` subclass, so it drops into the
training loop unchanged.

### Fused cross-entropy ([triton_cross_entropy.py](llm-systems/llm_systems/kernels/triton_cross_entropy.py))

Uses online softmax (Milakov & Gimelshein, 2018) to compute logsumexp without
materializing the full `[B*T, V]` probability matrix. The forward kernel writes
the mean-reduced gradient in-place into the logits buffer; backward simply
returns it, scaling by `grad_output` only when needed.

### Fused RMSNorm ([triton_rms_norm.py](llm-systems/llm_systems/kernels/triton_rms_norm.py))

Forward and backward as row-wise kernels. The reduction runs in fp32 for
numerical stability, the reciprocal RMS is cached for backward, and partial
weight gradients accumulate across programs before a final reduction in PyTorch.

### Fused SwiGLU ([triton_swiglu.py](llm-systems/llm_systems/kernels/triton_swiglu.py))

Merges the gate and up projections into one `Linear(d_model, 2 * d_ff)`, then
fuses `silu(gate) * up` into a custom autograd function. The down projection
stays a plain linear layer, leaving room for future epilogue fusion.

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

```bash
python generate.py \
  --config checkpoints/sample/model_config.json \
  --checkpoint checkpoints/sample/ckpt_final.pt \
  --vocab vocab.json --merges merges.json \
  --max-tokens 200 --temperature 0.8 --top-k 40
```

You'll get an interactive prompt. `Ctrl+C` to exit.

## Roadmap

- Persistent `flat_full` buffer in `FSDPUnit` so `--parallel fsdp` + `--compile` work together
- Tensor / pipeline parallelism
- Activation checkpointing
- KV-cache and continuous batching for `generate.py`
