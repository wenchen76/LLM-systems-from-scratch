"""CUDA-graph capture of the loop-free decode forward (model.forward_decode).

A decode step launches dozens of small kernels and runs Python scheduling between
them, so the GPU sits ~40% idle waiting for the host (see nvidia-smi during a
decode-heavy run). A CUDA graph records that kernel sequence once and replays it
with a single launch, removing the per-step launch + Python overhead.

Capture requires fixed shapes and fixed addresses: one graph per bucket batch
size, reading from static input buffers that the engine fills in place each step
(growing block tables on the host before replay, since allocation can't happen
inside a captured graph). This is the GPU-only Piece 2 of the static-decode work;
it is exercised through the engine's cuda_graph path.
"""
from __future__ import annotations

import torch


class DecodeGraphs:
    """Captures model.forward_decode once per bucket batch size and replays it.

    scratch holds one reserved block per layer: warmup/capture scatters the
    (garbage) KV there so it can never clobber a real request's block. Only run()
    is on the hot path; _capture happens lazily the first time a bucket is seen.
    """

    def __init__(self, engine, max_blocks: int):
        self.e = engine
        self.max_blocks = max_blocks
        self.graphs: dict[int, dict] = {}
        self.scratch = [pool.allocate() for pool in engine.pools]  # capture-time KV sink

    def run(self, batch):
        """Fill the static buffers from `batch`, replay the graph, return logits (B, vocab)."""
        B = len(batch)
        if B not in self.graphs:
            self.graphs[B] = self._capture(B)
        state = self.graphs[B]
        self._fill(batch, state)
        state["graph"].replay()
        return state["logits"]

    # ---- capture -------------------------------------------------------------
    def _alloc_static(self, B: int) -> dict:
        dev = self.e.device
        n_layers = len(self.e.pools)
        return dict(
            input_ids=torch.zeros(B, dtype=torch.long, device=dev),
            positions=torch.zeros(B, dtype=torch.long, device=dev),
            seq_lens=torch.ones(B, dtype=torch.int32, device=dev),
            q_positions=torch.arange(B, dtype=torch.int32, device=dev),  # constant per bucket
            block_tables=[torch.zeros(B, self.max_blocks, dtype=torch.int32, device=dev) for _ in range(n_layers)],
            phys=[torch.full((B,), self.scratch[i], dtype=torch.long, device=dev) for i in range(n_layers)],
            offset=[torch.zeros(B, dtype=torch.long, device=dev) for _ in range(n_layers)],
        )

    def _forward(self, s: dict):
        return self.e.model.forward_decode(
            s["input_ids"], s["positions"], self.e.pools,
            s["block_tables"], s["phys"], s["offset"], s["seq_lens"], s["q_positions"],
        )

    def _capture(self, B: int) -> dict:
        s = self._alloc_static(B)
        # Warm up on a side stream so the kernels (paged_decode autotune, cuBLAS) are
        # compiled/selected before capture — capture can't tolerate those one-off launches.
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                self._forward(s)
        torch.cuda.current_stream().wait_stream(warm)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            s["logits"] = self._forward(s)
        s["graph"] = graph
        return s

    # ---- per-step fill (host side, before replay) ----------------------------
    def _fill(self, batch, s: dict):
        bs = self.e.block_size
        max_blocks = s["block_tables"][0].shape[1]
        lengths = [r.kv_caches[0].length for r in batch]  # same across layers
        s["input_ids"].copy_(torch.tensor([r.output_ids[-1] for r in batch]))
        s["positions"].copy_(torch.tensor(lengths))
        s["seq_lens"].copy_(torch.tensor([L + 1 for L in lengths], dtype=torch.int32))
        for layer, pool in enumerate(self.e.pools):
            rows, phys_vals, offset_vals = [], [], []
            for req in batch:
                cache = req.kv_caches[layer]
                L = cache.length
                if L // bs >= len(cache.block_table):  # grow on the host before replay
                    cache.block_table.append(pool.allocate())
                table = cache.block_table
                rows.append(table + [0] * (max_blocks - len(table)))  # pad to the fixed width
                phys_vals.append(table[L // bs])
                offset_vals.append(L % bs)
            # one host->device copy per layer (not one per request) -- the metadata fill, not the
            # graph replay, was the cost that made large batches slow.
            s["block_tables"][layer].copy_(torch.tensor(rows, dtype=torch.int32))
            s["phys"][layer].copy_(torch.tensor(phys_vals))
            s["offset"][layer].copy_(torch.tensor(offset_vals))
