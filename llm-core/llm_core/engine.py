"""Continuous-batching inference engine (Phase 4).

Schedules generation at iteration granularity (Orca / vLLM style): each step()
admits waiting requests, runs ONE mixed prefill+decode forward over all running
requests — each with its own KV cache — samples one token per request, and
retires finished ones immediately so new requests can take their slots. This
keeps the batch full instead of waiting for the longest sequence the way static
batching does.

Built on TransformerLM.forward_varlen with per-request KV caches (see
Request.kv_caches): token-wise layers run on the whole flat batch at once, only
attention is done per request.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import torch

from llm_core.model import KVCache, TransformerLM


@dataclass
class SamplingParams:
    """Per-request sampling configuration."""

    max_tokens: int = 64
    temperature: float = 1.0
    top_k: int | None = None
    eos_token_id: int | None = None


class RequestState(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    """A single generation request and its runtime state."""

    request_id: int
    prompt_ids: list[int]
    sampling: SamplingParams
    output_ids: list[int] = field(default_factory=list)
    kv_caches: list[KVCache] | None = None  # per-layer; allocated at admission
    state: RequestState = RequestState.WAITING
    prefilled: bool = False
    finish_reason: str | None = None

    @property
    def num_cached(self) -> int:
        """Tokens currently in this request's KV cache (its next absolute position)."""
        return 0 if self.kv_caches is None else self.kv_caches[0].length


class LLMEngine:
    """Iteration-level scheduler for continuous batching."""

    def __init__(self, model: TransformerLM, device: str = "cpu", max_running: int = 64):
        self.model = model.eval()
        self.device = device
        self.context_length = model.context_length
        self.max_running = max_running
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self._next_id = 0

    # ---- public API ---------------------------------------------------------
    def add_request(self, prompt_ids, sampling: SamplingParams | None = None) -> int:
        """Enqueue a request. It is admitted on a later step() when a slot is free."""
        prompt = list(prompt_ids)[-self.context_length:]  # fit context window
        req = Request(self._next_id, prompt, sampling or SamplingParams())
        self.waiting.append(req)
        self._next_id += 1
        return req.request_id

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    @torch.no_grad()
    def step(self) -> list[Request]:
        """Run one scheduler iteration; return the requests that finished this step."""
        self._admit()
        if not self.running:
            return []

        flat, cu_seqlens, position_ids, caches, last_idx = self._build_batch()
        logits = self.model.forward_varlen(flat, cu_seqlens, position_ids, request_kv_caches=caches)
        next_ids = self._sample(logits[last_idx])  # one token per running request

        finished = []
        for req, token in zip(self.running, next_ids.tolist()):
            req.prefilled = True
            req.output_ids.append(token)
            req.finish_reason = self._finish_reason(req, token)
            if req.finish_reason is not None:
                req.state = RequestState.FINISHED
                finished.append(req)

        self.running = [r for r in self.running if r.state is RequestState.RUNNING]
        return finished

    def generate(self, prompts, sampling: SamplingParams | None = None) -> list[list[int]]:
        """Convenience: enqueue all prompts, run to completion, return outputs in order."""
        ids = [self.add_request(p, sampling) for p in prompts]
        done: dict[int, list[int]] = {}
        while self.has_work():
            for req in self.step():
                done[req.request_id] = req.output_ids
        return [done[i] for i in ids]

    # ---- internals ----------------------------------------------------------
    def _admit(self) -> None:
        """Move waiting requests into the running set, allocating their KV caches."""
        while self.waiting and len(self.running) < self.max_running:
            req = self.waiting.popleft()
            req.kv_caches = self.model.new_kv_cache()
            req.state = RequestState.RUNNING
            self.running.append(req)

    def _build_batch(self):
        """Flatten each running request's new tokens into one varlen batch.

        A not-yet-prefilled request contributes its whole prompt; a decoding one
        contributes just its last token. Each token gets its absolute position.
        """
        flat: list[int] = []
        positions: list[int] = []
        cu: list[int] = [0]
        last_idx: list[int] = []
        for req in self.running:
            if not req.prefilled:
                tokens, start = req.prompt_ids, 0
            else:
                tokens, start = req.output_ids[-1:], req.num_cached
            flat.extend(tokens)
            positions.extend(range(start, start + len(tokens)))
            cu.append(cu[-1] + len(tokens))
            last_idx.append(cu[-1] - 1)  # index of this request's last token
        to_t = lambda xs: torch.tensor(xs, dtype=torch.long, device=self.device)  # noqa: E731
        return (
            to_t(flat),
            to_t(cu),
            to_t(positions),
            [req.kv_caches for req in self.running],
            to_t(last_idx),
        )

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Per-request sampling over (R, vocab) logits aligned with self.running.

        Fully vectorized: temperature divides per row, and top-k (which may
        differ per request) masks each row below its own k-th largest logit via
        one batched topk + a per-row gather of the threshold — no Python loop —
        then one multinomial draw covers the whole batch.
        """
        device = logits.device
        temps = torch.tensor([r.sampling.temperature for r in self.running], device=device)
        scaled = logits / temps.unsqueeze(-1)

        ks = [r.sampling.top_k for r in self.running]
        if any(ks):
            vocab = scaled.size(-1)
            k_per_row = torch.tensor([min(k, vocab) if k else 0 for k in ks], device=device)
            max_k = int(k_per_row.max())
            kth = torch.topk(scaled, max_k, dim=-1).values.gather(  # each row's k-th largest logit
                1, (k_per_row - 1).clamp(min=0).unsqueeze(1)
            )
            kth = kth.masked_fill((k_per_row == 0).unsqueeze(1), float("-inf"))  # rows without top-k keep all
            scaled = scaled.masked_fill(scaled < kth, float("-inf"))

        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)  # (R,)

    def _finish_reason(self, req: Request, token: int) -> str | None:
        sp = req.sampling
        if sp.eos_token_id is not None and token == sp.eos_token_id:
            return "eos"
        if len(req.output_ids) >= sp.max_tokens:
            return "length"
        if req.num_cached >= self.context_length:
            return "context_full"  # no room left for the next absolute position
        return None
