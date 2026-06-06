"""Phase 1 guardrail tests for autoregressive generation.

These pin down the *current* (no-KV-cache) behavior so the upcoming KV-cache /
incremental-decode work can be validated token-for-token. Everything runs on
CPU with a tiny seeded model for determinism and speed.

The key invariant `test_causal_consistency` checks is exactly what a correct KV
cache must preserve: the next-token logits produced incrementally must equal the
logits from a full forward over the whole sequence. It holds today (no-cache
decode *is* a full recompute) and must keep holding after the cache lands.
"""
import torch

from llm_core.model import TransformerLM

DEVICE = "cpu"

# Tiny model: big enough to exercise every code path, small enough to be instant.
SMALL_CONFIG = {
    "vocab_size": 256,
    "context_length": 64,
    "d_model": 64,
    "num_layers": 2,
    "num_heads": 4,
    "d_ff": 128,
    "rope_theta": 10000.0,
}


def make_model(seed: int = 0, **overrides) -> TransformerLM:
    config = {**SMALL_CONFIG, **overrides}
    torch.manual_seed(seed)
    model = TransformerLM(**config).to(DEVICE).eval()
    return model


def random_prompt(batch: int, length: int, vocab_size: int, seed: int = 1234) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, vocab_size, (batch, length), generator=gen, dtype=torch.long, device=DEVICE)


@torch.no_grad()
def greedy_reference(model: TransformerLM, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Deterministic greedy decode via full recompute — the ground-truth path."""
    ids = prompt
    for _ in range(max_new_tokens):
        ctx = ids[:, -model.context_length:]
        next_id = model(ctx)[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=-1)
    return ids[:, prompt.size(1):]


@torch.no_grad()
def test_causal_consistency_full_vs_prefix():
    """logits[:, t] from a full forward == last-position logits from forward over prefix[:t+1].

    This is the fundamental property a KV cache must preserve (causality: output at
    position t depends only on positions <= t).
    """
    model = make_model()
    ids = random_prompt(batch=2, length=16, vocab_size=SMALL_CONFIG["vocab_size"])

    full = model(ids)  # (batch, seq, vocab)
    for t in range(ids.size(1)):
        prefix_last = model(ids[:, : t + 1])[:, -1]  # (batch, vocab)
        assert torch.allclose(full[:, t], prefix_last, atol=1e-4, rtol=1e-4), f"mismatch at position {t}"
        # argmax must agree exactly — what generation actually consumes.
        assert torch.equal(full[:, t].argmax(-1), prefix_last.argmax(-1)), f"argmax mismatch at position {t}"


@torch.no_grad()
def test_generate_reproducible_with_seed():
    """Same seed -> identical sampled output (sanity for the sampling path)."""
    model = make_model()
    prompt = random_prompt(batch=1, length=8, vocab_size=SMALL_CONFIG["vocab_size"])

    torch.manual_seed(42)
    out1 = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=50)
    torch.manual_seed(42)
    out2 = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=50)

    assert torch.equal(out1, out2)


@torch.no_grad()
def test_topk_one_collapses_to_greedy():
    """Regression lock for the top-k fix: with top_k=1 only the argmax survives,
    so sampling must reproduce greedy decoding regardless of seed."""
    model = make_model()
    prompt = random_prompt(batch=1, length=8, vocab_size=SMALL_CONFIG["vocab_size"])

    expected = greedy_reference(model, prompt, max_new_tokens=20)
    for seed in (0, 1, 7):
        torch.manual_seed(seed)
        out = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=1)
        assert torch.equal(out, expected), f"top_k=1 diverged from greedy at seed {seed}"
