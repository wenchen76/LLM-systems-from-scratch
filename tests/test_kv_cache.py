"""Phase 1 unit tests for the KV cache at the attention-layer level.

The incremental path (append K/V + shifted causal mask + absolute RoPE offset)
must reproduce, position-for-position, the output of a single full forward over
the whole sequence. This isolates the cache logic before it is threaded through
the full model.
"""
import torch

from llm_core.model import CausalMultiHeadSelfAttention, KVCache, RotaryEmbedding, TransformerLM

SMALL_CONFIG = {
    "vocab_size": 256,
    "context_length": 64,
    "d_model": 64,
    "num_layers": 2,
    "num_heads": 4,
    "d_ff": 128,
    "rope_theta": 10000.0,
}

D_MODEL = 64
NUM_HEADS = 4
CONTEXT_LENGTH = 64


def build_attention(seed: int = 0) -> CausalMultiHeadSelfAttention:
    torch.manual_seed(seed)
    rope = RotaryEmbedding(context_length=CONTEXT_LENGTH, d=D_MODEL // NUM_HEADS, theta=10000.0)
    return CausalMultiHeadSelfAttention(D_MODEL, NUM_HEADS, rope).eval()


@torch.no_grad()
def test_attention_token_by_token_matches_full():
    """Decode one token at a time through the cache == full forward."""
    attn = build_attention()
    torch.manual_seed(1)
    x = torch.randn(2, 12, D_MODEL)

    full = attn(x)  # no cache: (batch, seq, d_model)

    cache = KVCache()
    steps = [attn(x[:, t : t + 1], kv_cache=cache) for t in range(x.size(1))]
    incremental = torch.cat(steps, dim=1)

    assert cache.length == x.size(1)
    assert torch.allclose(full, incremental, atol=1e-5), (full - incremental).abs().max().item()


@torch.no_grad()
def test_generate_cached_matches_uncached():
    """End-to-end: cached generate == no-cache generate, token-for-token.

    Uses greedy (top_k=1) so the comparison is robust to tiny float differences
    between the two paths, and keeps total length within context_length where the
    absolute-position cache and the sliding-window no-cache path are equivalent.
    """
    torch.manual_seed(0)
    model = TransformerLM(**SMALL_CONFIG).eval()
    gen = torch.Generator().manual_seed(1234)
    prompt = torch.randint(0, SMALL_CONFIG["vocab_size"], (1, 8), generator=gen)

    max_new = SMALL_CONFIG["context_length"] - prompt.size(1)  # stay within the window
    cached = model.generate(prompt, max_new_tokens=max_new, top_k=1, use_cache=True)
    uncached = model.generate(prompt, max_new_tokens=max_new, top_k=1, use_cache=False)

    assert torch.equal(cached, uncached)
    assert cached.size(1) == max_new


@torch.no_grad()
def test_attention_prefill_then_decode_matches_full():
    """Prefill a chunk in one pass, then decode the rest one token at a time."""
    attn = build_attention(seed=3)
    torch.manual_seed(2)
    x = torch.randn(1, 10, D_MODEL)

    full = attn(x)

    cache = KVCache()
    prefill_len = 4
    steps = [attn(x[:, :prefill_len], kv_cache=cache)]
    for t in range(prefill_len, x.size(1)):
        steps.append(attn(x[:, t : t + 1], kv_cache=cache))
    incremental = torch.cat(steps, dim=1)

    assert cache.length == x.size(1)
    assert torch.allclose(full, incremental, atol=1e-5), (full - incremental).abs().max().item()
