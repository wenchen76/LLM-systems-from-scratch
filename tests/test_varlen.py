"""Phase 3 tests for selective (ragged) batching.

A flat batch of different-length sequences run through forward_varlen must give,
for every token, the same logits as running each sequence on its own — proving
there is no cross-sequence attention leakage and that per-token RoPE positions
are correct.
"""
import pytest
import torch

from llm_core.model import TransformerLM, pack_varlen, varlen_position_ids

SMALL_CONFIG = {
    "vocab_size": 256,
    "context_length": 64,
    "d_model": 64,
    "num_layers": 2,
    "num_heads": 4,
    "d_ff": 128,
    "rope_theta": 10000.0,
}


def make_sequences(lengths, vocab_size, seed=7):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab_size, (length,), generator=gen) for length in lengths]


def test_pack_varlen_shapes():
    seqs = make_sequences([5, 12, 3], SMALL_CONFIG["vocab_size"])
    flat, cu, pos = pack_varlen(seqs)

    assert flat.shape == (20,)
    assert cu.tolist() == [0, 5, 17, 20]
    # position_ids restart at 0 for each sequence
    assert pos.tolist() == list(range(5)) + list(range(12)) + list(range(3))
    assert torch.equal(varlen_position_ids(cu), pos)


@torch.no_grad()
@pytest.mark.parametrize("use_flash_attn", [False, True])
def test_varlen_matches_individual_runs(use_flash_attn):
    torch.manual_seed(0)
    model = TransformerLM(**SMALL_CONFIG, use_flash_attn=use_flash_attn).eval()
    seqs = make_sequences([5, 12, 3, 8], SMALL_CONFIG["vocab_size"])

    flat, cu, pos = pack_varlen(seqs)
    packed = model.forward_varlen(flat, cu, pos)  # (total_tokens, vocab)
    assert packed.shape == (flat.numel(), SMALL_CONFIG["vocab_size"])

    bounds = cu.tolist()
    for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        reference = model(seqs[i].unsqueeze(0))[0]  # (len_i, vocab) run alone
        got = packed[start:end]
        max_diff = (got - reference).abs().max().item()
        assert torch.allclose(got, reference, atol=1e-4), f"seq {i} diverged, max diff {max_diff}"


@torch.no_grad()
@pytest.mark.parametrize("use_flash_attn", [False, True])
def test_varlen_with_per_request_cache_matches_individual(use_flash_attn):
    """The continuous-batching forward path: ragged prefill then lockstep decode,
    each request attending to its own KV cache, must match running each alone.
    """
    torch.manual_seed(0)
    model = TransformerLM(**SMALL_CONFIG, use_flash_attn=use_flash_attn).eval()
    sa, sb = make_sequences([6, 4], SMALL_CONFIG["vocab_size"])
    ref_a = model(sa.unsqueeze(0))[0]  # (6, vocab) run alone
    ref_b = model(sb.unsqueeze(0))[0]  # (4, vocab)

    caches = [model.new_kv_cache(), model.new_kv_cache()]  # one per request

    def step(tok_a, pos_a, tok_b, pos_b):
        flat = torch.cat([tok_a, tok_b])
        cu = torch.tensor([0, tok_a.numel(), tok_a.numel() + tok_b.numel()])
        pos = torch.cat([pos_a, pos_b])
        logits = model.forward_varlen(flat, cu, pos, request_kv_caches=caches)
        return logits[: tok_a.numel()], logits[tok_a.numel():]

    # Ragged prefill: 4 tokens of A, 2 tokens of B.
    la, lb = step(sa[:4], torch.arange(4), sb[:2], torch.arange(2))
    assert torch.allclose(la[-1], ref_a[3], atol=1e-4)
    assert torch.allclose(lb[-1], ref_b[1], atol=1e-4)

    # Lockstep decode of the remaining tokens (each at its own absolute position).
    for pa, pb in ((4, 2), (5, 3)):
        la, lb = step(sa[pa:pa + 1], torch.tensor([pa]), sb[pb:pb + 1], torch.tensor([pb]))
        assert torch.allclose(la[0], ref_a[pa], atol=1e-4), f"A pos {pa}"
        assert torch.allclose(lb[0], ref_b[pb], atol=1e-4), f"B pos {pb}"

    assert caches[0][0].length == 6 and caches[1][0].length == 4


@torch.no_grad()
@pytest.mark.parametrize("use_flash_attn", [False, True])
def test_varlen_interleaved_decode_prefill_decode(use_flash_attn):
    """A batch interleaving the two attention regimes — decode | fresh prefill |
    decode — must match running each sequence alone, pinning the dispatch and
    the scatter of per-regime results back to original token slots."""
    torch.manual_seed(0)
    model = TransformerLM(**SMALL_CONFIG, use_flash_attn=use_flash_attn).eval()
    sa, sb, sc = make_sequences([6, 5, 7], SMALL_CONFIG["vocab_size"])
    ref_a, ref_b, ref_c = (model(s.unsqueeze(0))[0] for s in (sa, sb, sc))

    caches = [model.new_kv_cache() for _ in range(3)]

    # Advance A and C to position 4 via their own prefill steps.
    for s, cache in ((sa, caches[0]), (sc, caches[2])):
        model.forward_varlen(s[:4], torch.tensor([0, 4]), torch.arange(4), request_kv_caches=[cache])

    # Mixed step: decode A's token 4 | prefill all of B | decode C's token 4.
    flat = torch.cat([sa[4:5], sb, sc[4:5]])
    cu = torch.tensor([0, 1, 6, 7])
    pos = torch.cat([torch.tensor([4]), torch.arange(5), torch.tensor([4])])
    logits = model.forward_varlen(flat, cu, pos, request_kv_caches=caches)

    assert torch.allclose(logits[0], ref_a[4], atol=1e-4)
    assert torch.allclose(logits[1:6], ref_b, atol=1e-4)
    assert torch.allclose(logits[6], ref_c[4], atol=1e-4)
    assert [c[0].length for c in caches] == [5, 5, 5]


@torch.no_grad()
def test_varlen_position_ids_derived_when_omitted():
    """forward_varlen should derive per-sequence positions from cu_seqlens."""
    torch.manual_seed(0)
    model = TransformerLM(**SMALL_CONFIG).eval()
    seqs = make_sequences([6, 4], SMALL_CONFIG["vocab_size"])
    flat, cu, pos = pack_varlen(seqs)

    with_pos = model.forward_varlen(flat, cu, pos)
    derived = model.forward_varlen(flat, cu, position_ids=None)
    assert torch.equal(with_pos, derived)
