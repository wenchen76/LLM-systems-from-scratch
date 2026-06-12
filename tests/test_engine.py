"""Phase 4 tests for the continuous-batching LLMEngine.

The decisive correctness property: batching requests together (with mid-flight
admission and retirement) must produce exactly the same tokens as running each
request on its own through model.generate. Greedy (top_k=1) makes the comparison
deterministic.
"""
import pytest
import torch

from llm_core.engine import LLMEngine, Request, RequestState, SamplingParams
from llm_core.model import TransformerLM

SMALL_CONFIG = {
    "vocab_size": 256,
    "context_length": 64,
    "d_model": 64,
    "num_layers": 2,
    "num_heads": 4,
    "d_ff": 128,
    "rope_theta": 10000.0,
}


def make_model():
    torch.manual_seed(0)
    return TransformerLM(**SMALL_CONFIG).eval()


def make_prompts(lengths, vocab_size, seed=7):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab_size, (length,), generator=gen).tolist() for length in lengths]


def greedy_reference(model, prompt, max_tokens):
    out = model.generate(torch.tensor([prompt]), max_new_tokens=max_tokens, top_k=1)
    return out[0].tolist()


def test_engine_greedy_matches_sequential_generate():
    model = make_model()
    prompts = make_prompts([5, 12, 3, 8], SMALL_CONFIG["vocab_size"])
    sp = SamplingParams(max_tokens=20, top_k=1)

    engine = LLMEngine(model, device="cpu")
    outputs = engine.generate(prompts, sp)

    for prompt, out in zip(prompts, outputs):
        assert out == greedy_reference(model, prompt, 20)
    assert not engine.has_work()


def test_paged_engine_greedy_matches_sequential():
    """End-to-end paging: continuous batching over a paged KV cache must produce
    the same tokens as sequential generate, and all blocks return to the pool."""
    model = make_model()
    prompts = make_prompts([5, 12, 3, 8], SMALL_CONFIG["vocab_size"])
    sp = SamplingParams(max_tokens=20, top_k=1)

    engine = LLMEngine(model, device="cpu", paged=True, block_size=4, num_blocks=512)
    outputs = engine.generate(prompts, sp)

    for prompt, out in zip(prompts, outputs):
        assert out == greedy_reference(model, prompt, 20)
    assert all(pool.num_free == pool.num_blocks for pool in engine.pools)  # no block leak


def test_paged_admission_control_queues_when_blocks_scarce():
    """With too few blocks to run everything at once, requests queue (no
    BlockPool exhaustion) and still finish correctly; blocks/reservations clear."""
    model = make_model()
    prompts = make_prompts([5, 6, 7, 8, 5, 6], SMALL_CONFIG["vocab_size"])
    sp = SamplingParams(max_tokens=20, top_k=1)

    # block_size=4: each request reserves ceil((prompt+20)/4) ~= 7 blocks, so only
    # ~2 of 16 fit at once -> the rest must wait rather than raise.
    engine = LLMEngine(model, device="cpu", paged=True, block_size=4, num_blocks=16, max_running=8)
    outputs = engine.generate(prompts, sp)

    for prompt, out in zip(prompts, outputs):
        assert out == greedy_reference(model, prompt, 20)
    assert engine.reserved_blocks == 0
    assert all(pool.num_free == pool.num_blocks for pool in engine.pools)


def test_paged_request_too_large_raises():
    """A single request that can never fit the pool fails fast, not silently hangs."""
    model = make_model()
    engine = LLMEngine(model, device="cpu", paged=True, block_size=4, num_blocks=2, max_running=4)
    with pytest.raises(ValueError):
        engine.generate(make_prompts([5], SMALL_CONFIG["vocab_size"]), SamplingParams(max_tokens=20, top_k=1))


def test_continuous_batching_with_capacity_limit_and_mid_flight_admission():
    """max_running smaller than the queue forces requests to wait and be admitted
    as slots free up; a request added mid-run must also complete correctly."""
    model = make_model()
    prompts = make_prompts([6, 10, 4, 7], SMALL_CONFIG["vocab_size"])
    extra = make_prompts([9], SMALL_CONFIG["vocab_size"], seed=99)[0]
    sp = SamplingParams(max_tokens=15, top_k=1)

    engine = LLMEngine(model, device="cpu", max_running=2)  # only 2 slots
    ids = [engine.add_request(p, sp) for p in prompts]

    collected: dict[int, list[int]] = {}
    steps = 0
    extra_id = None
    while engine.has_work():
        for req in engine.step():
            collected[req.request_id] = req.output_ids
        steps += 1
        if steps == 3:  # inject a new request mid-flight
            extra_id = engine.add_request(extra, sp)

    assert len(engine.running) == 0
    for rid, prompt in zip(ids, prompts):
        assert collected[rid] == greedy_reference(model, prompt, 15)
    assert collected[extra_id] == greedy_reference(model, extra, 15)


def test_per_request_max_tokens_honored():
    """Different requests may stop at different lengths in the same batch."""
    model = make_model()
    prompts = make_prompts([5, 5], SMALL_CONFIG["vocab_size"])

    engine = LLMEngine(model, device="cpu")
    engine.add_request(prompts[0], SamplingParams(max_tokens=8, top_k=1))
    engine.add_request(prompts[1], SamplingParams(max_tokens=20, top_k=1))

    done = {}
    while engine.has_work():
        for req in engine.step():
            done[req.request_id] = req
    assert len(done[0].output_ids) == 8 and done[0].finish_reason == "length"
    assert len(done[1].output_ids) == 20
    assert all(r.state is RequestState.FINISHED for r in done.values())


def test_vectorized_sampling_respects_per_request_top_k():
    """The vectorized _sample must mask each row by its own k: every sampled
    token stays within that request's top-k set (None = unrestricted)."""
    torch.manual_seed(0)
    engine = LLMEngine(make_model(), device="cpu")
    ks = [1, 5, None, 50]
    engine.running = [Request(i, [1], SamplingParams(top_k=k)) for i, k in enumerate(ks)]

    vocab = 200
    logits = torch.randn(len(ks), vocab)
    allowed = [set(torch.topk(logits[i], k).indices.tolist()) if k else None for i, k in enumerate(ks)]

    for _ in range(50):  # multinomial is stochastic; every draw must stay in-set
        tokens = engine._sample(logits)
        for i, k in enumerate(ks):
            if k:
                assert tokens[i].item() in allowed[i]


def test_eos_stops_request_early():
    model = make_model()
    prompt = make_prompts([6], SMALL_CONFIG["vocab_size"])[0]
    # Greedy reference to learn the first generated token, then set it as EOS.
    first_token = greedy_reference(model, prompt, 1)[0]

    engine = LLMEngine(model, device="cpu")
    out = engine.generate([prompt], SamplingParams(max_tokens=20, top_k=1, eos_token_id=first_token))[0]

    assert out == [first_token]  # stops immediately after emitting EOS
