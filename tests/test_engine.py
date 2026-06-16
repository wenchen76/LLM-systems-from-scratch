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


def decode_inputs(engine, batch):
    """Build the fixed-batch tensors model.forward_decode expects from `batch`'s paged caches,
    growing each layer's block table for the new token (what the CUDA-graph path fills in place)."""
    bs, dev = engine.block_size, engine.device
    lengths = [r.kv_caches[0].length for r in batch]
    token_ids = torch.tensor([r.output_ids[-1] for r in batch], dtype=torch.long, device=dev)
    positions = torch.tensor(lengths, dtype=torch.long, device=dev)
    seq_lens = torch.tensor([L + 1 for L in lengths], dtype=torch.int32, device=dev)
    q_positions = torch.arange(len(batch), dtype=torch.int32, device=dev)
    max_blocks = engine.pools[0].max_seq_blocks or max(L // bs + 1 for L in lengths)
    block_tables, phys, offset = [], [], []
    for layer, pool in enumerate(engine.pools):
        bt = torch.zeros(len(batch), max_blocks, dtype=torch.int32, device=dev)
        ph = torch.empty(len(batch), dtype=torch.long, device=dev)
        of = torch.empty(len(batch), dtype=torch.long, device=dev)
        for i, req in enumerate(batch):
            cache = req.kv_caches[layer]
            L = cache.length
            if L // bs >= len(cache.block_table):
                cache.block_table.append(pool.allocate())
            bt[i, : len(cache.block_table)] = torch.tensor(cache.block_table, dtype=torch.int32, device=dev)
            ph[i] = cache.block_table[L // bs]
            of[i] = L % bs
        block_tables.append(bt)
        phys.append(ph)
        offset.append(of)
    return token_ids, positions, block_tables, phys, offset, seq_lens, q_positions


@pytest.mark.skipif(not torch.cuda.is_available(), reason="forward_decode runs the CUDA paged_decode kernel")
def test_forward_decode_matches_varlen():
    """The loop-free forward_decode (the forward that gets CUDA-graph captured) must compute
    the same decode logits as the forward_varlen decode path for the same paged caches. Two
    identically-prefilled engines run one decode step each, one via each path."""
    model = make_model().to("cuda")
    for module in model.modules():  # so the forward_varlen decode path also uses paged_decode
        if hasattr(module, "use_flash_attn"):
            module.use_flash_attn = True
    prompts = make_prompts([5, 7, 4], SMALL_CONFIG["vocab_size"])
    sp = SamplingParams(max_tokens=10, top_k=1)

    def prefilled_engine():
        e = LLMEngine(model, device="cuda", paged=True, block_size=4, num_blocks=64)
        for p in prompts:
            e.add_request(p, sp)
        e.step()  # admit + prefill; every request is now one token into decode
        return e

    e_fd, e_fv = prefilled_engine(), prefilled_engine()
    ins = decode_inputs(e_fd, e_fd.running)
    logits_fd = model.forward_decode(ins[0], ins[1], e_fd.pools, *ins[2:])     # (R, vocab)
    flat, cu, pos, caches, last_idx = e_fv._build_batch()
    logits_fv = model.forward_varlen(flat, cu, pos, request_kv_caches=caches)[last_idx]

    assert torch.allclose(logits_fd, logits_fv, atol=1e-3)


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
