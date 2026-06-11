from __future__ import annotations

import functools
import json
import logging
import math
import os
from einops import rearrange, einsum
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int
import torch.cuda.nvtx as nvtx


from .nn_functional import softmax

logger = logging.getLogger(__name__)


class Linear(nn.Module):
    """Linear layer with truncated normal initialization."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        init_std = math.sqrt(2 / (d_in + d_out))
        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(d_out, d_in), std=init_std, a=-3 * init_std, b=3 * init_std),
        )

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

    def extra_repr(self) -> str:
        return f"d_in={self.weight.shape[1]}, d_out={self.weight.shape[0]}"


class Embedding(nn.Module):
    """Lookup table with truncated normal initialization (std=1.0)."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        init_std = 1.0
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(vocab_size, d_model), std=init_std, a=-3 * init_std, b=3 * init_std),
        )

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids, :]

    def extra_repr(self) -> str:
        return f"vocab_size={self.weight.shape[0]}, d_model={self.weight.shape[1]}"


class RMSNorm(nn.Module):
    """Root mean square layer normalization (https://arxiv.org/abs/1910.07467)."""

    def __init__(self, d_model: int, eps: float = 1e-5, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device))
        self.eps = eps

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        # Upcast to fp32 to prevent overflow when squaring
        # https://github.com/pytorch/pytorch/issues/66707
        input_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * (x * rms)).to(input_dtype)

    def extra_repr(self) -> str:
        return f"d_model={self.weight.shape[0]}, eps={self.eps}"


class RotaryEmbedding(nn.Module):
    """Rotary positional encoding (RoPE). Precomputes and caches cos/sin frequencies."""

    def __init__(self, context_length: int, d: int, theta: float = 10000.0):
        super().__init__()
        self.register_buffer(
            "_freq_cache",
            RotaryEmbedding._build_cache(context_length, d, theta), persistent=False
        )

    @staticmethod
    def _build_cache(context_length: int, d: int, theta: float) -> Float[Tensor, " 2 context_length half_d"]:
        assert d % 2 == 0
        inv_freq = theta ** -(torch.arange(0, d, 2) / d)
        positions = torch.arange(context_length)
        angles = einsum(positions, inv_freq, "context_length, half_d -> context_length half_d")
        return torch.stack((torch.cos(angles), torch.sin(angles)))

    def forward(self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"]) -> Float[Tensor, " ... seq d"]:
        x1 = x[..., 0::2]  # (..., seq, d/2)
        x2 = x[..., 1::2]  # (..., seq, d/2)

        # _freq_cache: (2, context_length, d/2) -> cos, sin: (..., seq, d/2)
        cos, sin = self._freq_cache[:, pos_ids, :]

        # Apply 2D rotation to each pair of dimensions
        x1_rot = cos * x1 - sin * x2  # (..., seq, d/2)
        x2_rot = sin * x1 + cos * x2  # (..., seq, d/2)
        # (2, ..., seq, d/2) -> (..., seq, d)
        return rearrange([x1_rot, x2_rot], 'xy ... half_d -> ... (half_d xy)', xy=2).contiguous()

    def extra_repr(self) -> str:
        return f"context_length={self._freq_cache.shape[1]}, d={self._freq_cache.shape[2] * 2}"


def pack_varlen(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack ragged sequences into flat tensors for selective batching.

    Args:
        sequences: list of 1-D token-id tensors, possibly of different lengths.

    Returns:
        flat_ids: (total_tokens,) all tokens concatenated, with no padding.
        cu_seqlens: (num_seqs + 1,) cumulative lengths, cu_seqlens[0] == 0.
        position_ids: (total_tokens,) per-sequence positions, each restarting at 0.
    """
    device = sequences[0].device
    flat_ids = torch.cat([s.reshape(-1) for s in sequences])
    lengths = torch.tensor([s.numel() for s in sequences], device=device)
    cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.long, device=device), lengths.cumsum(0)])
    position_ids = torch.cat([torch.arange(s.numel(), device=device) for s in sequences])
    return flat_ids, cu_seqlens, position_ids


def varlen_position_ids(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Per-sequence positions (each restarting at 0) for a packed varlen batch."""
    bounds = cu_seqlens.tolist()
    device = cu_seqlens.device
    return torch.cat([torch.arange(end - start, device=device) for start, end in zip(bounds[:-1], bounds[1:])])


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model with RoPE, RMSNorm, and SwiGLU.

    Architecture: token embedding -> N x TransformerBlock -> RMSNorm -> linear head.
    Outputs unnormalized logits of shape (batch, seq, vocab_size).

    Args:
        vocab_size: Size of the token vocabulary.
        context_length: Maximum sequence length the model can process.
        d_model: Dimensionality of embeddings and sublayer outputs.
        num_layers: Number of TransformerBlock layers.
        num_heads: Number of attention heads. d_model must be divisible by num_heads.
        d_ff: Inner dimensionality of the feed-forward layer.
        rope_theta: Base frequency for rotary positional encoding.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        use_flash_attn: bool = False,
        use_custom_triton: bool = False,
    ):
        # Capture config for serialization
        self.config = {
            k: v for k, v in locals().items() if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.positional_encoder = RotaryEmbedding(
            context_length=context_length,
            d=d_model // num_heads,
            theta=rope_theta,
        )
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                positional_encoder=self.positional_encoder,
                use_flash_attn=use_flash_attn,
                use_custom_triton=use_custom_triton,
            )
            for _ in range(num_layers)
        ])
        if use_custom_triton:
            from llm_systems.kernels.triton_rms_norm import TritonRMSNorm
            self.final_norm = TritonRMSNorm(d_model)
        else:
            self.final_norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

        logger.info(f"non-embedding parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return parameter count. Excludes lm_head (weight-tied) by default."""
        num_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            num_params -= self.lm_head.weight.numel()
        return num_params

    def forward(
        self,
        token_ids: Int[Tensor, " ... seq"],
        token_positions: Int[Tensor, " ... seq"] | None = None,
        kv_caches: list[KVCache] | None = None,
    ) -> Float[Tensor, " ... seq vocab_size"]:
        x = self.token_embeddings(token_ids)  # (..., seq, d_model)
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            x = layer(x, token_positions=token_positions, kv_cache=kv_cache)  # (..., seq, d_model)
        x = self.final_norm(x)                 # (..., seq, d_model)
        return self.lm_head(x)                 # (..., seq, vocab_size)

    def new_kv_cache(self) -> list[KVCache]:
        """Create a fresh, empty KV cache (one KVCache per layer) for decoding."""
        return [KVCache() for _ in self.layers]

    @torch.no_grad()
    def forward_varlen(
        self,
        token_ids: Int[Tensor, " total_tokens"],
        cu_seqlens: Int[Tensor, " num_seqs_plus_1"],
        position_ids: Int[Tensor, " total_tokens"] | None = None,
        request_kv_caches: list[list[KVCache]] | None = None,
    ) -> Float[Tensor, " total_tokens vocab_size"]:
        """Selective-batching forward over a flat, ragged batch (no padding).

        Multiple sequences of different lengths are concatenated along a single
        token axis. Token-wise layers run on the whole flat batch at once; only
        attention is computed per sequence, bounded by cu_seqlens. See
        pack_varlen for building the inputs.

        Args:
            token_ids: (total_tokens,) all sequences' tokens concatenated.
            cu_seqlens: (num_seqs + 1,) cumulative lengths, cu_seqlens[0] == 0.
            position_ids: (total_tokens,) per-token positions; derived from
                cu_seqlens (each sequence restarting at 0) when None. Must be
                supplied (as absolute positions) when request_kv_caches is given.
            request_kv_caches: one per sequence, each a per-layer list[KVCache]
                (i.e. the output of new_kv_cache). Enables the mixed prefill+decode
                path of continuous batching, where each sequence attends over its
                own cached history. When None, every sequence is a fresh prefill.

        Returns:
            logits of shape (total_tokens, vocab_size).
        """
        if position_ids is None:
            if request_kv_caches is not None:
                raise ValueError("position_ids (absolute) must be provided when using KV caches")
            position_ids = varlen_position_ids(cu_seqlens)
        x = self.token_embeddings(token_ids)  # (total_tokens, d_model)
        for j, layer in enumerate(self.layers):
            layer_caches = [rc[j] for rc in request_kv_caches] if request_kv_caches is not None else None
            x = layer.forward_varlen(x, cu_seqlens, position_ids, layer_caches)  # (total_tokens, d_model)
        x = self.final_norm(x)                 # (total_tokens, d_model)
        return self.lm_head(x)                 # (total_tokens, vocab_size)

    @torch.no_grad()
    def prefill(
        self,
        prompt_ids: torch.Tensor,
        kv_caches: list[KVCache] | None = None,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        """Run the prompt through the model, populating a KV cache.

        Args:
            prompt_ids: Token IDs of shape (batch, seq) or (seq,).
            kv_caches: Caches to fill; a fresh one is allocated if None.

        Returns:
            (logits, kv_caches), where logits are the last-position next-token
            logits of shape (batch, vocab_size).
        """
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if kv_caches is None:
            kv_caches = self.new_kv_cache()
        logits = self.forward(prompt_ids, kv_caches=kv_caches)[:, -1]
        return logits, kv_caches

    @torch.no_grad()
    def decode_step(
        self,
        token_ids: torch.Tensor,
        kv_caches: list[KVCache],
    ) -> tuple[torch.Tensor, list[KVCache]]:
        """Decode a single step given the most recent token(s).

        Args:
            token_ids: The last token per sequence, shape (batch,) or (batch, 1).
            kv_caches: The caches to extend (mutated in place).

        Returns:
            (logits, kv_caches), with logits of shape (batch, vocab_size).
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(-1)  # (batch, 1)
        logits = self.forward(token_ids, kv_caches=kv_caches)[:, -1]
        return logits, kv_caches

    def _sample_next(self, logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
        """Sample the next token id from last-step logits. Returns (batch, 1)."""
        scaled_logits = logits / temperature
        if top_k:
            top_values, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
            threshold = top_values[:, -1:]  # (batch, 1) so it broadcasts over vocab
            scaled_logits = scaled_logits.masked_fill(scaled_logits < threshold, float("-inf"))
        probs = softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, 1)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive token generation with temperature scaling and optional top-k sampling.

        Args:
            prompt_ids: Input token IDs of shape (1, seq) or (seq,).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Softmax temperature for sampling.
            top_k: If set, only sample from the top-k highest probability tokens.
            eos_token_id: If set, stop generation upon producing this token.
            use_cache: If True (default), decode incrementally with a KV cache —
                prefill the prompt once, then feed a single token per step. Yields
                the same tokens as the no-cache path but avoids recomputing the
                full sequence each step.

        Returns:
            Generated token IDs of shape (1, num_generated).

        Note:
            With use_cache, total length (prompt + generated) is capped at
            context_length because RoPE positions are absolute. The no-cache path
            instead slides a context_length window. The two agree as long as the
            total stays within context_length (a sliding-window cache is future work).
        """
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        if not use_cache:
            input_ids = prompt_ids
            generated = []
            for _ in range(max_new_tokens):
                context_ids = input_ids[:, -self.context_length:]  # slide context window
                logits = self.forward(context_ids)[:, -1]  # (batch, vocab_size)
                next_id = self._sample_next(logits, temperature, top_k)
                if eos_token_id is not None and next_id.item() == eos_token_id:
                    break
                input_ids = torch.cat((input_ids, next_id), dim=-1)
                generated.append(next_id)
            return torch.cat(generated, dim=-1) if generated else prompt_ids[:, 0:0]

        # Cached path: prefill once, then decode one token at a time.
        prefill_ids = prompt_ids[:, -self.context_length:]  # fit the context window
        logits, caches = self.prefill(prefill_ids)
        generated = []
        for _ in range(max_new_tokens):
            next_id = self._sample_next(logits, temperature, top_k)
            if eos_token_id is not None and next_id.item() == eos_token_id:
                break
            generated.append(next_id)
            if caches[0].length >= self.context_length:
                break  # reached the absolute-position / context limit
            logits, caches = self.decode_step(next_id, caches)
        return torch.cat(generated, dim=-1) if generated else prompt_ids[:, 0:0]

    @classmethod
    def from_pretrained(cls, config_path: str, checkpoint_path: str) -> TransformerLM:
        with open(config_path) as f:
            config = json.load(f)
        model = cls(**config)

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # Unwrap checkpoint saved by save_checkpoint
        if "model" in state_dict:
            state_dict = state_dict["model"]

        # Strip _orig_mod. prefix from torch.compile'd checkpoints
        compiled_prefix = "_orig_mod."
        for key in list(state_dict.keys()):
            if key.startswith(compiled_prefix):
                state_dict[key[len(compiled_prefix):]] = state_dict.pop(key)

        model.load_state_dict(state_dict)
        return model


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        positional_encoder: RotaryEmbedding,
        use_flash_attn: bool = False,
        use_custom_triton: bool = False,
    ):
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            positional_encoder=positional_encoder,
            use_flash_attn=use_flash_attn,
        )
        if use_custom_triton:
            from llm_systems.kernels.triton_rms_norm import TritonRMSNorm
            from llm_systems.kernels.triton_swiglu import FusedSwiGLU
            self.ffn = FusedSwiGLU(d_model=d_model, d_ff=d_ff)
            self.attn_norm = TritonRMSNorm(d_model)
            self.ffn_norm = TritonRMSNorm(d_model)
        else:
            self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
            self.attn_norm = RMSNorm(d_model)
            self.ffn_norm = RMSNorm(d_model)

    def forward(
        self,
        x: Float[Tensor, " ... seq d_model"],
        token_positions: Int[Tensor, " ... seq"] | None = None,
        kv_cache: KVCache | None = None,
    ) -> Float[Tensor, " ... seq d_model"]:
        x = x + self.attn(self.attn_norm(x), token_positions=token_positions, kv_cache=kv_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def forward_varlen(
        self,
        x: Float[Tensor, " total_tokens d_model"],
        cu_seqlens: Int[Tensor, " num_seqs_plus_1"],
        position_ids: Int[Tensor, " total_tokens"],
        kv_caches: list[KVCache] | None = None,
    ) -> Float[Tensor, " total_tokens d_model"]:
        x = x + self.attn.forward_varlen(self.attn_norm(x), cu_seqlens, position_ids, kv_caches)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: W2(SiLU(W1(x)) * W3(x))."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = Linear(d_model, d_ff)
        self.down_proj = Linear(d_ff, d_model)
        self.up_proj = Linear(d_model, d_ff)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))


def scaled_dot_product_attention (
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention (Eq. 1 of Vaswani et al.)."""
    d_k = K.shape[-1]
    # (..., queries, keys)
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    attention_weights = softmax(attention_scores, dim=-1)

    # (..., queries, d_v)
    return einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")


class KVCache:
    """Per-layer key/value cache for incremental (autoregressive) decoding.

    Holds K and V of shape (batch, num_heads, seq, d_head), growing along the
    sequence dim as tokens are appended. One instance per attention layer; the
    model owns a list of them (see TransformerLM.new_kv_cache).

    This is mutated in place so attention can keep returning a plain tensor,
    leaving the training / full-forward path untouched.
    """

    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    @property
    def length(self) -> int:
        """Number of tokens currently cached."""
        return 0 if self.k is None else self.k.size(-2)

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new keys/values and return the full cached K, V."""
        self.k = k if self.k is None else torch.cat((self.k, k), dim=-2)
        self.v = v if self.v is None else torch.cat((self.v, v), dim=-2)
        return self.k, self.v


class CausalMultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoder: RotaryEmbedding,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_flash_attn = use_flash_attn

        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, num_heads * self.d_head)
        self.k_proj = Linear(d_model, num_heads * self.d_head)
        self.v_proj = Linear(d_model, num_heads * self.d_head)
        self.output_proj = Linear(num_heads * self.d_head, d_model)

        self.positional_encoder = positional_encoder

    def forward(
        self,
        x: Float[Tensor, " ... seq d_model"],
        token_positions: Int[Tensor, " ... seq"] | None = None,
        kv_cache: KVCache | None = None,
    ) -> Float[Tensor, " ... seq d_model"]:
        *batch_dims, seq_len, d_model = x.size()
        assert d_model == self.d_model

        # Project to Q, K, V: (..., seq, d_model) -> (..., seq, num_heads * d_head)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split heads: (..., seq, num_heads * d_head) -> (..., num_heads, seq, d_head)
        Q, K, V = (
            rearrange(t, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
            for t in (Q, K, V)
        )

        # Apply RoPE. New tokens are positioned after whatever is already cached;
        # with no cache past_len == 0, so this reduces to arange(seq_len).
        past_len = kv_cache.length if kv_cache is not None else 0
        if token_positions is None:
            token_positions = torch.arange(
                past_len, past_len + seq_len, device=x.device
            ).view(*([1] * len(batch_dims)), seq_len)
        # Expand for heads: (..., seq) -> (..., 1, seq)
        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
        Q = self.positional_encoder(Q, token_positions)
        K = self.positional_encoder(K, token_positions)

        # Attention
        if kv_cache is None:
            # Full forward / training path (unchanged).
            if self.use_flash_attn:
                attn_output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
            else:
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
                attn_output = scaled_dot_product_attention(Q, K, V, causal_mask)
        else:
            # Incremental path: append new K/V, then attend over the full history.
            K, V = kv_cache.append(K, V)
            kv_len = K.size(-2)
            # Query i is at absolute position past_len + i and may attend to keys
            # 0..past_len+i. For pure decode (seq_len == 1) this keeps every key.
            q_pos = torch.arange(past_len, past_len + seq_len, device=x.device).unsqueeze(-1)
            k_pos = torch.arange(kv_len, device=x.device).unsqueeze(0)
            attn_mask = k_pos <= q_pos  # (seq_len, kv_len), True = attend
            if self.use_flash_attn:
                attn_output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
            else:
                attn_output = scaled_dot_product_attention(Q, K, V, attn_mask)

        # Merge heads: (..., num_heads, seq, d_head) -> (..., seq, num_heads * d_head)
        attn_output = rearrange(attn_output, "... heads seq d_head -> ... seq (heads d_head)").contiguous()

        return self.output_proj(attn_output)

    def forward_varlen(
        self,
        x: Float[Tensor, " total_tokens d_model"],
        cu_seqlens: Int[Tensor, " num_seqs_plus_1"],
        position_ids: Int[Tensor, " total_tokens"],
        kv_caches: list[KVCache] | None = None,
    ) -> Float[Tensor, " total_tokens d_model"]:
        """Selective-batching attention over a flat, ragged batch.

        Sequences are concatenated along one token axis with no padding;
        cu_seqlens marks their boundaries and position_ids gives each token's
        absolute position. Projections run on the whole flat batch at once; only
        attention is done per sequence, since attention is the single op that
        mixes the sequence dimension.

        When kv_caches is given (one KVCache per sequence, each holding this
        layer's (heads, seq, d_head)), every sequence attends over its cached
        history plus its new tokens — the mixed prefill+decode path used by
        continuous batching. Without it, each sequence is a fresh prefill.

        Sequences are dispatched by regime — prefill (a run of new tokens) vs
        decode (one new token over cached history) — because the two parallelize
        differently: prefill has many queries per sequence, decode must split
        along the KV length instead. Each group is an explicit per-sequence loop
        here; production fuses each into a single kernel (varlen flash attention
        for prefill, flash-decoding for decode).
        """
        total_tokens, d_model = x.shape
        assert d_model == self.d_model

        # Token-wise projections over the entire flat batch (no padding).
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q, K, V = (
            rearrange(t, "tokens (heads d) -> heads tokens d", heads=self.num_heads)
            for t in (Q, K, V)
        )  # (heads, total_tokens, d_head)

        # Per-token RoPE with absolute positions; (1, total_tokens) broadcasts over heads.
        pos = position_ids.view(1, total_tokens)
        Q = self.positional_encoder(Q, pos)
        K = self.positional_encoder(K, pos)

        # Attention is the only op that mixes the sequence dimension. Split the
        # sequences by regime and attend each group, scattering results back to
        # every token's original slot (the groups may interleave in the batch).
        bounds = cu_seqlens.tolist()
        prefill_seqs, decode_seqs = [], []
        for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            cache = kv_caches[i] if kv_caches is not None else None
            if cache is not None and end - start == 1:
                decode_seqs.append((start, cache))
            else:
                prefill_seqs.append((start, end, cache))

        attn_output = torch.empty_like(V)  # (heads, total_tokens, d_head)
        self._attend_prefill(Q, K, V, prefill_seqs, attn_output)
        self._attend_decode(Q, K, V, decode_seqs, attn_output)

        attn_output = rearrange(attn_output, "heads tokens d -> tokens (heads d)").contiguous()
        return self.output_proj(attn_output)

    def _attend_prefill(
        self,
        Q: Float[Tensor, " heads total_tokens d_head"],
        K: Float[Tensor, " heads total_tokens d_head"],
        V: Float[Tensor, " heads total_tokens d_head"],
        seqs: list[tuple[int, int, KVCache | None]],
        out: Float[Tensor, " heads total_tokens d_head"],
    ) -> None:
        """Causal attention over each sequence's run of new tokens, optionally on
        top of cached history. Masks are built explicitly (True = attend) so the
        non-square with-cache case works.

        On CUDA with no cached history (the common case) the whole group is one
        fused varlen kernel launch; otherwise (CPU, or chunked prefill over a
        non-empty cache) it falls back to one eager call per sequence.
        """
        if not seqs:
            return

        # Append new K/V to each cache for later decode steps (bookkeeping); read
        # past lengths first so we can tell empty-cache prefill from chunked.
        past_lens = [cache.length if cache is not None else 0 for _, _, cache in seqs]
        for start, end, cache in seqs:
            if cache is not None:
                cache.append(K[:, start:end], V[:, start:end])

        fresh = all(p == 0 for p in past_lens)  # no cached history -> plain causal self-attention
        if self.use_flash_attn and Q.is_cuda and fresh:
            try:
                from llm_systems.kernels.triton_flash_attention import flash_prefill_varlen
            except ImportError:
                flash_prefill_varlen = None
            if flash_prefill_varlen is not None:
                device = Q.device
                starts = torch.tensor([s for s, _, _ in seqs], device=device, dtype=torch.int32)
                lengths = torch.tensor([e - s for s, e, _ in seqs], device=device, dtype=torch.int32)
                flash_prefill_varlen(Q.contiguous(), K.contiguous(), V.contiguous(), starts, lengths, out)
                return

        for (start, end, cache), past_len in zip(seqs, past_lens):
            q = Q[:, start:end]
            q_len = end - start
            if cache is None:
                k, v = K[:, start:end], V[:, start:end]
                attn_mask = torch.tril(torch.ones(q_len, q_len, device=q.device, dtype=torch.bool))
            else:
                k, v = cache.k, cache.v  # already appended above -> past + new
                kv_len = k.size(-2)
                q_pos = torch.arange(past_len, past_len + q_len, device=q.device).unsqueeze(-1)
                k_pos = torch.arange(kv_len, device=q.device).unsqueeze(0)
                attn_mask = k_pos <= q_pos  # (q_len, kv_len)
            if self.use_flash_attn:
                out[:, start:end] = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                out[:, start:end] = scaled_dot_product_attention(q, k, v, attn_mask)

    def _attend_decode(
        self,
        Q: Float[Tensor, " heads total_tokens d_head"],
        K: Float[Tensor, " heads total_tokens d_head"],
        V: Float[Tensor, " heads total_tokens d_head"],
        seqs: list[tuple[int, KVCache]],
        out: Float[Tensor, " heads total_tokens d_head"],
    ) -> None:
        """Single-query attention over each sequence's cached history plus its
        new token. The query sits at the newest position and sees every key, so
        no mask is needed.

        On CUDA the whole group is one fused launch: caches are concatenated
        along the key axis (k_starts / k_lens mark each sequence's span) and read
        via those offsets; otherwise it falls back to one eager call per sequence.
        """
        if not seqs:
            return

        for start, cache in seqs:
            cache.append(K[:, start : start + 1], V[:, start : start + 1])

        if self.use_flash_attn and Q.is_cuda:
            try:
                from llm_systems.kernels.triton_flash_attention import flash_decode
            except ImportError:
                flash_decode = None
            if flash_decode is not None:
                device = Q.device
                k_flat = torch.cat([cache.k for _, cache in seqs], dim=1).contiguous()
                v_flat = torch.cat([cache.v for _, cache in seqs], dim=1).contiguous()
                lens = [cache.k.size(1) for _, cache in seqs]
                starts = [0]
                for length in lens[:-1]:
                    starts.append(starts[-1] + length)
                flash_decode(
                    Q.contiguous(), k_flat, v_flat, out,
                    torch.tensor([s for s, _ in seqs], device=device, dtype=torch.int32),
                    torch.tensor(starts, device=device, dtype=torch.int32),
                    torch.tensor(lens, device=device, dtype=torch.int32),
                )
                return

        for start, cache in seqs:
            q = Q[:, start : start + 1]  # (heads, 1, d_head)
            if self.use_flash_attn:
                out[:, start : start + 1] = torch.nn.functional.scaled_dot_product_attention(q, cache.k, cache.v)
            else:
                out[:, start : start + 1] = scaled_dot_product_attention(q, cache.k, cache.v)


def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)
