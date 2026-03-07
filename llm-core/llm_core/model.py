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
            )
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

        logger.info(f"non-embedding parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return parameter count. Excludes lm_head (weight-tied) by default."""
        num_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            num_params -= self.lm_head.weight.numel()
        return num_params

    def forward(self, token_ids: Int[Tensor, " ... seq"]) -> Float[Tensor, " ... seq vocab_size"]:
        x = self.token_embeddings(token_ids)  # (..., seq, d_model)
        for layer in self.layers:
            x = layer(x)                       # (..., seq, d_model)
        x = self.final_norm(x)                 # (..., seq, d_model)
        return self.lm_head(x)                 # (..., seq, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive token generation with temperature scaling and optional top-k sampling.

        Args:
            prompt_ids: Input token IDs of shape (1, seq) or (seq,).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Softmax temperature for sampling.
            top_k: If set, only sample from the top-k highest probability tokens.
            eos_token_id: If set, stop generation upon producing this token.

        Returns:
            Generated token IDs of shape (1, num_generated).
        """
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        input_ids = prompt_ids
        generated = []
        for _ in range(max_new_tokens):
            # Truncate to context window
            context_ids = input_ids[:, -self.context_length:] if input_ids.size(1) > self.context_length else input_ids

            # Get next-token logits and apply temperature
            logits = self.forward(context_ids)[:, -1]  # (1, vocab_size)
            scaled_logits = logits / temperature

            # Top-k filtering
            if top_k:
                top_values, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                threshold = top_values[:, -1]
                scaled_logits.masked_fill(scaled_logits < threshold, float("-inf"))

            probs = softmax(scaled_logits, dim=-1)
            next_id = torch.multinomial(probs, 1)

            if eos_token_id is not None and next_id.item() == eos_token_id:
                break

            input_ids = torch.cat((input_ids, next_id), dim=-1)
            generated.append(next_id)

        return torch.cat(generated, dim=-1) if generated else input_ids[:, 0:0]

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
    ):
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            positional_encoder=positional_encoder,
            use_flash_attn=use_flash_attn,
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)

    def forward(self, x: Float[Tensor, " ... seq d_model"]) -> Float[Tensor, " ... seq d_model"]:
        x = x + self.attn(self.attn_norm(x))
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

        # Apply RoPE to Q and K
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).view(*([1] * len(batch_dims)), seq_len)
        # Expand for heads: (..., seq) -> (..., 1, seq)
        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
        Q = self.positional_encoder(Q, token_positions)
        K = self.positional_encoder(K, token_positions)

        # Attention
        if self.use_flash_attn:
            attn_output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
        else:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
            attn_output = scaled_dot_product_attention(Q, K, V, causal_mask)

        # Merge heads: (..., num_heads, seq, d_head) -> (..., seq, num_heads * d_head)
        attn_output = rearrange(attn_output, "... heads seq d_head -> ... seq (heads d_head)").contiguous()

        return self.output_proj(attn_output)

def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)
