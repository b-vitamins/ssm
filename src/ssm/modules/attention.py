from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


try:  # pragma: no cover - optional flash attention dependency
    from flash_attn import flash_attn_with_kvcache  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover - best effort import
    flash_attn_with_kvcache = None

try:  # pragma: no cover - optional rotary dependency
    from flash_attn.layers.rotary import RotaryEmbedding  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover - best effort import
    RotaryEmbedding = None


class MHA(nn.Module):
    """Multi-Head Attention with optional KV cache and depthwise convolution.

    Args:
        embed_dim: Input and output embedding dimension.
        num_heads: Number of attention heads for queries.
        num_heads_kv: Number of key/value heads for MQA/GQA setups.
        head_dim: Explicit head dimension. Defaults to ``embed_dim // num_heads``.
        mlp_dim: Optional MLP stream dimension packed with the QKV projection.
        qkv_proj_bias: Whether to use bias terms for the input projection.
        out_proj_bias: Whether to use bias terms on the output projection.
        softmax_scale: Optional override for the attention scaling factor.
        causal: Whether to apply causal masking during attention.
        layer_idx: Layer index for KV-cache addressing during decoding.
        d_conv: Depthwise convolution width applied before attention (0 disables).
        rotary_emb_dim: Rotary embedding dimension; 0 disables rotary embedding.
        rotary_emb_base: Base for rotary embeddings.
        rotary_emb_interleaved: Whether rotary dimensions are interleaved.
        device: Optional device placement for parameters.
        dtype: Optional dtype for parameters.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_heads_kv: int | None = None,
        head_dim: int | None = None,
        mlp_dim: int = 0,
        qkv_proj_bias: bool = True,
        out_proj_bias: bool = True,
        softmax_scale: float | None = None,
        causal: bool = False,
        layer_idx: int | None = None,
        d_conv: int = 0,
        rotary_emb_dim: int = 0,
        rotary_emb_base: float = 10000.0,
        rotary_emb_interleaved: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_idx = layer_idx
        self.d_conv = d_conv
        self.rotary_emb_dim = rotary_emb_dim
        self.softmax_scale = softmax_scale
        self.causal = causal

        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        if self.num_heads % self.num_heads_kv != 0:
            raise ValueError("num_heads must be divisible by num_heads_kv")
        if head_dim is None:
            if embed_dim % num_heads != 0:
                raise ValueError(
                    "embed_dim must be divisible by num_heads when head_dim is None"
                )
        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads
        self.mlp_dim = math.ceil(mlp_dim / 256) * 256 if mlp_dim > 0 else 0
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        out_dim = self.head_dim * self.num_heads

        if self.rotary_emb_dim > 0:
            if RotaryEmbedding is None:
                raise RuntimeError(
                    "Rotary embedding requested but flash_attn is unavailable"
                )
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )
        else:
            self.rotary_emb = None

        self.in_proj = nn.Linear(
            embed_dim,
            qkv_dim + self.mlp_dim,
            bias=qkv_proj_bias,
            device=device,
            dtype=dtype,
        )
        if self.d_conv > 0:
            self.conv1d = nn.Conv1d(
                qkv_dim,
                qkv_dim,
                kernel_size=self.d_conv,
                padding=self.d_conv - 1,
                groups=qkv_dim,
                device=device,
                dtype=dtype,
            )
        else:
            self.conv1d = None
        self.out_proj = nn.Linear(
            out_dim + self.mlp_dim // 2,
            embed_dim,
            bias=out_proj_bias,
            device=device,
            dtype=dtype,
        )

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype: torch.dtype | None = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Allocate KV (and optional conv) cache for decoding."""

        weight_dtype = self.out_proj.weight.dtype
        dtype = weight_dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        kv_cache = torch.empty(
            batch_size,
            max_seqlen,
            2,
            self.num_heads_kv,
            self.head_dim,
            dtype=dtype,
            device=device,
        )
        if self.conv1d is not None:
            conv_state = torch.zeros(
                batch_size,
                self.conv1d.weight.shape[0],
                self.d_conv,
                dtype=dtype,
                device=device,
            )
        else:
            conv_state = None
        return kv_cache, conv_state

    def _update_kv_cache(self, kv: torch.Tensor, inference_params) -> torch.Tensor:
        if self.layer_idx is None:
            raise RuntimeError(
                "Generation requires layer_idx to be set on the attention module"
            )
        if not hasattr(inference_params, "key_value_memory_dict"):
            raise AttributeError("inference_params must provide key_value_memory_dict")
        cache = inference_params.key_value_memory_dict[self.layer_idx]
        kv_cache, _ = cache
        batch_start = getattr(inference_params, "batch_size_offset", 0)
        batch_end = batch_start + kv.shape[0]
        sequence_start = getattr(inference_params, "seqlen_offset", 0)
        sequence_end = sequence_start + kv.shape[1]
        kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
        return kv_cache[batch_start:batch_end, :sequence_end, ...]

    def forward(self, x: torch.Tensor, inference_params=None, **kwargs) -> torch.Tensor:
        """Apply attention with optional cache updates."""

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        cache_dict = None
        if inference_params is not None:
            if self.layer_idx is None:
                raise RuntimeError("inference_params passed but layer_idx is None")
            if not hasattr(inference_params, "key_value_memory_dict"):
                inference_params.key_value_memory_dict = {}
            cache_dict = inference_params.key_value_memory_dict
            if cache_dict.get(self.layer_idx) is None:
                cache_dict[self.layer_idx] = self.allocate_inference_cache(
                    x.shape[0],
                    getattr(inference_params, "max_seqlen", x.shape[1]),
                    dtype=x.dtype,
                )
        qkv = self.in_proj(x)
        x_mlp: Optional[torch.Tensor]
        if self.mlp_dim > 0:
            qkv, mlp_stream = qkv.split(
                [qkv.shape[-1] - self.mlp_dim, self.mlp_dim], dim=-1
            )
            up, gate = mlp_stream.chunk(2, dim=-1)
            x_mlp = up * F.silu(gate)
        else:
            x_mlp = None

        if self.conv1d is not None:
            qkv = self._apply_depthwise_conv(qkv, inference_params)

        q, kv = torch.split(
            qkv,
            [self.num_heads * self.head_dim, 2 * self.num_heads_kv * self.head_dim],
            dim=-1,
        )
        batch, seqlen, _ = q.shape
        q = q.view(batch, seqlen, self.num_heads, self.head_dim)
        kv = kv.view(batch, seqlen, 2, self.num_heads_kv, self.head_dim)

        seqlen_offset = (
            getattr(inference_params, "seqlen_offset", 0)
            if inference_params is not None
            else 0
        )
        max_seqlen = (
            getattr(inference_params, "max_seqlen", None)
            if inference_params is not None
            else None
        )

        if self.rotary_emb is not None:
            q, kv = self.rotary_emb(
                q, kv, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen
            )

        if inference_params is None:
            k, v = kv.unbind(dim=2)
            k = k.repeat_interleave(self.num_heads // self.num_heads_kv, dim=2)
            v = v.repeat_interleave(self.num_heads // self.num_heads_kv, dim=2)
            context = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=self.causal,
                scale=self.softmax_scale,
            ).transpose(1, 2)
        else:
            cache_kv = self._update_kv_cache(kv, inference_params)
            if (
                flash_attn_with_kvcache is not None
                and getattr(inference_params, "seqlen_offset", 0) > 0
            ):
                cache, _ = inference_params.key_value_memory_dict[self.layer_idx]
                cache = cache[:batch]
                cache_lengths = getattr(inference_params, "lengths_per_sample", None)
                if cache_lengths is None:
                    cache_lengths = getattr(inference_params, "seqlen_offset", 0)
                context = flash_attn_with_kvcache(
                    q,
                    cache[:, :, 0],
                    cache[:, :, 1],
                    kv[:, :, 0],
                    kv[:, :, 1],
                    cache_seqlens=cache_lengths,
                    softmax_scale=self.softmax_scale,
                    causal=self.causal,
                )
            else:
                k, v = cache_kv.unbind(dim=2)
                k = k.repeat_interleave(self.num_heads // self.num_heads_kv, dim=2)
                v = v.repeat_interleave(self.num_heads // self.num_heads_kv, dim=2)
                causal_flag = self.causal and cache_kv.shape[1] == q.shape[1]
                context = F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    is_causal=causal_flag,
                    scale=self.softmax_scale,
                ).transpose(1, 2)

        context = context.reshape(batch, seqlen, -1)
        if x_mlp is not None:
            context = torch.cat([context, x_mlp], dim=-1)
        return self.out_proj(context)

    def _apply_depthwise_conv(
        self, qkv: torch.Tensor, inference_params
    ) -> torch.Tensor:
        assert self.conv1d is not None
        if (
            inference_params is None
            or getattr(inference_params, "seqlen_offset", 0) == 0
        ):
            qkv_t = qkv.transpose(1, 2)
            conv_out = self.conv1d(qkv_t)
            if self.d_conv > 1:
                conv_out = conv_out[..., : -(self.d_conv - 1)]
            conv_out = conv_out.transpose(1, 2).contiguous()
            if inference_params is not None:
                _, conv_state = inference_params.key_value_memory_dict[self.layer_idx]
                if qkv_t.shape[-1] >= self.d_conv:
                    window = qkv_t[..., -self.d_conv :]
                else:
                    pad = self.d_conv - qkv_t.shape[-1]
                    window = F.pad(qkv_t, (pad, 0))
                conv_state.copy_(window)
            return conv_out

        _, conv_state = inference_params.key_value_memory_dict[self.layer_idx]
        if qkv.shape[1] != 1:
            raise ValueError("Incremental decoding expects seqlen=1 inputs")
        qkv_token = qkv[:, 0, :]
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = qkv_token
        weight = self.conv1d.weight.view(self.conv1d.weight.shape[0], -1)
        conv = (conv_state * weight.unsqueeze(0)).sum(dim=-1)
        if self.conv1d.bias is not None:
            conv = conv + self.conv1d.bias
        return conv.unsqueeze(1)
