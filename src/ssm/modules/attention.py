from __future__ import annotations

import torch
from torch import nn


class MHA(nn.Module):
    """Multi-Head Attention (optional hybrid component).

    Design-only stub mirroring a standard MHA with optional KV-cache update for
    decoding. Not implemented in the scaffold.

    Args:
        embed_dim: Input/output embedding dimension.
        num_heads: Number of query heads.
        num_heads_kv: Number of key/value heads for MQA/GQA; defaults to `num_heads`.
        head_dim: Optional head dimension; if None, `embed_dim // num_heads`.
        mlp_dim: Optional fused MLP dim to pack with attention.
        qkv_proj_bias: If True, use bias for QKV projection.
        out_proj_bias: If True, use bias for output projection.
        softmax_scale: Optional attention scaling factor; if None, `1/sqrt(head_dim)`.
        causal: If True, apply causal masking.
        layer_idx: Optional layer index for inference cache addressing.
        d_conv: Optional depthwise conv width for pre-attention conv.
        rotary_emb_dim: Optional rotary embedding dimension (if used).
        rotary_emb_base: Rotary base.
        rotary_emb_interleaved: Whether to interleave rotary dims.
        device: Optional device.
        dtype: Optional dtype.

    Raises:
        NotImplementedError: Always, until implemented.
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
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads
        self.mlp_dim = mlp_dim
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.layer_idx = layer_idx
        self.d_conv = d_conv
        self.rotary_emb_dim = rotary_emb_dim
        self.rotary_emb_base = rotary_emb_base
        self.rotary_emb_interleaved = rotary_emb_interleaved

    def forward(self, x: torch.Tensor, inference_params=None, **kwargs) -> torch.Tensor:
        """Apply attention.

        Args:
            x: Input tensor `(B, L, D)`.
            inference_params: Optional KV-cache handle for decoding.
            **kwargs: Reserved for future options (e.g., attention mask).

        Returns:
            torch.Tensor: Output tensor `(B, L, D)`.

        Raises:
            NotImplementedError: Implementation to be provided later.
        """
        raise NotImplementedError("MHA.forward is not implemented in the scaffold.")

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype | None = None):
        """Allocate KV cache for decoding.

        Args:
            batch_size: Max batch size for decoding.
            max_seqlen: Max target sequence length for decoding.
            dtype: Dtype for the cache tensors.

        Returns:
            torch.Tensor: KV cache tensor with shape `(B, max_seqlen, 2, nheads_kv, head_dim)`.

        Raises:
            NotImplementedError: Implementation to be provided later.
        """
        raise NotImplementedError("MHA.allocate_inference_cache is not implemented in the scaffold.")
