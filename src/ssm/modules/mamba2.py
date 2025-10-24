from __future__ import annotations

import torch
from torch import nn


class Mamba2(nn.Module):
    """State Space Dual (Mamba v2) block.

    This is a design-first stub. The implementation is provided by fused ops
    in `ssm.ops` once available. Until then, the forward raises
    NotImplementedError to enforce contract clarity.

    Args:
        d_model: Model embedding dimension.
        d_state: Per-head SSM state dimension.
        d_conv: Local depthwise conv width (causal).
        headdim: Head dimension for SSD formulation.
        expand: Block expansion factor. Inner dim is `expand * d_model`.
        ngroups: Number of SSM groups for B/C branches.
        d_ssm: If set, number of inner dims using SSM; remainder uses gated MLP.
        chunk_size: SSD chunk length for chunked scan.
        bias: If True, enables bias in linear projections.
        conv_bias: If True, enables bias in the depthwise conv.
        layer_idx: Optional layer index for inference cache addressing.
        device: Optional device to initialize parameters on.
        dtype: Optional dtype for parameter initialization.

    Shape:
        - Input: `hidden_states` of shape `(B, L, D)` where `D == d_model`.
        - Output: tensor of shape `(B, L, D)`.

    Varlen:
        - `seq_idx`: `(1, total_tokens)` int32 mapping positions to batch elements.
        - `cu_seqlens`: `(B + 1,)` int32 cumulative lengths for ragged batches.

    Raises:
        NotImplementedError: Always, until an op backend is provided.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        headdim: int = 64,
        expand: int = 2,
        ngroups: int = 1,
        d_ssm: int | None = None,
        chunk_size: int = 256,
        bias: bool = False,
        conv_bias: bool = True,
        layer_idx: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.headdim = headdim
        self.expand = expand
        self.ngroups = ngroups
        self.d_ssm = d_ssm
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_params=None,
        seq_idx: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Apply the Mamba2 block.

        Args:
            hidden_states: Input tensor `(B, L, D)`.
            inference_params: Optional decoding cache handle; if provided, the
                forward may execute in step mode.
            seq_idx: Optional varlen index mapping positions to batch samples.
            cu_seqlens: Optional cumulative sequence lengths for varlen batches.
            **kwargs: Reserved for future options.

        Returns:
            torch.Tensor: Output tensor `(B, L, D)`.

        Raises:
            NotImplementedError: Implementation to be supplied in ops backends.
        """
        raise NotImplementedError("Mamba2.forward is not implemented in the scaffold.")

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype | None = None, **kwargs):
        """Allocate decoding cache tensors for step-wise inference.

        Args:
            batch_size: Max batch size expected during decoding.
            max_seqlen: Max target sequence length for decoding.
            dtype: Optional dtype for states; if None, selects per-backend default.
            **kwargs: Reserved for backend-specific knobs.

        Returns:
            Any: Backend-specific structure describing the cache (e.g., tuple or dict).

        Note:
            This stub defines the expected contract but does not allocate tensors.
            Implementations must return real structures matching the documented shapes.
        """
        raise NotImplementedError("Mamba2.allocate_inference_cache is not implemented in the scaffold.")
