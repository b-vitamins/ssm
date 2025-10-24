from __future__ import annotations

import torch
from torch import nn


class Mamba1(nn.Module):
    """Selective SSM (Mamba v1) block.

    This is a design-first stub. The implementation is provided by fused ops
    in `ssm.ops` once available. Until then, the forward raises
    NotImplementedError to enforce contract clarity.

    Args:
        d_model: Model embedding dimension.
        d_state: Internal SSM state dimension expansion.
        d_conv: Local depthwise conv width (causal).
        expand: Block expansion factor. Inner dim is `expand * d_model`.
        dt_min: Lower bound for softplus(dt_bias) at init (documentation only).
        dt_max: Upper bound for softplus(dt_bias) at init (documentation only).
        dt_init_floor: Min floor for dt during initialization (documentation only).
        bias: If True, enables bias in linear projections.
        conv_bias: If True, enables bias in the depthwise conv.
        use_fast_path: If True, dispatches to fused kernel backend when available.
        layer_idx: Optional layer index for inference cache addressing.
        device: Optional device to initialize parameters on.
        dtype: Optional dtype for parameter initialization.

    Shape:
        - Input: `hidden_states` of shape `(B, L, D)` where `D == d_model`.
        - Output: tensor of shape `(B, L, D)`.

    Raises:
        NotImplementedError: Always, until an op backend is provided.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        use_fast_path: bool = True,
        layer_idx: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.layer_idx = layer_idx
        self.use_fast_path = use_fast_path

        # Parameters are intentionally not allocated here to avoid implying a
        # specific implementation. Implementations will define and register
        # parameters when added.

    def forward(
        self, hidden_states: torch.Tensor, inference_params=None, **kwargs
    ) -> torch.Tensor:
        """Apply the Mamba1 block.

        Args:
            hidden_states: Input tensor `(B, L, D)`.
            inference_params: Optional decoding cache handle; if provided, the
                forward may execute in step mode.
            **kwargs: Reserved for future options (e.g., z gating toggles).

        Returns:
            torch.Tensor: Output tensor `(B, L, D)`.

        Raises:
            NotImplementedError: Implementation to be supplied in ops backends.
        """
        raise NotImplementedError("Mamba1.forward is not implemented in the scaffold.")

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        """Allocate decoding cache tensors for step-wise inference.

        Args:
            batch_size: Max batch size expected during decoding.
            max_seqlen: Max target sequence length for decoding.
            dtype: Optional dtype for states; if None, selects per-backend default.
            **kwargs: Reserved for backend-specific knobs.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: `(conv_state, ssm_state)` where
            - `conv_state`: `(B, expand * d_model, d_conv)` state for the depthwise conv.
            - `ssm_state`: `(B, expand * d_model, d_state)` state for the SSM.

        Note:
            This stub defines the expected shapes but does not allocate tensors.
            Implementations must return real tensors matching the above shapes.
        """
        raise NotImplementedError(
            "Mamba1.allocate_inference_cache is not implemented in the scaffold."
        )
