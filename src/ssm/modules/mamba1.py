from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from ssm import ops


class Mamba1(nn.Module):
    """Selective SSM (Mamba v1) block backed by the reference ops.

    Args:
        d_model: Model embedding dimension.
        d_state: State size used by the selective scan dynamics.
        d_conv: Depthwise convolution kernel width.
        expand: Multiplicative expansion factor for the intermediate channel count.
        dt_min: Lower bound for the time-step initialization (kept for API parity).
        dt_max: Upper bound for the time-step initialization (kept for API parity).
        dt_init_floor: Floor value for the time-step initialization (kept for API parity).
        bias: Whether to include bias terms on the linear projections.
        conv_bias: Whether to include bias terms on the convolution path.
        use_fast_path: Unused flag reserved for fused-kernel integration.
        layer_idx: Layer index metadata for potential parameter sharing.
        device: Optional device for parameter initialization.
        dtype: Optional dtype for parameter initialization.
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
        """Initialize the reference Mamba1 block.

        Args:
            d_model: Model embedding dimension.
            d_state: State size used by the selective scan dynamics.
            d_conv: Depthwise convolution kernel width.
            expand: Multiplicative expansion factor for the intermediate channel count.
            dt_min: Lower bound for the time-step initialization (kept for API parity).
            dt_max: Upper bound for the time-step initialization (kept for API parity).
            dt_init_floor: Floor value for the time-step initialization (kept for API parity).
            bias: Whether to include bias terms on the linear projections.
            conv_bias: Whether to include bias terms on the convolution path.
            use_fast_path: Unused flag reserved for fused-kernel integration.
            layer_idx: Layer index metadata for potential parameter sharing.
            device: Optional device for parameter initialization.
            dtype: Optional dtype for parameter initialization.
        """
        super().__init__()
        del dt_min, dt_max, dt_init_floor  # Documented for parity with fused kernels.

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.inner_dim = expand * d_model
        self.layer_idx = layer_idx
        self.use_fast_path = use_fast_path

        factory_kwargs: dict[str, Any] = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        self.in_proj = nn.Linear(d_model, 2 * self.inner_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=bias, **factory_kwargs)
        self.dt_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True, **factory_kwargs)

        conv_weight = torch.randn(self.inner_dim, d_conv, **factory_kwargs) * 0.02
        self.conv_weight = nn.Parameter(conv_weight)
        if conv_bias:
            self.conv_bias = nn.Parameter(torch.zeros(self.inner_dim, **factory_kwargs))
        else:
            self.register_parameter("conv_bias", None)

        # Selective scan parameterisation (broadcast to ops contract).
        self.A_log = nn.Parameter(torch.zeros(self.inner_dim, d_state, **factory_kwargs))
        self.B = nn.Parameter(torch.randn(self.inner_dim, d_state, **factory_kwargs) * 0.02)
        self.C = nn.Parameter(torch.randn(self.inner_dim, d_state, **factory_kwargs) * 0.02)
        self.D = nn.Parameter(torch.ones(self.inner_dim, **factory_kwargs))
        self.dt_bias = nn.Parameter(torch.zeros(self.inner_dim, **factory_kwargs))

    def forward(
        self, hidden_states: torch.Tensor, inference_params=None, **kwargs
    ) -> torch.Tensor:
        """Compute the reference Mamba1 forward pass.

        Args:
            hidden_states: torch.Tensor shaped ``(batch, seqlen, d_model)``.
            inference_params: Optional inference helper parameters (unused for the reference path).
            **kwargs: Additional keyword arguments kept for API compatibility.

        Returns:
            torch.Tensor: Output tensor with the same shape as ``hidden_states``.

        Raises:
            ValueError: If ``hidden_states`` does not have rank 3 with ``d_model`` features.
        """

        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.d_model:
            raise ValueError("hidden_states must have shape (B, L, d_model).")

        projected, gate = self.in_proj(hidden_states).chunk(2, dim=-1)
        projected = cast(torch.Tensor, projected)
        gate = cast(torch.Tensor, gate)

        conv_input = projected.permute(0, 2, 1)
        conv_out = ops.dw_causal_conv(
            conv_input,
            self.conv_weight,
            bias=self.conv_bias,
            activation="silu",
        ).permute(0, 2, 1)

        delta = self.dt_proj(conv_out).permute(0, 2, 1)
        u = conv_out.permute(0, 2, 1)
        z = gate.permute(0, 2, 1)

        A = -torch.exp(self.A_log)
        scan_out = ops.selective_scan(
            u=u,
            delta=delta,
            A=A,
            B=self.B,
            C=self.C,
            D=self.D,
            z=z,
            dt_bias=self.dt_bias,
            softplus=True,
        )
        if isinstance(scan_out, tuple):
            output = scan_out[0]
        else:
            output = scan_out

        output = cast(torch.Tensor, output).permute(0, 2, 1)
        return self.out_proj(output)

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate decoding cache tensors for streaming inference.

        Args:
            batch_size: Maximum batch size expected during decode.
            max_seqlen: Upper bound on the decoded sequence length. Unused by the reference path.
            dtype: Optional dtype override for the cache tensors.
            **kwargs: Additional keyword arguments accepted for API parity.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(conv_state, ssm_state)`` where ``conv_state``
            is shaped ``(batch_size, expand * d_model, d_conv)`` and ``ssm_state`` is shaped
            ``(batch_size, expand * d_model, d_state)``.
        """

        del max_seqlen  # Shape-independent for the reference implementation.

        param = next(self.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        cache_dtype = dtype or (
            param.dtype if param is not None else torch.get_default_dtype()
        )

        conv_state = torch.zeros(
            batch_size,
            self.inner_dim,
            self.d_conv,
            device=device,
            dtype=cache_dtype,
        )
        ssm_state = torch.zeros(
            batch_size,
            self.inner_dim,
            self.d_state,
            device=device,
            dtype=cache_dtype,
        )
        return conv_state, ssm_state

