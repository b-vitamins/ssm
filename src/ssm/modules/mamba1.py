from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from ssm import ops


def _compute_dtype(tensor: torch.Tensor) -> torch.dtype:
    """Return the accumulation dtype for a given tensor."""

    if tensor.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return tensor.dtype


def _linear_1x1(linear: nn.Linear, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Apply a linear layer to channel-first inputs without extra transposes."""

    weight = linear.weight.to(dtype)
    bias = linear.bias.to(dtype) if linear.bias is not None else None
    out = torch.einsum("oc,bcl->bol", weight, x)
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    return out


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

        self.in_proj = nn.Linear(
            d_model, 2 * self.inner_dim, bias=bias, **factory_kwargs
        )
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=bias, **factory_kwargs)
        self.dt_proj = nn.Linear(
            self.inner_dim, self.inner_dim, bias=True, **factory_kwargs
        )

        conv_weight = torch.randn(self.inner_dim, d_conv, **factory_kwargs) * 0.02
        self.conv_weight = nn.Parameter(conv_weight)
        if conv_bias:
            self.conv_bias = nn.Parameter(torch.zeros(self.inner_dim, **factory_kwargs))
        else:
            self.register_parameter("conv_bias", None)

        # Selective scan parameterisation (broadcast to ops contract).
        self.A_log = nn.Parameter(
            torch.zeros(self.inner_dim, d_state, **factory_kwargs)
        )
        self.B = nn.Parameter(
            torch.randn(self.inner_dim, d_state, **factory_kwargs) * 0.02
        )
        self.C = nn.Parameter(
            torch.randn(self.inner_dim, d_state, **factory_kwargs) * 0.02
        )
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

        if isinstance(inference_params, tuple):
            conv_state, ssm_state = inference_params
            return self._step_tokens(hidden_states, conv_state, ssm_state)

        if inference_params is not None:
            return self._forward_cached(hidden_states, inference_params)

        projected, gate = self.in_proj(hidden_states).chunk(2, dim=-1)
        projected = cast(torch.Tensor, projected)
        gate = cast(torch.Tensor, gate)

        compute_dtype = _compute_dtype(hidden_states)

        conv_input = projected.permute(0, 2, 1)
        conv_out = ops.dw_causal_conv(
            conv_input,
            self.conv_weight,
            bias=self.conv_bias,
            activation="silu",
        ).to(compute_dtype)

        delta = _linear_1x1(self.dt_proj, conv_out, compute_dtype)
        u = conv_out
        z = gate.permute(0, 2, 1).to(compute_dtype)

        A = -torch.exp(self.A_log.to(compute_dtype))
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
            output = cast(torch.Tensor, scan_out[0])
        else:
            output = cast(torch.Tensor, scan_out)

        output = output.permute(0, 2, 1)
        proj_input = output.to(self.out_proj.weight.dtype)
        projected_out = self.out_proj(proj_input)
        return projected_out.to(hidden_states.dtype)

    def _forward_cached(
        self, hidden_states: torch.Tensor, inference_params
    ) -> torch.Tensor:
        if self.layer_idx is None:
            raise RuntimeError("inference requires layer_idx to be set")
        if not hasattr(inference_params, "key_value_memory_dict"):
            inference_params.key_value_memory_dict = {}
        cache_dict = inference_params.key_value_memory_dict
        batch = hidden_states.shape[0]
        batch_start = getattr(inference_params, "batch_size_offset", 0)
        batch_end = batch_start + batch
        target_batch = max(batch_end, 0)
        cache = cache_dict.get(self.layer_idx)
        if (
            cache is None
            or cache[0].shape[0] < target_batch
            or cache[1].shape[0] < target_batch
        ):
            cache = self.allocate_inference_cache(
                batch_size=target_batch,
                max_seqlen=getattr(
                    inference_params,
                    "max_seqlen",
                    hidden_states.shape[1],
                ),
                dtype=hidden_states.dtype,
            )
            cache_dict[self.layer_idx] = cache

        conv_cache, ssm_cache = cache
        conv_slice = conv_cache[batch_start:batch_end]
        ssm_slice = ssm_cache[batch_start:batch_end]

        return self._step_tokens(hidden_states, conv_slice, ssm_slice)

    def _step_tokens(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.shape[1] == 0:
            return hidden_states.new_zeros(
                hidden_states.shape[0], 0, hidden_states.shape[-1]
            )

        outputs: list[torch.Tensor] = []
        for t in range(hidden_states.shape[1]):
            token = hidden_states[:, t : t + 1]
            out, _, _ = self.step(token, conv_state, ssm_state)
            outputs.append(out)
        return torch.cat(outputs, dim=1)

    def step(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError("step expects inputs of shape (B, 1, d_model)")

        batch = hidden_states.shape[0]
        if conv_state.shape[0] != batch or conv_state.shape[1] != self.inner_dim:
            raise ValueError("conv_state shape does not match batch or inner dim")
        if ssm_state.shape[0] != batch or ssm_state.shape[1] != self.inner_dim:
            raise ValueError("ssm_state shape does not match batch or inner dim")

        projected, gate = self.in_proj(hidden_states).chunk(2, dim=-1)
        projected = projected.squeeze(1)
        gate = gate.squeeze(1)

        compute_dtype = _compute_dtype(hidden_states)

        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = projected.to(conv_state.dtype)

        conv_history = conv_state.to(compute_dtype)
        kernel = self.conv_weight.to(compute_dtype).view(1, self.inner_dim, self.d_conv)
        conv_out = (conv_history * kernel).sum(dim=-1)
        if self.conv_bias is not None:
            conv_out = conv_out + self.conv_bias.to(compute_dtype)
        conv_out = F.silu(conv_out)

        delta = F.linear(
            conv_out,
            self.dt_proj.weight.to(compute_dtype),
            self.dt_proj.bias.to(compute_dtype),
        )
        A = -torch.exp(self.A_log.to(compute_dtype))
        z = gate.to(compute_dtype)

        output = ops.selective_state_step(
            ssm_state,
            conv_out,
            delta,
            A,
            self.B,
            self.C,
            self.D,
            z=z,
            dt_bias=self.dt_bias,
            softplus=True,
        )

        proj_input = output.to(self.out_proj.weight.dtype)
        projected_out = self.out_proj(proj_input)
        return projected_out.unsqueeze(1).to(hidden_states.dtype), conv_state, ssm_state

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
